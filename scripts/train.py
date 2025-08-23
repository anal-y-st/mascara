import argparse, yaml, importlib, importlib.util, os, sys, logging
import torch
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tqdm import tqdm
from models.registry import get_model
from trainers.trainer import train_and_evaluate
from utils.cli import set_by_path

def setup_logging(log_level=logging.INFO, log_file=None):
    """Setup logging configuration"""
    handlers = [logging.StreamHandler()]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers,
        force=True
    )
    
    return logging.getLogger(__name__)

def import_function(dotted: str):
    if ":" in dotted:
        module_name, func_name = dotted.split(":", 1)
    elif "#" in dotted:
        module_name, func_name = dotted.split("#", 1)
    else:
        raise ValueError("Use format 'module.submodule:function_name'")
    mod = importlib.import_module(module_name)
    fn = getattr(mod, func_name)
    return fn

def parse_value(value_str):
    """Parse a string value to its appropriate type with robust type checking"""
    # Handle non-string inputs
    if not isinstance(value_str, str):
        return value_str
    
    value_str = value_str.strip()
    
    # Empty string
    if not value_str:
        return ""
    
    # Handle boolean values (case insensitive)
    lower_val = value_str.lower()
    if lower_val in ('true', 'false', 'yes', 'no'):
        return lower_val in ('true', 'yes')
    
    # Handle None/null
    if lower_val in ('none', 'null', 'nil'):
        return None
    
    # Try to parse as integer first (but not if it contains decimal point or scientific notation)
    if '.' not in value_str and 'e' not in lower_val and 'inf' not in lower_val:
        try:
            int_val = int(value_str)
            # Verify the string representation matches
            if str(int_val) == value_str or (value_str.startswith('-') and str(int_val) == value_str):
                return int_val
        except (ValueError, OverflowError):
            pass
    
    # Try to parse as float (including scientific notation)
    try:
        float_val = float(value_str)
        # Check for special float values
        if str(float_val).lower() in ('inf', '-inf', 'nan'):
            return float_val
        return float_val
    except (ValueError, OverflowError):
        pass
    
    # Handle quoted strings (remove outer quotes)
    if ((value_str.startswith('"') and value_str.endswith('"')) or 
        (value_str.startswith("'") and value_str.endswith("'"))):
        return value_str[1:-1]
    
    # Return as string (default case)
    return value_str

def ensure_numeric_config(cfg):
    """Recursively ensure that numeric values in config are properly typed"""
    if isinstance(cfg, dict):
        for key, value in cfg.items():
            if isinstance(value, str):
                # Try to parse string values that should be numeric
                parsed = parse_value(value)
                if parsed != value:  # Only update if parsing changed the value
                    cfg[key] = parsed
            elif isinstance(value, dict):
                ensure_numeric_config(value)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, str):
                        parsed = parse_value(item)
                        if parsed != item:
                            value[i] = parsed
                    elif isinstance(item, dict):
                        ensure_numeric_config(item)
    return cfg

def evaluate_model(model, loader, device, label_mean=None, label_std=None, logger=None):
    """
    Evaluate model with robust NaN handling and progress tracking
    
    Args:
        model: PyTorch model to evaluate
        loader: DataLoader for evaluation data
        device: Device to run evaluation on
        label_mean: Mean for label denormalization (scalar)
        label_std: Std for label denormalization (scalar)
        logger: Logger instance for reporting
    
    Returns:
        tuple: (r2, mae, rmse) or dict with detailed metrics if NaN issues
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    model.eval()
    y_trues, y_preds = [], []
    
    logger.info("Starting model evaluation...")
    
    with torch.no_grad():
        # Use tqdm for progress bar
        pbar = tqdm(loader, desc="Evaluating", leave=False)
        
        for batch_idx, (xb, yb) in enumerate(pbar):
            xb = xb.to(device)
            yb = yb.to(device)
            
            # Forward pass
            try:
                pred = model(xb)
            except Exception as e:
                logger.error(f"Model forward pass failed on batch {batch_idx}: {e}")
                continue
            
            # Check for NaN in model output immediately
            if torch.isnan(pred).any():
                nan_count = torch.isnan(pred).sum().item()
                total = pred.numel()
                logger.warning(f"Batch {batch_idx}: Model output contains {nan_count}/{total} NaN values")
            
            # Rescale to original label units if normalized
            if (label_mean is not None) and (label_std is not None):
                # Convert to CPU and numpy first, then rescale
                pred_cpu = pred.cpu().numpy()
                yb_cpu = yb.cpu().numpy()
                
                pred_res = (pred_cpu * label_std) + label_mean
                yb_res = (yb_cpu * label_std) + label_mean
            else:
                pred_res = pred.cpu().numpy()
                yb_res = yb.cpu().numpy()
            
            # Check for NaN after rescaling
            if np.isnan(pred_res).any() or np.isnan(yb_res).any():
                pred_nan = np.isnan(pred_res).sum()
                true_nan = np.isnan(yb_res).sum()
                if pred_nan > 0:
                    logger.warning(f"Batch {batch_idx}: {pred_nan} NaN in rescaled predictions")
                if true_nan > 0:
                    logger.warning(f"Batch {batch_idx}: {true_nan} NaN in rescaled targets")
            
            y_preds.append(pred_res.ravel())
            y_trues.append(yb_res.ravel())
            
            # Update progress bar with batch info
            pbar.set_postfix({
                'batch': f"{batch_idx+1}/{len(loader)}",
                'pred_range': f"[{pred_res.min():.2f}, {pred_res.max():.2f}]"
            })
    
    # Concatenate all predictions and targets
    logger.info("Concatenating predictions...")
    y_pred = np.concatenate(y_preds)
    y_true = np.concatenate(y_trues)
    
    logger.info(f"Total samples: {len(y_pred)}")
    logger.info(f"Prediction range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
    logger.info(f"Target range: [{y_true.min():.4f}, {y_true.max():.4f}]")
    
    # Check for NaN values in final arrays
    pred_has_nan = np.isnan(y_pred).any()
    true_has_nan = np.isnan(y_true).any()
    
    if pred_has_nan or true_has_nan:
        pred_nan_count = np.isnan(y_pred).sum()
        true_nan_count = np.isnan(y_true).sum()
        logger.warning(f"Final arrays - Predictions: {pred_nan_count} NaN, Targets: {true_nan_count} NaN")
        
        # Create mask for valid (non-NaN) values
        valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        
        if not valid_mask.any():
            logger.error("All values are NaN! Cannot compute metrics.")
            return {'r2': float('nan'), 'mae': float('nan'), 'rmse': float('nan'), 
                    'valid_samples': 0, 'total_samples': len(y_pred)}
        
        logger.info(f"Using {valid_mask.sum()}/{len(y_pred)} valid samples for metrics")
        y_true_clean = y_true[valid_mask]
        y_pred_clean = y_pred[valid_mask]
    else:
        y_true_clean = y_true
        y_pred_clean = y_pred
    
    # Check for infinite values
    if np.isinf(y_pred_clean).any() or np.isinf(y_true_clean).any():
        logger.warning("Found infinite values, filtering them out...")
        finite_mask = np.isfinite(y_true_clean) & np.isfinite(y_pred_clean)
        if not finite_mask.any():
            logger.error("No finite values found!")
            return {'r2': float('nan'), 'mae': float('nan'), 'rmse': float('nan'),
                    'valid_samples': 0, 'total_samples': len(y_pred)}
        y_true_clean = y_true_clean[finite_mask]
        y_pred_clean = y_pred_clean[finite_mask]
    
    # Compute metrics
    logger.info("Computing evaluation metrics...")
    try:
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
        
        # R2 can fail if all targets are the same (zero variance)
        try:
            r2 = r2_score(y_true_clean, y_pred_clean)
        except ValueError as e:
            logger.warning(f"R2 computation failed: {e} (possibly zero variance in targets)")
            r2 = float('nan')
        
        logger.info(f"Metrics computed successfully:")
        logger.info(f"  R²: {r2:.6f}")
        logger.info(f"  MAE: {mae:.6f}")
        logger.info(f"  RMSE: {rmse:.6f}")
        logger.info(f"  Samples used: {len(y_true_clean)}/{len(y_pred)}")
        
        return r2, mae, rmse
    
    except Exception as e:
        logger.error(f"Error computing metrics: {e}")
        return float('nan'), float('nan'), float('nan')

def check_for_nans(tensor, name="tensor"):
    """Check if tensor contains NaN values and log warning"""
    if torch.isnan(tensor).any():
        nan_count = torch.isnan(tensor).sum().item()
        total = tensor.numel()
        logging.warning(f"{name} contains {nan_count}/{total} NaN values!")
        return True
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Chemin du YAML de config")
    parser.add_argument("--loaders", required=True, help="module:function retournant (train_loader, val_loader, stats)")
    parser.add_argument("--set", action="append", default=[], help="Override (ex: --set training.lr=1e-4)")
    parser.add_argument("--log-file", help="Chemin du fichier de log (optionnel)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Mode verbose (DEBUG)")
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(log_level, args.log_file)
    
    logger.info("=== TRAINING STARTED ===")
    logger.info(f"Config file: {args.config}")
    logger.info(f"Loaders: {args.loaders}")
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    logger.info("Configuration loaded successfully")
    
    # Ensure all numeric values in config are properly typed
    cfg = ensure_numeric_config(cfg)
    logger.debug("Configuration numeric values normalized")

    # Parse overrides with proper type conversion and validation
    if args.set:
        logger.info("Applying configuration overrides:")
    
    for s in args.__dict__.get("set", []) or []:
        if "=" not in s:
            raise ValueError(f"Invalid --set '{s}', expected key=value")
        k, v = s.split("=", 1)
        key = k.strip()
        
        # Validate key format
        if not key:
            raise ValueError(f"Empty key in --set '{s}'")
        
        parsed_value = parse_value(v.strip())
        
        # Debug output with type information
        if isinstance(parsed_value, str):
            logger.info(f"  Override: {key} = '{parsed_value}' (string)")
        elif isinstance(parsed_value, bool):
            logger.info(f"  Override: {key} = {parsed_value} (boolean)")
        elif isinstance(parsed_value, int):
            logger.info(f"  Override: {key} = {parsed_value} (integer)")
        elif isinstance(parsed_value, float):
            logger.info(f"  Override: {key} = {parsed_value} (float)")
        else:
            logger.info(f"  Override: {key} = {parsed_value} (type: {type(parsed_value).__name__})")
        
        # Apply override directly without additional parsing
        set_by_path(cfg, key, parsed_value)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    logger.info("Creating model...")
    model = get_model(cfg["model"]).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created - Total params: {num_params:,}, Trainable: {num_trainable:,}")

    logger.info("Loading data...")
    get_loaders = import_function(args.loaders)
    train_loader, val_loader, stats = get_loaders(cfg)
    
    logger.info(f"Data loaded - Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Ensure stats values are properly typed - FIXED: keep arrays as arrays
    input_mean = stats.get("input_mean", 0.0)  # Keep as array (shape: [channels])
    input_std  = stats.get("input_std", 1.0)   # Keep as array (shape: [channels])
    label_mean = float(stats.get("label_mean", 0.0))  # Scalar value
    label_std  = float(stats.get("label_std", 1.0))   # Scalar value
    
    logger.info(f"Dataset stats - Label mean: {label_mean:.4f}, Label std: {label_std:.4f}")
    if hasattr(input_mean, 'shape'):
        logger.info(f"Input mean shape: {input_mean.shape}")
    
    # Check for problematic statistics that could cause NaNs
    if hasattr(input_std, '__iter__'):
        if (input_std <= 0).any():
            logger.warning("Some input_std values are <= 0, this may cause NaN!")
    else:
        if input_std <= 0:
            logger.warning("input_std is <= 0, this may cause NaN!")
    
    if label_std <= 0:
        logger.warning("label_std is <= 0, this may cause NaN!")

    training_config = cfg.get("training", {})
    epochs = training_config.get("epochs", 10)
    logger.info(f"Starting training for {epochs} epochs...")

    best, history = train_and_evaluate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        input_mean=input_mean,
        input_std=input_std,
        label_mean=label_mean,
        label_std=label_std,
        **cfg["training"]
    )
    
    logger.info("=== TRAINING COMPLETED ===")
    logger.info(f"Best validation metrics: {best}")
    
    if args.log_file:
        logger.info(f"Full logs saved to: {args.log_file}")

if __name__ == "__main__":
    main()
