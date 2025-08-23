import argparse, yaml, importlib, importlib.util, os, sys, logging
import torch
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
        logger=logger,  # Pass logger to train_and_evaluate
        **cfg["training"]
    )
    
    logger.info("=== TRAINING COMPLETED ===")
    logger.info(f"Best validation metrics: {best}")
    
    if args.log_file:
        logger.info(f"Full logs saved to: {args.log_file}")

if __name__ == "__main__":
    main()
