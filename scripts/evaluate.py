import argparse, yaml, importlib, logging
import torch
from tqdm import tqdm
from models.registry import get_model
from trainers.trainer import evaluate_model
from utils.cli import set_by_path

def setup_logging(log_level=logging.INFO, log_file=None):
    """Setup logging configuration"""
    handlers = [logging.StreamHandler()]
    
    if log_file:
        import os
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--loaders", required=True, help="module:function retournant (train_loader, val_loader, stats)")
    parser.add_argument("--checkpoint", required=True, help="Chemin du .pth (best model)")
    parser.add_argument("--set", action="append", default=[], help="Override (ex: --set model.in_ch=18)")
    parser.add_argument("--log-file", help="Chemin du fichier de log (optionnel)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Mode verbose (DEBUG)")
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(log_level, args.log_file)
    
    logger.info("=== EVALUATION STARTED ===")
    logger.info(f"Config file: {args.config}")
    logger.info(f"Checkpoint: {args.checkpoint}")
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
    logger.info(f"Model created - Total params: {num_params:,}")

    # Load checkpoint (model_state only)
    logger.info("Loading checkpoint...")
    try:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=True)
        logger.info("Checkpoint loaded successfully")
        if "epoch" in ckpt:
            logger.info(f"Checkpoint from epoch: {ckpt['epoch']}")
        if "val_loss" in ckpt:
            logger.info(f"Checkpoint validation loss: {ckpt['val_loss']:.6f}")
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        raise

    logger.info("Loading data...")
    get_loaders = import_function(args.loaders)
    _, val_loader, stats = get_loaders(cfg)
    
    logger.info(f"Data loaded - Validation batches: {len(val_loader)}")
    
    # Ensure stats values are properly typed - FIXED: keep arrays as arrays
    input_mean = stats.get("input_mean", 0.0)  # Keep as array
    input_std  = stats.get("input_std", 1.0)   # Keep as array  
    label_mean = float(stats.get("label_mean", 0.0))  # Scalar value
    label_std  = float(stats.get("label_std", 1.0))   # Scalar value

    logger.info(f"Dataset stats - Label mean: {label_mean:.4f}, Label std: {label_std:.4f}")
    if hasattr(input_mean, 'shape'):
        logger.info(f"Input mean shape: {input_mean.shape}")

    logger.info("Starting evaluation...")
    metrics = evaluate_model(model, val_loader, device, 
                           label_mean=label_mean, label_std=label_std,
                           logger=logger)  # Pass logger to evaluate_model
    
    logger.info("=== EVALUATION COMPLETED ===")
    logger.info("Final evaluation metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.6f}")
        else:
            logger.info(f"  {key}: {value}")
    
    if args.log_file:
        logger.info(f"Full logs saved to: {args.log_file}")

if __name__ == "__main__":
    main()
