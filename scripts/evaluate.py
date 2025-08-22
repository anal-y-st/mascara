import argparse, yaml, importlib, importlib.util, os, sys
import torch
from models.registry import get_model
from trainers.trainer import train_and_evaluate
from utils.cli import override_config, set_by_path

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

def parse_value(value_str: str):
    """Parse a string value to its appropriate type with robust type checking"""
    if not isinstance(value_str, str):
        return value_str
    
    value_str = value_str.strip()
    
    # Empty string
    if not value_str:
        return ""
    
    # Handle boolean values (case insensitive)
    lower_val = value_str.lower()
    if lower_val in ('true', 'false', 'yes', 'no', '1', '0'):
        if lower_val in ('true', 'yes', '1'):
            return True
        elif lower_val in ('false', 'no', '0'):
            return False
    
    # Handle None/null
    if lower_val in ('none', 'null', 'nil'):
        return None
    
    # Try to parse as integer
    try:
        # Check if it looks like an integer (no decimal point, no scientific notation)
        if '.' not in value_str and 'e' not in lower_val and 'inf' not in lower_val:
            # Additional check: ensure it's actually a valid integer string
            int_val = int(value_str)
            # Verify the string representation matches (handles cases like "007" -> 7)
            if str(int_val) == value_str or (value_str.startswith('-') and str(int_val) == value_str):
                return int_val
    except (ValueError, OverflowError):
        pass
    
    # Try to parse as float
    try:
        float_val = float(value_str)
        # Check for special float values
        if str(float_val).lower() in ('inf', '-inf', 'nan'):
            return float_val
        # Verify it's a valid float representation
        return float_val
    except (ValueError, OverflowError):
        pass
    
    # Handle quoted strings (remove outer quotes)
    if ((value_str.startswith('"') and value_str.endswith('"')) or 
        (value_str.startswith("'") and value_str.endswith("'"))):
        return value_str[1:-1]
    
    # Return as string (default case)
    return value_str

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Chemin du YAML de config")
    parser.add_argument("--loaders", required=True, help="module:function retournant (train_loader, val_loader, stats)")
    parser.add_argument("--set", action="append", default=[], help="Override (ex: --set training.lr=1e-4)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Parse overrides with proper type conversion and validation
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
        type_name = type(parsed_value).__name__
        if isinstance(parsed_value, str):
            print(f"Override: {key} = '{parsed_value}' (string)")
        elif isinstance(parsed_value, bool):
            print(f"Override: {key} = {parsed_value} (boolean)")
        elif isinstance(parsed_value, int):
            print(f"Override: {key} = {parsed_value} (integer)")
        elif isinstance(parsed_value, float):
            print(f"Override: {key} = {parsed_value} (float)")
        else:
            print(f"Override: {key} = {parsed_value} (type: {type_name})")
        
        # Apply override directly without additional parsing
        set_by_path(cfg, key, parsed_value)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(cfg["model"]).to(device)

    get_loaders = import_function(args.loaders)
    train_loader, val_loader, stats = get_loaders(cfg)
    
    # Ensure stats values are properly typed
    input_mean = float(stats.get("input_mean", 0.0))
    input_std  = float(stats.get("input_std", 1.0))
    label_mean = float(stats.get("label_mean", 0.0))
    label_std  = float(stats.get("label_std", 1.0))

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

if __name__ == "__main__":
    main()
