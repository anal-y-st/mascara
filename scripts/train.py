import argparse, yaml, importlib, importlib.util, os, sys
import torch
from models.registry import get_model
from trainers.trainer import train_and_evaluate
from utils.cli import override_config

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
    """Parse a string value to its appropriate type (int, float, bool, or str)"""
    value_str = value_str.strip()
    
    # Handle boolean values
    if value_str.lower() in ('true', 'false'):
        return value_str.lower() == 'true'
    
    # Handle None
    if value_str.lower() in ('none', 'null'):
        return None
    
    # Try to parse as int
    try:
        if '.' not in value_str and 'e' not in value_str.lower():
            return int(value_str)
    except ValueError:
        pass
    
    # Try to parse as float
    try:
        return float(value_str)
    except ValueError:
        pass
    
    # Handle lists/arrays (basic support for comma-separated values)
    if value_str.startswith('[') and value_str.endswith(']'):
        inner = value_str[1:-1].strip()
        if inner:
            items = [parse_value(item.strip()) for item in inner.split(',')]
            return items
        return []
    
    # Return as string if nothing else works
    return value_str

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Chemin du YAML de config")
    parser.add_argument("--loaders", required=True, help="module:function retournant (train_loader, val_loader, stats)")
    parser.add_argument("--set", action="append", default=[], help="Override (ex: --set training.lr=1e-4)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Parse overrides with proper type conversion
    pairs = []
    for s in args.__dict__.get("set", []) or []:
        if "=" not in s:
            raise ValueError(f"Invalid --set '{s}', expected key=value")
        k, v = s.split("=", 1)
        key = k.strip()
        parsed_value = parse_value(v.strip())
        pairs.append((key, parsed_value))
        print(f"Override: {key} = {parsed_value} (type: {type(parsed_value).__name__})")
    
    cfg = override_config(cfg, pairs)

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
