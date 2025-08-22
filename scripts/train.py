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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Chemin du YAML de config")
    parser.add_argument("--loaders", required=True, help="module:function retournant (train_loader, val_loader, stats)")
    parser.add_argument("--set", action="append", default=[], help="Override (ex: --set training.lr=1e-4)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Parse overrides
    pairs = []
    for s in args.__dict__.get("set", []) or []:
        if "=" not in s:
            raise ValueError(f"Invalid --set '{s}', expected key=value")
        k, v = s.split("=", 1)
        pairs.append((k.strip(), v.strip()))
    cfg = override_config(cfg, pairs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(cfg["model"]).to(device)

    get_loaders = import_function(args.loaders)
    train_loader, val_loader, stats = get_loaders(cfg)
    input_mean = stats.get("input_mean", 0.0)
    input_std  = stats.get("input_std", 1.0)
    label_mean = stats.get("label_mean", 0.0)
    label_std  = stats.get("label_std", 1.0)

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
