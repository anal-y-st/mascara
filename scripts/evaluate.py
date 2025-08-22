import argparse, yaml, importlib
import torch
from models.registry import get_model
from trainers.trainer import evaluate_model
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
    parser.add_argument("--config", required=True)
    parser.add_argument("--loaders", required=True, help="module:function retournant (train_loader, val_loader, stats)")
    parser.add_argument("--checkpoint", required=True, help="Chemin du .pth (best model)")
    parser.add_argument("--set", action="append", default=[], help="Override (ex: --set model.in_ch=18)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    pairs = []
    for s in args.__dict__.get("set", []) or []:
        if "=" not in s:
            raise ValueError(f"Invalid --set '{s}', expected key=value")
        k, v = s.split("=", 1)
        pairs.append((k.strip(), v.strip()))
    cfg = override_config(cfg, pairs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(cfg["model"]).to(device)

    # Load checkpoint (model_state only)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=True)

    get_loaders = import_function(args.loaders)
    _, val_loader, stats = get_loaders(cfg)
    label_mean = stats.get("label_mean", 0.0)
    label_std  = stats.get("label_std", 1.0)

    metrics = evaluate_model(model, val_loader, device, label_mean=label_mean, label_std=label_std)
    print("Evaluation metrics:", metrics)

if __name__ == "__main__":
    main()
