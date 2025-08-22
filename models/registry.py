def get_model(cfg_model: dict):
    name = cfg_model.get("name")
    if name == "AdvancedSwinUNet":
        from .swin_unet import AdvancedSwinUNet
        return AdvancedSwinUNet(
            in_ch=cfg_model.get("in_ch", 18),
            out_ch=cfg_model.get("out_ch", 1),
            embed_dim=cfg_model.get("embed_dim", 120),
            depth=cfg_model.get("depth", 3),
            heads=cfg_model.get("heads", 5),
            img_size=tuple(cfg_model.get("img_size", (128, 128)))
        )
    raise ValueError(f"Unknown model name: {name}")

def list_models():
    return ["AdvancedSwinUNet"]
