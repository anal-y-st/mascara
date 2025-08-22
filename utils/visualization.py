import os
import json
import numpy as np
import matplotlib.pyplot as plt

def plot_curves(history, outdir):
    os.makedirs(outdir, exist_ok=True)
    epochs = [h['epoch'] for h in history]

    # Loss
    plt.figure()
    plt.plot(epochs, [h['train_loss'] for h in history], label='train_loss')
    plt.plot(epochs, [h['val_r2'] for h in history], label='val_r2')
    plt.plot(epochs, [h['val_mae'] for h in history], label='val_mae')
    plt.plot(epochs, [h['val_rmse'] for h in history], label='val_rmse')
    plt.xlabel("epoch"); plt.ylabel("value"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "curves.png"))
    plt.close()

    with open(os.path.join(outdir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

def save_prediction(figpath, gt_np, pr_np):
    diff = np.abs(pr_np - gt_np)
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1,3,figsize=(12,4))
    im0 = axs[0].imshow(gt_np); axs[0].set_title('GT'); axs[0].axis('off')
    im1 = axs[1].imshow(pr_np); axs[1].set_title('Pred'); axs[1].axis('off')
    im2 = axs[2].imshow(diff); axs[2].set_title('Abs diff'); axs[2].axis('off')
    plt.tight_layout()
    plt.savefig(figpath)
    plt.close(fig)
