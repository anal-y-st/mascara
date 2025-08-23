import os, math, json
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from losses import R2Loss, ssim_loss
from metrics.evaluation import evaluate_numpy
from utils.visualization import plot_curves, save_prediction
from utils.reporting import save_report

# Remplacer la fonction evaluate_model dans trainers/trainer.py par ceci :

def evaluate_model(model, loader, device, label_mean=None, label_std=None, logger=None):
    """Version corrigée qui évite les NaN"""
    import logging
    if logger is None:
        logger = logging.getLogger(__name__)
    
    model.eval()
    y_trues, y_preds = [], []
    
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            
            # Vérifier les NaN dans la prédiction du modèle
            if torch.isnan(pred).any():
                logger.warning("NaN detected in model output!")
                
            # Rescale to original label units if normalized (SUR LES TENSEURS)
            if (label_mean is not None) and (label_std is not None):
                pred_res = (pred.cpu().numpy() * label_std) + label_mean
                yb_res = (yb.cpu().numpy() * label_std) + label_mean
            else:
                pred_res = pred.cpu().numpy()
                yb_res = yb.cpu().numpy()
            
            # Vérifier les NaN après dénormalisation
            if np.isnan(pred_res).any() or np.isnan(yb_res).any():
                logger.warning("NaN detected after denormalization!")
                logger.warning(f"label_mean={label_mean}, label_std={label_std}")
                
            y_preds.append(pred_res.ravel())
            y_trues.append(yb_res.ravel())
    
    y_pred = np.concatenate(y_preds)
    y_true = np.concatenate(y_trues)
    
    # Retour au format du codebase
    from metrics.evaluation import evaluate_numpy
    return evaluate_numpy(y_true, y_pred)

def train_and_evaluate(model, train_loader, val_loader, device, input_mean, input_std, label_mean, label_std,
                       epochs=30, outdir='results_advanced', alpha=0.5, beta=0.3, gamma=0.2, lr=1e-4,
                       weight_decay=1e-2, amp=True, clip_grad=1.0, scheduler=None, early_stopping=None, seed=1337):
    os.makedirs(outdir, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed)

    huber = nn.HuberLoss()
    r2loss = R2Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if scheduler and scheduler.get("name","none") == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max',
            factor=scheduler.get("factor",0.5),
            patience=scheduler.get("patience",4)
        )
    else:
        sched = None

    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    best = {"epoch": 0, "r2": -1e9, "mae": float("inf"), "rmse": float("inf")}
    history = []
    es_counter = 0
    patience = (early_stopping or {}).get("patience", 10) if (early_stopping or {}).get("enabled", False) else None

    for epoch in range(1, epochs+1):
        model.train()
        running_loss, n = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for xb, yb in pbar:
            xb = xb.to(device).float()
            yb = yb.to(device).float()
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                pr = model(xb)
                l_h = huber(pr, yb)
                l_r2 = r2loss(pr, yb)
                l_ssim = ssim_loss(pr, yb)
                loss = alpha * l_h + beta * l_r2 + gamma * l_ssim
            scaler.scale(loss).backward()
            if clip_grad and clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item())
            n += 1
            pbar.set_postfix(loss=running_loss/max(n,1))

        train_loss = running_loss / max(n,1)

        # Validation
        val_metrics = evaluate_model(model, val_loader, device, label_mean=label_mean, label_std=label_std)
        val_r2, val_mae, val_rmse = val_metrics["r2"], val_metrics["mae"], val_metrics["rmse"]

        if sched is not None:
            sched.step(val_r2)

        # Track history
        hist = dict(epoch=epoch, train_loss=train_loss, val_r2=val_r2, val_mae=val_mae, val_rmse=val_rmse)
        history.append(hist)

        # Save per-epoch visualization on a small sample
        try:
            xb_vis, yb_vis = next(iter(val_loader))
            xb_vis = xb_vis.to(device).float()
            yb_vis = yb_vis.to(device).float()
            with torch.no_grad():
                pr_vis = model(xb_vis)
            pr_np = (pr_vis[0,0].detach().cpu().numpy() * label_std) + label_mean
            gt_np = (yb_vis[0,0].detach().cpu().numpy() * label_std) + label_mean
            save_prediction(os.path.join(outdir, f'epoch_{epoch:03d}.png'), gt_np, pr_np)
        except Exception as e:
            # Silent best-effort
            pass

        # Save best
        if val_r2 > best["r2"]:
            best.update(epoch=epoch, r2=val_r2, mae=val_mae, rmse=val_rmse)
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'r2': best["r2"]
            }, os.path.join(outdir, 'best_model.pth'))
            es_counter = 0
        else:
            if patience is not None:
                es_counter += 1

        # Early stopping
        if patience is not None and es_counter >= patience:
            print(f"Early stopping at epoch {epoch} (no R2 improvement for {patience} epochs).")
            break

        print(f"Epoch {epoch}/{epochs} | TrainLoss={train_loss:.4f} | "
              f"ValR2={val_r2:.4f} | ValMAE={val_mae:.6f} | ValRMSE={val_rmse:.6f}")

    # Curves & Report
    plot_curves(history, outdir)
    save_report(outdir, {
        "model": getattr(model, "__class__", type(model)).__name__,
    }, best["epoch"], {"r2": best["r2"], "mae": best["mae"], "rmse": best["rmse"]})

    print("Training finished. Best:", best)
    return best, history
