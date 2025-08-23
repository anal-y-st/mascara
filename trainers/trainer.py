# Remplacer la fonction train_and_evaluate dans trainers/trainer.py :

import os, math, json
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from losses import R2Loss, ssim_loss
from utils.visualization import plot_curves, save_prediction
from utils.reporting import save_report

def evaluate_model_exact(model, loader, device, label_mean=None, label_std=None):
    """Version EXACTE de votre fonction evaluate_model"""
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    
    model.eval()
    y_trues, y_preds = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            # rescale to original label units if normalized
            if (label_mean is not None) and (label_std is not None):
                pred_res = (pred.cpu().numpy() * label_std) + label_mean
                yb_res = (yb.cpu().numpy() * label_std) + label_mean
            else:
                pred_res = pred.cpu().numpy()
                yb_res = yb.cpu().numpy()
            y_preds.append(pred_res.ravel())
            y_trues.append(yb_res.ravel())
    y_pred = np.concatenate(y_preds)
    y_true = np.concatenate(y_trues)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r2, mae, rmse

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
    best_r2 = -1e9
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
                pred = model(xb)
                l_h = huber(pred, yb)
                l_r2 = r2loss(pred, yb)
                l_ssim = ssim_loss(pred, yb)
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

        # Validation - UTILISE VOTRE FONCTION EXACTE
        val_r2, val_mae, val_rmse = evaluate_model_exact(model, val_loader, device, 
                                                         label_mean=label_mean, label_std=label_std)

        if sched is not None:
            sched.step(val_r2)

        # Track history
        hist = dict(epoch=epoch, train_loss=train_loss, val_r2=val_r2, val_mae=val_mae, val_rmse=val_rmse)
        history.append(hist)

        # Save per-epoch visualization - EXACTEMENT COMME DANS VOTRE CODE
        try:
            xb_vis, yb_vis = next(iter(val_loader))
            xb_vis = xb_vis.to(device).float()
            yb_vis = yb_vis.to(device).float()
            with torch.no_grad():
                pr_vis = model(xb_vis)
            pr_np = (pr_vis[0,0].cpu().numpy() * label_std) + label_mean
            gt_np = (yb_vis[0,0].cpu().numpy() * label_std) + label_mean
            save_prediction(os.path.join(outdir, f'epoch_{epoch:03d}.png'), gt_np, pr_np)
        except Exception as e:
            pass

        # Save best - COMME DANS VOTRE CODE
        if val_r2 > best_r2:
            best_r2 = val_r2
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'r2': best_r2
            }, os.path.join(outdir, 'best_model.pth'))
            print(f"Saved best model (R2={best_r2:.4f})")
            es_counter = 0
        else:
            if patience is not None:
                es_counter += 1

        # Early stopping
        if patience is not None and es_counter >= patience:
            print(f"Early stopping at epoch {epoch} (no R2 improvement for {patience} epochs).")
            break

        # PRINT EXACTEMENT COMME DANS VOTRE CODE
        print(f"Epoch {epoch}/{epochs} | TrainLoss={train_loss:.4f} | ValR2={val_r2:.4f} | ValMAE={val_mae:.4f} | ValRMSE={val_rmse:.4f}")

    # Curves & Report
    plot_curves(history, outdir)
    save_report(outdir, {
        "model": getattr(model, "__class__", type(model)).__name__,
    }, history[-1]["epoch"], {"r2": best_r2, "mae": val_mae, "rmse": val_rmse})

    print("Training finished. Best Val R2:", best_r2)
    
    # Retour compatible avec le codebase
    best = {"epoch": history[-1]["epoch"], "r2": best_r2, "mae": val_mae, "rmse": val_rmse}
    return best, history
