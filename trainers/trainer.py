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
    """Version EXACTE de votre fonction evaluate_model avec gestion des NaN"""
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    
    model.eval()
    y_trues, y_preds = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            
            # Vérifier les NaN dans les prédictions
            if torch.isnan(pred).any():
                print(f"Warning: NaN detected in model predictions!")
                # Remplacer les NaN par zéro ou une valeur par défaut
                pred = torch.where(torch.isnan(pred), torch.zeros_like(pred), pred)
            
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
    
    # Vérifier et nettoyer les NaN avant le calcul des métriques
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
    
    if not valid_mask.any():
        print("Warning: All predictions are NaN/inf, returning default metrics")
        return -1.0, float('inf'), float('inf')
    
    y_true_clean = y_true[valid_mask]
    y_pred_clean = y_pred[valid_mask]
    
    try:
        r2 = r2_score(y_true_clean, y_pred_clean)
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    except Exception as e:
        print(f"Error computing metrics: {e}")
        return -1.0, float('inf'), float('inf')
    
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

    # Correction 1: Utiliser la nouvelle API pour GradScaler
    if amp and torch.cuda.is_available():
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = None
        amp = False  # Désactiver AMP si pas de CUDA
    
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
            
            # Vérifier les données d'entrée
            if torch.isnan(xb).any() or torch.isnan(yb).any():
                print("Warning: NaN in input data, skipping batch")
                continue
            
            optimizer.zero_grad(set_to_none=True)
            
            # Correction 2: Utiliser la nouvelle API pour autocast
            if amp:
                with torch.amp.autocast('cuda'):
                    pred = model(xb)
                    
                    # Vérifier les prédictions avant de calculer les losses
                    if torch.isnan(pred).any():
                        print("Warning: NaN in model output!")
                        continue
                    
                    l_h = huber(pred, yb)
                    l_r2 = r2loss(pred, yb)
                    l_ssim = ssim_loss(pred, yb)
                    loss = alpha * l_h + beta * l_r2 + gamma * l_ssim
            else:
                pred = model(xb)
                
                # Vérifier les prédictions avant de calculer les losses
                if torch.isnan(pred).any():
                    print("Warning: NaN in model output!")
                    continue
                
                l_h = huber(pred, yb)
                l_r2 = r2loss(pred, yb)
                l_ssim = ssim_loss(pred, yb)
                loss = alpha * l_h + beta * l_r2 + gamma * l_ssim
            
            # Vérifier la loss avant la backpropagation
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss detected: {loss.item()}")
                continue
            
            # Backpropagation avec ou sans AMP
            if amp and scaler is not None:
                scaler.scale(loss).backward()
                if clip_grad and clip_grad > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if clip_grad and clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()

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
                
            # Vérifier les NaN dans les visualisations
            if not torch.isnan(pr_vis).any() and not torch.isnan(yb_vis).any():
                pr_np = (pr_vis[0,0].cpu().numpy() * label_std) + label_mean
                gt_np = (yb_vis[0,0].cpu().numpy() * label_std) + label_mean
                save_prediction(os.path.join(outdir, f'epoch_{epoch:03d}.png'), gt_np, pr_np)
        except Exception as e:
            print(f"Warning: Could not save prediction visualization: {e}")

        # Save best - COMME DANS VOTRE CODE
        if val_r2 > best_r2 and not np.isnan(val_r2):
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
