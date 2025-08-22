# Mask Learning Tool 🧩

Outil léger et extensible pour **entraîner** et **évaluer** des modèles de vision (construction de masques, régression/segmentation), en laissant **l'utilisateur gérer ses DataLoader**.

## ✨ Points clés
- Modèles plug-and-play (registre de modèles).
- Entraînement générique (mélange Huber/R2/SSIM, AMP optionnel, early stopping, scheduler).
- Visualisations par epoch (GT / Pred / Diff), courbes de métriques et **rapport automatique** (`report.json` + `report.md`).
- Config YAML + **override** en ligne de commande (`--set training.epochs=50 --set training.lr=1e-5`).
- Agnostique aux données : vous fournissez un module avec `get_loaders(...)` qui retourne vos DataLoader et stats.

## 🚀 Installation
```bash
pip install -r requirements.txt
# (optionnel) installation locale
pip install -e .
```

## 🧪 Démarrage rapide
1) Adaptez l'exemple de loader dans `examples/loader_template.py` à vos données.
2) Lancez l'entraînement :
```bash
python scripts/train.py   --config configs/swin_unet.yaml   --loaders examples.loader_template:get_loaders   --set training.epochs=5 --set training.outdir=results_demo
```
3) Évaluation (recharge le best checkpoint) :
```bash
python scripts/evaluate.py   --config configs/swin_unet.yaml   --loaders examples.loader_template:get_loaders   --checkpoint results_demo/best_model.pth
```

## 🧩 Fournir vos DataLoader
Votre module doit exposer une fonction :
```python
def get_loaders(config: dict):
    """
    Retourne
      - train_loader (torch.utils.data.DataLoader)
      - val_loader   (torch.utils.data.DataLoader)
      - stats: dict avec les clefs suivantes
          input_mean, input_std, label_mean, label_std (scalaires ou arrays broadcastables)
    """
    return train_loader, val_loader, {
        "input_mean": 0.0, "input_std": 1.0,
        "label_mean": 0.0, "label_std": 1.0
    }
```
Pour un problème de **segmentation/régression par pixel**, les tenseurs `y` sont attendus de forme `[B, 1, H, W]`.

## ⚙️ Override de config en CLI
- `--set a.b.c=value` met à jour récursivement la config (types inférés : `true/false/int/float`).
- Exemples :
  - `--set training.lr=1e-4`
  - `--set model.embed_dim=96`
  - `--set training.amp=true --set training.early_stopping.patience=8`

## 📦 Résultats produits
Dans `training.outdir` :
- `best_model.pth` (checkpoint meilleur R2)
- `history.json` (loss, R2, MAE, RMSE par epoch)
- `curves.png` (courbes d’apprentissage)
- `epoch_XX.png` (GT/Pred/Diff par epoch)
- `report.json` et `report.md` (résumé complet)

## 🧱 Ajouter un modèle
Ajoutez un fichier dans `models/` et enregistrez-le dans `models/registry.py`.

---

Made with ❤️ for research & reproducibility.
