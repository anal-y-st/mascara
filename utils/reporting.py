import os, json, datetime, yaml

def save_report(outdir, config, best_epoch, best_metrics, extra=None):
    os.makedirs(outdir, exist_ok=True)
    report = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "config": config,
    }
    if extra:
        report["extra"] = extra

    with open(os.path.join(outdir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # Markdown version
    md = []
    md.append(f"# Training Report\n")
    md.append(f"- **Best epoch:** {best_epoch}")
    md.append(f"- **R2:** {best_metrics.get('r2'):.4f}")
    md.append(f"- **MAE:** {best_metrics.get('mae'):.6f}")
    md.append(f"- **RMSE:** {best_metrics.get('rmse'):.6f}\n")
    md.append("## Config\n")
    md.append("```yaml\n" + yaml.safe_dump(config, sort_keys=False) + "```\n")
    with open(os.path.join(outdir, "report.md"), "w") as f:
        f.write("\n".join(md))
