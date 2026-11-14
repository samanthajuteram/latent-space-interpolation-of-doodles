# Beta sweep: retrain the VAE with different KL weights and rerun interpolation metrics

# beta scales the KL term (Higgins et al., 2017)
# higher beta --> should close gaps between classes but blur images
# goal: find the balance between image clarity and interpolation performance

# Outputs:
#   src/vae_beta{b}.pth                        one model per beta
#   results/training_losses_beta{b}.png        loss curves per beta
#   results/interpolation/beta_{b}/            full interpolation figures + metrics per beta
#   results/interpolation/beta_sweep.png       trade off plot across betas
#   results/interpolation/beta_sweep.json      summary numbers across betas

# Usage (from repo root):
#   python src/_5_beta_sweep.py
# (models that already exist are not retrained -> delete the .pth to retrain)

import json
from pathlib import Path

import matplotlib.pyplot as plt

from _2_train import train
import _4_interpolate as interp

BETAS = [0.1, 0.5, 1.0, 2.0, 5.0]


def main():
    src_dir = Path(__file__).parent
    out_dir = src_dir.parent / "results" / "interpolation"
    sweep = {}

    for beta in BETAS:
        print(f"\n===== beta = {beta} =====")

        # TRAIN --
        # skip if this beta was already trained
        if not (src_dir / f"vae_beta{beta}.pth").exists():
            train(beta=beta, save_name=f"vae_beta{beta}.pth",
                  plot_name=f"training_losses_beta{beta}.png", show=False)

        # INTERPOLATE --
        # same metrics + figures as the baseline, saved to results/interpolation/beta_{beta}/
        sweep[beta] = interp.main(["--beta", str(beta)])

    # SUMMARY --
    # pull out the numbers we care about for each beta
    rows = {
        b: {
            "recon_bce": r["test_recon_bce_mean"],                              # clarity (lower = sharper)
            "latent_probe_acc": r["latent_linear_probe_test_acc"],              # class separation in latent space
            "max_step_ratio": r["cross_lerp"]["max_step_ratio_median"],         # smoothness (1 = perfectly even)
            "peak_off_manifold": r["cross_lerp"]["peak_off_manifold_median"],   # gaps (>1 = passes through empty space)
            "transition_width": r["cross_lerp"]["transition_width_median"],     # how gradual the cat -> clock switch is
        }
        for b, r in sweep.items()
    }

    with open(out_dir / "beta_sweep.json", "w") as f:
        json.dump(rows, f, indent=2)

    print("\nbeta  | recon BCE | probe acc | max step | off manifold | transition")
    for b, r in rows.items():
        print(f"{b:<5} | {r['recon_bce']:9.2f} | {r['latent_probe_acc']:9.3f} | "
              f"{r['max_step_ratio']:8.3f} | {r['peak_off_manifold']:12.3f} | {r['transition_width']:10.3f}")

    # TRADE OFF PLOT --
    # one panel per metric, beta on a log x axis
    panels = [
        ("recon_bce", "Reconstruction BCE (clarity)", "lower = sharper"),
        ("max_step_ratio", "Max step / mean step (smoothness)", "1 = perfectly even"),
        ("peak_off_manifold", "Peak distance to real codes (gaps)", "× typical NN spacing"),
        ("transition_width", "Cat -> clock transition width", "fraction of path"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(16, 3.8))
    for ax, (key, title, ylabel) in zip(axes, panels):
        ax.plot(BETAS, [rows[b][key] for b in BETAS], "o-")
        ax.set_xscale("log")
        ax.set_xticks(BETAS)
        ax.set_xticklabels([str(b) for b in BETAS])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("beta")
        ax.set_ylabel(ylabel)
    fig.suptitle("Beta sweep: image clarity vs interpolation performance", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / "beta_sweep.png", dpi=200)
    plt.close("all")

    print(f"\nSaved sweep results to {out_dir}")


if __name__ == "__main__":
    main()
