# Latent space interpolation: cat <-> clock

# Encode selected test examples, interpolate between their latent vectors,
# decode the intermediate points, and measure whether the transitions are
# smooth and meaningful.

# Outputs (results/interpolation/):
#   cross_class_grid.png     cat -> clock paths (linear interpolation)
#   within_class_grid.png    cat -> cat and clock -> clock paths (control)
#   lerp_vs_slerp.png        linear vs spherical latent interpolation vs pixel crossfade
#   path_metrics.png         classifier P(clock), step size, and off-manifold distance along the path
#   latent_pca.png           2D PCA of test set latent means with interpolation paths overlaid
#   metrics.json             aggregate numbers over many random pairs


import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from _1_vae import VAE
from _0_data import get_dataloader

CAT, CLOCK = 0, 1
LATENT_DIM = 64  # must match train scrip


# INTERPOLATION --

# linear interpolation: straight line blend 
def lerp(z0, z1, ts):
    ts = ts[:, None]
    return (1 - ts) * z0 + ts * z1

# spherical linear interpolation: moves along arc
def slerp(z0, z1, ts):
    # normalize vectors
    z0n, z1n = z0 / z0.norm(), z1 / z1.norm()
    omega = torch.acos(torch.clamp((z0n * z1n).sum(), -1 + 1e-7, 1 - 1e-7))
    if omega.abs() < 1e-4: #fall back to lerp to prevent zero division
        return lerp(z0, z1, ts)
    so = torch.sin(omega)
    ts = ts[:, None]
    return (torch.sin((1 - ts) * omega) / so) * z0 + (torch.sin(ts * omega) / so) * z1


# HELPERS --

# drain data loader into one big tensor of images and labels 
def collect(loader):
    xs, ys = [], []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
    return torch.cat(xs), torch.cat(ys)

# encode images and keep only mu, not a random sample
@torch.no_grad()
def encode_mu(model, X, device, bs=512):
    mus = []
    for i in range(0, len(X), bs):
        mu, _ = model.encode(X[i:i + bs].to(device))
        mus.append(mu.cpu())
    return torch.cat(mus)

# map latent vectors back to images 
@torch.no_grad()
def decode(model, Z, device):
    return model.decode(Z.to(device)).cpu()

# encode each image, decode mu and compute per image binary cross entropy against original
@torch.no_grad()
def recon_error(model, X, device):
    """Per-image BCE of deterministic reconstruction (decode(mu))."""
    mu = encode_mu(model, X, device)
    R = decode(model, mu, device)
    return F.binary_cross_entropy(R, X, reduction="none").flatten(1).sum(1)


def show_row(axes, imgs, titles=None):
    for j, ax in enumerate(axes):
        ax.imshow(imgs[j].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        if titles is not None:
            ax.set_title(titles[j], fontsize=8)

# metrics for a single decoded path 
def path_stats(imgs, zs, clf, train_mu_np, nn_scale):
    flat = imgs.flatten(1).numpy()
    p_clock = clf.predict_proba(flat)[:, 1] # prob that each frame looks like a clock (tests whether the cat is actually changing)

    # pixel distance between consecutive frames
    steps = np.linalg.norm(np.diff(flat, axis=0), axis=1)

    # biggest step/ avg step
    # 1.0 = perfectly even change per step. large = one abrupt jump
    max_step_ratio = float(steps.max() / (steps.mean() + 1e-8))

    # how far each path point is from the nearest real training latent code,
    # relative to the typical nearest neighbour spacing among real codes.
    # >1 means the path is passing through a region with no real data 
    d = torch.cdist(zs, torch.from_numpy(train_mu_np)).min(1).values.numpy()
    off_manifold = d / nn_scale

    return dict(p_clock=p_clock, steps=steps, max_step_ratio=max_step_ratio,
                off_manifold=off_manifold)

# how gradually the category switches
# small width = abrupt flip
# large width = gradual morph 
# nan = path never cleanly goes from not clock to clock
def transition_width(p, ts):
    p = np.maximum.accumulate(p) # force p(clock) to be non decreasing
    if p[0] > 0.2 or p[-1] < 0.8:
        return np.nan
    t_lo = np.interp(0.2, p, ts)
    t_hi = np.interp(0.8, p, ts)
    return float(t_hi - t_lo)


# MAIN --

# argv lets the beta sweep call main() directly
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10, help="number of interpolation points incl. endpoints")
    ap.add_argument("--n-show", type=int, default=5, help="pairs shown in the grids")
    ap.add_argument("--n-pairs", type=int, default=200, help="random pairs used for aggregate metrics")
    ap.add_argument("--cat-idx", type=int, nargs="*", help="indices into the test set cats to use as endpoints")
    ap.add_argument("--clock-idx", type=int, nargs="*", help="indices into the test set clocks to use as endpoints")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", action="store_true", help="also display figures")
    ap.add_argument("--beta", type=float, help="load vae_beta{beta}.pth from the beta sweep instead of the baseline")
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(__file__).parent.parent / "results" / "interpolation"

    # pick checkpoint: baseline by default, or one model from the beta sweep (own results folder)
    model_name = "vae_baseline.pth"
    if args.beta is not None:
        model_name = f"vae_beta{args.beta}.pth"
        out_dir = out_dir / f"beta_{args.beta}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # load model
    model = VAE(latent_dim=LATENT_DIM).to(device)
    model.load_state_dict(torch.load(Path(__file__).parent / model_name, weights_only=True, map_location=device))
    model.eval()

    # data (same split as training)
    train_loader, _, test_loader = get_dataloader(batch_size=512)
    X_tr, y_tr = collect(train_loader)
    X_te, y_te = collect(test_loader)

    # encode everything (comput emu for every train and test image)
    mu_tr = encode_mu(model, X_tr, device)
    mu_te = encode_mu(model, X_te, device)

    # train the judge
    # fit a logistic regression model on raw pixels to tell cats and clocks
    # never sees VAE so its an independent opinion on what a decoded image looks like
    clf = LogisticRegression(max_iter=2000, C=0.1)
    clf.fit(X_tr.flatten(1).numpy(), y_tr.numpy())
    clf_acc = clf.score(X_te.flatten(1).numpy(), y_te.numpy())
    print(f"Pixel classifier test accuracy: {clf_acc:.3f}")

    # get off manifold metric a unit of measurement
    # sample 1000, find nearest neighbor among all, take median
    # normal spacing the off manifold measuremnt is measured against
    mu_tr_np = mu_tr.numpy()
    sub = torch.from_numpy(mu_tr_np[rng.choice(len(mu_tr_np), 1000, replace=False)])
    dmat = torch.cdist(sub, torch.from_numpy(mu_tr_np))
    dmat[dmat < 1e-6] = float("inf")  # ignore self distance
    nn_scale = float(dmat.min(1).values.median())

    # ENDPOINT SELECTION --
    # Prefer well reconstructed test examples so that a bad interpolation can't be blamed on a bad encoding of the endpoints.
    err = recon_error(model, X_te, device)
    cat_ids = torch.where(y_te == CAT)[0]
    clock_ids = torch.where(y_te == CLOCK)[0]

    def good(ids):
        e = err[ids]
        return ids[e <= e.median()].numpy()

    good_cats, good_clocks = good(cat_ids), good(clock_ids)

    if args.cat_idx:
        show_cats = cat_ids.numpy()[args.cat_idx]
    else:
        show_cats = rng.choice(good_cats, args.n_show, replace=False)
    if args.clock_idx:
        show_clocks = clock_ids.numpy()[args.clock_idx]
    else:
        show_clocks = rng.choice(good_clocks, args.n_show, replace=False)
    n_show = min(len(show_cats), len(show_clocks))

    ts = torch.linspace(0, 1, args.steps)
    ts_np = ts.numpy()

    # FIGURE 1: cat -> clock grid --
    # deocded lerp steps from t=0 to t=1
    # middle cols show we get plausible hybrids or blurry mush

    fig, axes = plt.subplots(n_show, args.steps + 2, figsize=(1.1 * (args.steps + 2), 1.2 * n_show))
    axes = np.atleast_2d(axes)
    for r in range(n_show):
        a, b = show_cats[r], show_clocks[r]
        imgs = decode(model, lerp(mu_te[a], mu_te[b], ts), device)
        row = [X_te[a]] + list(imgs) + [X_te[b]]
        titles = ["cat (orig)"] + [f"t={t:.2f}" for t in ts_np] + ["clock (orig)"] if r == 0 else None
        show_row(axes[r], row, titles)
    fig.suptitle("Cat ->clock: linear interpolation between latent means", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / "cross_class_grid.png", dpi=200)

    # FIGURE 2: within-class control --
    # cat--> cat and clock --> clock paths
    # should be smooth because both ends ahre a category
    fig, axes = plt.subplots(2 * n_show, args.steps, figsize=(1.1 * args.steps, 1.2 * 2 * n_show))
    for r in range(n_show):
        for k, (pool, name) in enumerate([(good_cats, "cat -> cat"), (good_clocks, "clock -> clock")]):
            a, b = rng.choice(pool, 2, replace=False)
            imgs = decode(model, lerp(mu_te[a], mu_te[b], ts), device)
            show_row(axes[2 * r + k], imgs)
            axes[2 * r + k, 0].set_ylabel(name, fontsize=8)
    fig.suptitle("Within class interpolation (control)", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / "within_class_grid.png", dpi=200)

    # FIGURE 3: lerp vs slerp vs pixel crossfade --
    # cross fade is simply just fading one image into the other (no learned structure baseline)
    a, b = show_cats[0], show_clocks[0]
    z_lerp = lerp(mu_te[a], mu_te[b], ts)
    z_slerp = slerp(mu_te[a], mu_te[b], ts)
    rows = {
        "latent lerp": decode(model, z_lerp, device),
        "latent slerp": decode(model, z_slerp, device),
        # baseline: what a model with no learned structure would give
        "pixel crossfade": lerp(X_te[a].flatten(), X_te[b].flatten(), ts).view(-1, 1, 28, 28),
    }
    fig, axes = plt.subplots(3, args.steps, figsize=(1.1 * args.steps, 3.8))
    for r, (name, imgs) in enumerate(rows.items()):
        show_row(axes[r], imgs, [f"t={t:.2f}" for t in ts_np] if r == 0 else None)
        axes[r, 0].set_ylabel(name, fontsize=8)
    fig.suptitle("Latent interpolation vs. naive pixel blending", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / "lerp_vs_slerp.png", dpi=200)

    # Aggregate metrics over random pairs --
    dense_ts = torch.linspace(0, 1, 21)
    dense_np = dense_ts.numpy()
    curves = {k: {"p_clock": [], "steps": [], "off": []} for k in
              ["cross_lerp", "cross_slerp", "cross_pixel", "cat_cat", "clock_clock"]}
    summary = {k: {"max_step_ratio": [], "transition_width": [], "peak_off_manifold": []} for k in curves}

    def record(key, imgs, zs):
        s = path_stats(imgs, zs, clf, mu_tr_np, nn_scale)
        curves[key]["p_clock"].append(s["p_clock"])
        curves[key]["steps"].append(s["steps"])
        curves[key]["off"].append(s["off_manifold"])
        summary[key]["max_step_ratio"].append(s["max_step_ratio"])
        summary[key]["peak_off_manifold"].append(float(s["off_manifold"].max()))
        if key.startswith("cross"):
            summary[key]["transition_width"].append(transition_width(s["p_clock"], dense_np))

    for _ in range(args.n_pairs):
        a, b = rng.choice(good_cats), rng.choice(good_clocks)
        za, zb = mu_te[a], mu_te[b]
        zl, zs_ = lerp(za, zb, dense_ts), slerp(za, zb, dense_ts)
        record("cross_lerp", decode(model, zl, device), zl)
        record("cross_slerp", decode(model, zs_, device), zs_)
        px = lerp(X_te[a].flatten(), X_te[b].flatten(), dense_ts).view(-1, 1, 28, 28)
        record("cross_pixel", px, encode_mu(model, px, device))  # off-manifold via re-encoding

        for key, pool in [("cat_cat", good_cats), ("clock_clock", good_clocks)]:
            c, d = rng.choice(pool, 2, replace=False)
            zw = lerp(mu_te[c], mu_te[d], dense_ts)
            record(key, decode(model, zw, device), zw)

    # FIGURE 4: path metrics --
    # an S curve for cross class paths. steeper --> more abrupt the category switch
    # step size vs t: flat line --> even change
    # distance to real code
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    labels = {"cross_lerp": "cat->clock (lerp)", "cross_slerp": "cat ->clock (slerp)",
              "cross_pixel": "cat->clock (pixel crossfade)",
              "cat_cat": "cat->cat (lerp)", "clock_clock": "clock->clock (lerp)"}
    mid_t = (dense_np[1:] + dense_np[:-1]) / 2
    for key, lab in labels.items():
        P = np.stack(curves[key]["p_clock"])
        S = np.stack(curves[key]["steps"])
        O = np.stack(curves[key]["off"])
        style = "--" if key == "cross_pixel" else "-"
        for ax, M, x in [(axes[0], P, dense_np), (axes[1], S, mid_t), (axes[2], O, dense_np)]:
            m = M.mean(0)
            lo, hi = np.percentile(M, [25, 75], axis=0)
            ax.plot(x, m, style, label=lab)
            ax.fill_between(x, lo, hi, alpha=0.15)
    axes[0].set_title("Classifier P(clock) along path")
    axes[0].set_ylabel("P(clock)")
    axes[1].set_title("Pixel change between consecutive steps")
    axes[1].set_ylabel("L2 distance")
    axes[2].set_title("Distance to nearest real latent code")
    axes[2].set_ylabel("× typical NN spacing")
    axes[2].axhline(1.0, color="gray", lw=0.8, ls=":")
    for ax in axes:
        ax.set_xlabel("t (interpolation position)")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Interpolation path metrics (mean and IQR over {args.n_pairs} random pairs)", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / "path_metrics.png", dpi=200)

    # FIGURE 5: latent PCA with paths --
    # project to 2D with PCA, colored by class
    # cat --> clock lines 
    # shows whether classes form spearate clusters + if paths cross a gap btwn them
    pca = PCA(n_components=2).fit(mu_tr_np)
    P2 = pca.transform(mu_te.numpy())
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(*P2[y_te == CAT].T, s=4, alpha=0.3, label="cat")
    ax.scatter(*P2[y_te == CLOCK].T, s=4, alpha=0.3, label="clock")
    for r in range(n_show):
        path = pca.transform(lerp(mu_te[show_cats[r]], mu_te[show_clocks[r]], ts).numpy())
        ax.plot(*path.T, "k-", lw=0.8)
        ax.plot(*path[0], "ko", ms=4)
        ax.plot(*path[-1], "k^", ms=5)
    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ev[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1%} var)")
    ax.set_title("Test set latent means (PCA) with cat -> clock paths")
    ax.legend(markerscale=3)
    plt.tight_layout()
    plt.savefig(out_dir / "latent_pca.png", dpi=200)

    # SUMMARY --
    # How separable are the classes in latent space?
    probe = LogisticRegression(max_iter=2000).fit(mu_tr_np, y_tr.numpy())
    latent_probe_acc = probe.score(mu_te.numpy(), y_te.numpy())

    result = {
        "pixel_classifier_test_acc": round(clf_acc, 4), #how reliable the judge is 
        "latent_linear_probe_test_acc": round(latent_probe_acc, 4), # log reg on mu
        "nn_spacing_real_codes": round(nn_scale, 4), # ref distance
        "test_recon_bce_mean": round(float(err.mean()), 2), # image clarity: lower --> sharper reconstructions
        "n_pairs": args.n_pairs,
        "path_points": len(dense_np),
    }
    for key, d in summary.items():
        entry = {
            "max_step_ratio_median": round(float(np.median(d["max_step_ratio"])), 3), # closer to 1 --> smoother
            "peak_off_manifold_median": round(float(np.median(d["peak_off_manifold"])), 3), # trpical worst point on path (>1 means passes through empty space)
        }
        if d["transition_width"]: # cross class only
            tw = np.array(d["transition_width"])
            entry["transition_width_median"] = round(float(np.nanmedian(tw)), 3)# how gradual is the switch
            entry["frac_paths_reaching_clock"] = round(float(np.mean(~np.isnan(tw))), 3) # share of paths with clean 0.2 --> 0.8 transition
            P = np.stack(curves[key]["p_clock"])
            entry["p_clock_at_midpoint_mean"] = round(float(P[:, len(dense_np) // 2].mean()), 3)# p = 0.5 suggests midpt genuinely ambiguous 
        result[key] = entry

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"\nSaved figures to {out_dir}")

    if args.show:
        plt.show()
    plt.close("all") # free figures so the sweep doesn't pile them up

    return result


if __name__ == "__main__":
    main()
