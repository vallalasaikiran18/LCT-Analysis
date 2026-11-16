"""Stage 3 - label-free comparison of the per-set M1 maps, plus the written report.

  * agreement between the sets' maps (Spearman, top-k Jaccard, point proximity) on the common domain,
  * the layer-count test: does the score distribution depend on how many layers a set has?
  * a sensor summary table (layers, families, stability, uncertainty, what the sensor cannot see),
  * recommendation.md, rewritten whenever new evidence (evaluation, field test) exists.

Agreement is NOT accuracy: two maps built from correlated assumptions can agree and both be wrong.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from . import utils as U

log = U.log


def load_maps(cfg, model="M1", sets=None):
    """{set: dict(pct, sd, raw, points, metrics, leaf)} for sets whose model output exists."""
    out, grid = {}, None
    for s in (sets or U.run_sets(cfg)):
        leaf = U.d_leaf(cfg, model, s, create=False)
        p = leaf / "score_pct.tif"
        if not p.exists():
            continue
        pct, _, grid = U.read_stack(p)
        sd, _, _ = U.read_stack(leaf / "score_pct_sd.tif")
        raw, _, _ = U.read_stack(leaf / "score_raw.tif")
        pts = pd.read_csv(leaf / "points.csv") if (leaf / "points.csv").exists() else pd.DataFrame()
        out[s] = dict(pct=pct[0], sd=sd[0], raw=raw[0], points=pts, metrics=U.read_json(leaf / "metrics.json"), leaf=leaf)
    return out, grid


# =========================================================================== agreement
def agreement(cfg, maps, grid, outdir):
    sets = list(maps)
    if len(sets) < 2:
        return pd.DataFrame()
    common = np.all([np.isfinite(maps[s]["pct"]) for s in sets], axis=0)
    rng = np.random.default_rng(cfg["project"]["seed"])
    idx = np.flatnonzero(common.ravel())
    take = rng.choice(idx, min(len(idx), 200000), replace=False)
    S = np.column_stack([maps[s]["pct"].ravel()[take] for s in sets])
    rows = []
    sep = cfg["points"]["min_separation_m"]
    for i, a in enumerate(sets):
        for j, b in enumerate(sets):
            if j <= i:
                continue
            d = dict(set_a=a, set_b=b, spearman=float(spearmanr(S[:, i], S[:, j])[0]))
            for f in (0.05, 0.10):
                ta, tb = S[:, i] >= 1 - f, S[:, j] >= 1 - f
                d[f"top{int(f * 100)}_jaccard"] = float((ta & tb).sum() / max((ta | tb).sum(), 1))
                d[f"top{int(f * 100)}_jaccard_if_independent"] = float(f * f / (2 * f - f * f))
            pa, pb = maps[a]["points"], maps[b]["points"]
            if len(pa) and len(pb):
                dist = np.hypot(pa.Easting.values[:, None] - pb.Easting.values[None], pa.Northing.values[:, None] - pb.Northing.values[None])
                d[f"points_a_within_{sep}m_of_b"] = float((dist.min(1) <= sep).mean())
            rows.append(d)
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "agreement_between_sets.csv", index=False)
    n = len(sets)
    M = np.eye(n)
    J = np.eye(n)
    for r in rows:
        i, j = sets.index(r["set_a"]), sets.index(r["set_b"])
        M[i, j] = M[j, i] = r["spearman"]
        J[i, j] = J[j, i] = r["top5_jaccard"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, A, t in zip(axs, (M, J), ("Spearman (whole map)", "Top-5% overlap (Jaccard)")):
        im = ax.imshow(A, vmin=-1 if A is M else 0, vmax=1, cmap="viridis")
        ax.set_xticks(range(n), sets)
        ax.set_yticks(range(n), sets)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", color="w", fontsize=9)
        ax.set_title(t)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(outdir / "agreement_between_sets.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    fig, axs = plt.subplots(1, n, figsize=(4.2 * n, 4))
    for ax, s in zip(np.atleast_1d(axs), sets):
        ax.imshow(maps[s]["pct"], vmin=0, vmax=1, cmap="viridis")
        p = maps[s]["points"]
        if len(p):
            ax.scatter(p.col, p.row, s=12, facecolor="none", edgecolor="red")
        ax.set_title(f"{s} ({maps[s]['metrics']['n_features']} layers)")
        ax.set_xticks([]), ax.set_yticks([])
    fig.savefig(outdir / "maps_side_by_side.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return df


# =========================================================================== layer-count test
def default_k_values(cfg, sizes):
    lc = cfg.get("comparison", {}).get("layer_count_test", {}) or {}
    if lc.get("k_values"):
        return sorted(set(int(k) for k in lc["k_values"]))
    kmin = min(sizes.values())
    return sorted(set(range(2, kmin + 1)) | set(sizes.values()))


def run_layer_count(cfg, store, sets, outdir, occ_flat=None, tag=""):
    """Score random k-layer subsets of every set on the same pixel sample. With occ_flat (flat pixel indices
    of calibration occurrences) the calibration AUC is reported too, so sets can be compared at equal k."""
    from .models import layer_count_experiment, prepare
    lc = cfg.get("comparison", {}).get("layer_count_test", {}) or {}
    preps = {s: prepare(cfg, store, s) for s in sets}
    sizes = {s: len(p.features) for s, p in preps.items()}
    ks = default_k_values(cfg, sizes)
    frames = []
    for i, (s, prep) in enumerate(preps.items()):
        N = prep.X.shape[0]
        rng = np.random.default_rng(cfg["project"]["seed"] + 3)
        rows = np.sort(rng.choice(N, min(N, lc.get("max_pixels", 200000)), replace=False))
        pos_rows = None
        if occ_flat is not None:
            pos_rows = np.searchsorted(prep.flat_idx, occ_flat)
            pos_rows = pos_rows[(pos_rows < len(prep.flat_idx)) & (prep.flat_idx[np.minimum(pos_rows, len(prep.flat_idx) - 1)] == occ_flat)]
        frames.append(layer_count_experiment(cfg, prep, ks, lc.get("n_subsets", 30), rows=rows, pos_rows=pos_rows,
                                             seed_offset=i))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(outdir / f"layer_count_test{tag}.csv", index=False)
    summ = []
    for s, g in df.groupby("set"):
        rho_sd = spearmanr(g.k, g.raw_sd)[0] if g.k.nunique() > 1 else np.nan
        rho_mean = spearmanr(g.k, g.raw_mean)[0] if g.k.nunique() > 1 else np.nan
        d = dict(set=s, n_layers=sizes[s], spearman_k_vs_raw_sd=rho_sd, spearman_k_vs_raw_mean=rho_mean)
        if "auc_calibration" in g:
            d["spearman_k_vs_auc"] = spearmanr(g.k, g.auc_calibration)[0] if g.k.nunique() > 1 else np.nan
            kc = min(sizes.values())
            gk = g[g.k == kc]
            d[f"auc_at_equal_k{kc}_median"] = float(gk.auc_calibration.median()) if len(gk) else np.nan
            d[f"auc_at_equal_k{kc}_p10"] = float(gk.auc_calibration.quantile(0.1)) if len(gk) else np.nan
            d[f"auc_at_equal_k{kc}_p90"] = float(gk.auc_calibration.quantile(0.9)) if len(gk) else np.nan
        summ.append(d)
    sm = pd.DataFrame(summ)
    sm.to_csv(outdir / f"layer_count_summary{tag}.csv", index=False)
    cols = ["raw_sd", "raw_mean"] + (["auc_calibration"] if "auc_calibration" in df else [])
    fig, axs = plt.subplots(1, len(cols), figsize=(5 * len(cols), 4))
    for ax, c in zip(np.atleast_1d(axs), cols):
        for s, g in df.groupby("set"):
            m = g.groupby("k")[c].agg(["median", lambda x: x.quantile(0.1), lambda x: x.quantile(0.9)])
            m.columns = ["med", "lo", "hi"]
            ax.plot(m.index, m.med, marker="o", label=s)
            ax.fill_between(m.index, m.lo, m.hi, alpha=0.2)
        ax.set(xlabel="number of layers (k)", ylabel=c, title=f"{c} vs layer count")
        ax.legend()
    fig.savefig(outdir / f"layer_count_test{tag}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    log.info("Layer-count test%s:\n%s", tag, sm.to_string(index=False))
    return df, sm


# =========================================================================== summary + report
def sensor_summary(cfg, maps, store=None):
    rows = []
    sens = cfg.get("models", {}).get("fuzzy_gamma", {})
    for s, m in maps.items():
        mt = m["metrics"]
        sens_p = m["leaf"] / "sensitivity_oat.csv"
        top_driver = ""
        if sens_p.exists():
            so = pd.read_csv(sens_p)
            dr = so[(so.level == "layer") & (so.factor == 0.0)].sort_values("impact", ascending=False)
            top_driver = ", ".join(f"{r.item} ({r.impact:.2f})" for r in dr.head(3).itertuples())
        rows.append(dict(set=s, n_layers=mt["n_features"], n_families=mt.get("n_families"),
                         scorable_pixels=mt["n_scorable_pixels"], n_points=mt["n_points"],
                         mc_stability_spearman=mt["stability"].get("mean_spearman"),
                         mc_stability_top5_overlap=mt["stability"].get("mean_top_overlap"),
                         mean_pct_sd_points=mt.get("mean_pct_sd_points"),
                         raw_score_sd=mt.get("raw_score_sd"),
                         ahp_cr=(mt.get("extra") or {}).get("ahp_consistency_ratio"),
                         drop_one_top_drivers=top_driver))
    return pd.DataFrame(rows)


def run_compare(cfg):
    from .features import Store
    out = U.d_stage(cfg, "comparison")
    maps, grid = load_maps(cfg)
    if not maps:
        raise FileNotFoundError("No M1 outputs found. Run --stage models first.")
    agreement(cfg, maps, grid, out)
    store = Store(cfg)
    sets = [s for s in maps if store.features_for(s)]
    run_layer_count(cfg, store, sets, out)
    sensor_summary(cfg, maps).to_csv(out / "sensor_summary_label_free.csv", index=False)
    write_report(cfg)
    log.info("Comparison written to %s", out)


def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:.{nd}f}" if isinstance(v, (float, np.floating)) else str(v)


def _md_table(df):
    if df is None or df.empty:
        return "_none_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(_fmt(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def write_report(cfg):
    """recommendation.md from whatever evidence exists (label-free comparison, evaluation, field test)."""
    cmp_d, ev_d = U.d_stage(cfg, "comparison"), U.d_stage(cfg, "evaluation")
    feat_d = U.d_stage(cfg, "features")
    L = [f"# Sensor comparison for LCT-pegmatite prospectivity: {cfg['project']['name']}", "",
         "Model: fuzzy gamma (M1), run separately on each data source, all on the same analysis pixels, "
         "with the same number of output points. Scores are rank percentiles.", ""]
    fs = U.d_fe(cfg, "store") / "feature_sets_summary.json"
    if fs.exists():
        j = U.read_json(fs)
        L += ["## Feature sets", "", "Layers selected per set (selection was run separately for each set): " +
              ", ".join(f"**{k}** {v}" for k, v in j["features_per_set"].items()) + ".", ""]
    prov = U.d_fe(cfg, "qa") / "lineament_provenance.json"
    if prov.exists():
        p = U.read_json(prov)
        L += ["## Lineament provenance", "",
              f"Declared source: {p.get('declared_derived_from')}. Assigned to set: {p.get('assigned_feature_set')}. "
              f"Conclusion: {p.get('conclusion', 'n/a')}. " +
              " ".join(f"{k.replace('edge_pct_on_lineaments_', 'edge percentile on lineaments, ')} = {_fmt(v, 2)};"
                       for k, v in p.items() if k.startswith("edge_pct_on_lineaments_")) +
              (f" **{p['warning']}**" if p.get("warning") else ""), ""]
    sp = U.d_fe(cfg, "spectral") / "spectral_check_summary.csv"
    if sp.exists():
        L += ["## Spectral check (library spectra resampled to each sensor)", "",
              _md_table(pd.read_csv(sp)),
              "Details per layer: `01_feature_engineering/00_spectral_check/spectral_check.csv`. "
              "`relabel_generic` layers detect a mineral class, not the named lithium mineral.", ""]
    else:
        st = U.d_fe(cfg, "spectral") / "spectral_library_status.json"
        if st.exists():
            L += ["## Spectral check", "", f"Not run: {U.read_json(st).get('error') or 'too few library minerals'}.", ""]
    s1 = cmp_d / "sensor_summary_label_free.csv"
    if s1.exists():
        L += ["## Label-free comparison", "", _md_table(pd.read_csv(s1))]
    ag = cmp_d / "agreement_between_sets.csv"
    if ag.exists():
        L += ["Agreement between sets (agreement is not accuracy):", "", _md_table(pd.read_csv(ag))]
    lc = cmp_d / "layer_count_summary.csv"
    if lc.exists():
        L += ["Layer-count test (Spearman of the number of layers against the raw score spread / mean). "
              "Points and percentiles are rank-based, so a spread that depends on k does not bias them; it does "
              "mean raw scores must never be compared across sets.", "", _md_table(pd.read_csv(lc))]
    comp = ev_d / "sensor_comparison.csv"
    L += ["## Evaluation against known occurrences", ""]
    if comp.exists():
        c = pd.read_csv(comp)
        lk = ev_d / "FREEZE.lock.json"
        if lk.exists():
            k = U.read_json(lk)
            L += [f"Weights frozen on {k['created']}. Held-out set consulted {k.get('holdout_consults', 1)} time(s).", ""]
        keep = [k for k in ["set", "status", "n_layers", "carries_signal", "holdout_n", "holdout_auc", "holdout_auc_ci_low",
                            "holdout_auc_ci_high", "holdout_auc_null95", "holdout_capture_5", "holdout_capture_5_null95",
                            "holdout_capture_10", "holdout_capture_10_null95", "success_auc", "mc_stability_spearman"] if k in c]
        L += [_md_table(c[keep]), "Full table with CIs and p-values: `04_evaluation/sensor_comparison.xlsx`.", ""]
        lce = ev_d / "layer_count_summary_calibration.csv"
        if lce.exists():
            L += ["Calibration AUC at equal layer count (random subsets):", "", _md_table(pd.read_csv(lce))]
        L += ["Per-layer evidence (calibration occurrences): `04_evaluation/layer_evidence.xlsx`.", ""]
        L += ["## Recommendation", ""] + _recommend(cfg, c)
    else:
        L += ["Not run yet (no `inputs.known_occurrences`, or `--stage evaluate` not run). Without ground truth this "
              "report can only describe stability and agreement, which do **not** show which source is right.", "",
              "## Recommendation", "",
              "No recommendation is possible yet. Provide known occurrences and run `--stage evaluate`, "
              "or run the blinded field test on each set.", ""]
    fe = U.d_stage(cfg, "field_test") / "field_evaluation.json"
    if fe.exists():
        f = U.read_json(fe)
        L += ["## Field test", "", f"Set tested: {f.get('feature_set', 'n/a')}. Top hit rate {_fmt(f['hit_rate_top'], 2)} "
              f"vs controls {_fmt(f['hit_rate_controls'], 2)} (ratio {_fmt(f['ratio'], 2)}, one-sided Fisher p = "
              f"{_fmt(f['fisher_one_sided_p'], 3)}). Pre-registered criterion met: **{f['success_criterion_met']}**.", ""]
    L += ["## What each source cannot see", ""]
    for s, txt in (cfg.get("sensor_limitations") or {}).items():
        L += [f"**{s}.** {txt}", ""]
    (U.d_stage(cfg, "evaluation") / "recommendation.md").write_text("\n".join(L), encoding="utf-8")


def _recommend(cfg, c):
    ok = c[c["carries_signal"] == True] if "carries_signal" in c else c.iloc[0:0]  # noqa: E712
    out = []
    inv = c[c.get("holdout_auc_ci_high", pd.Series(1.0, index=c.index)) < 0.5] if "holdout_auc_ci_high" in c else c.iloc[0:0]
    for r in inv.itertuples():
        out.append(f"**{r.set}** ranks held-out occurrences LOWER than random ground (AUC {_fmt(r.holdout_auc)}, CI "
                   f"{_fmt(r.holdout_auc_ci_low)}-{_fmt(r.holdout_auc_ci_high)}). That is a signal pointing the wrong way: "
                   "check the declared directions against the per-layer table (look for 'CONTRADICTS declared direction').")
    if ok.empty:
        out.append("No set passes the signal rule on the held-out occurrences (AUC confidence interval above 0.5 "
                   "AND top-area capture above the random-point envelope). Do not pick a set on map appearance; "
                   "revisit the evidence layers (see the per-layer table) or get field data.")
        return out + [""]
    best = ok.sort_values("holdout_auc", ascending=False).iloc[0]
    out.append(f"**{best.set}** carries the strongest held-out signal (prediction-rate AUC {_fmt(best.holdout_auc)}, "
               f"95% CI {_fmt(best.holdout_auc_ci_low)}-{_fmt(best.holdout_auc_ci_high)}; "
               f"captures {_fmt(100 * best.holdout_capture_10, 0)}% of held-out occurrences in the top 10% of area).")
    others = ok[ok.set != best.set]
    if len(others):
        overlap = [o for o in others.itertuples() if o.holdout_auc_ci_high >= best.holdout_auc_ci_low]
        if overlap:
            out.append("Its confidence interval overlaps with " + ", ".join(o.set for o in overlap) +
                       ": the difference between them is not established by these occurrences.")
    if best.set == "DEM":
        out.append("DEM is context, not detection: if it wins, the occurrences sit in a recognisable landscape "
                   "position. Use it to constrain an image-based model rather than on its own.")
    out.append("Refine weights using the calibration per-layer table only; every revision is scored on the "
               "held-out set and logged. Confirm with the blinded field test (`--stage field_test`).")
    return out + [""]
