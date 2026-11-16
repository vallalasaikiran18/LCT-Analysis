"""Stage 4 - evaluation against known occurrences, with a spatial calibration / held-out split and a freeze.

Order of operations (what keeps the comparison honest):
  1. occurrences are declustered (workings on one pegmatite count once) and restricted to the common domain;
  2. they are split by spatial BLOCKS into calibration and held-out; the split is written once and reused;
  3. per-layer evidence (AUC, Weights-of-Evidence contrast) uses CALIBRATION occurrences only - this is the
     table to refine weights from;
  4. map-level success-rate (calibration) and prediction-rate (held-out) curves, AUC with bootstrap CI,
     capture in the top 5% / 10% of area, against a random-point envelope;
  5. the first run writes FREEZE.lock.json (the fingerprint of every set's model). Any later change to a set's
     weights, directions, rules or features is detected and that set is re-scored on the HELD-OUT set only;
     each such consult is logged.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

from . import utils as U

log = U.log


# =========================================================================== occurrences
def load_occurrences(cfg, grid, domain):
    occ = U.load_points_file(cfg, "known_occurrences", grid)
    if occ is None or not len(occ):
        return None, {}
    ec = cfg["evaluation"]
    occ = occ[occ.geometry.notna() & ~occ.geometry.is_empty].copy()
    occ["geometry"] = occ.geometry.centroid if (occ.geom_type != "Point").any() else occ.geometry
    x, y = occ.geometry.x.values, occ.geometry.y.values
    r, c = grid.rc(x, y)
    inside = (r >= 0) & (r < grid.height) & (c >= 0) & (c < grid.width)
    rep = dict(n_input=int(len(occ)), outside_grid=int((~inside).sum()))
    df = pd.DataFrame(dict(x=x, y=y, row=r, col=c))[inside].reset_index(drop=True)
    df["src_index"] = np.flatnonzero(inside)
    keep, kx, ky = [], [], []                                    # greedy declustering
    for i, rr in df.iterrows():
        if kx and np.min(np.hypot(np.asarray(kx) - rr.x, np.asarray(ky) - rr.y)) < ec["decluster_m"]:
            continue
        keep.append(i)
        kx.append(rr.x)
        ky.append(rr.y)
    rep["merged_by_declustering"] = int(len(df) - len(keep))
    df = df.loc[keep].reset_index(drop=True)
    indom = domain[df.row.values, df.col.values]
    rep["outside_common_domain"] = int((~indom).sum())
    df = df[indom].reset_index(drop=True)
    df["flat"] = df.row.values * grid.width + df.col.values
    df = df.drop_duplicates("flat").reset_index(drop=True)
    rep["n_used"] = int(len(df))
    return df, rep


def spatial_split(cfg, occ, grid):
    ec = cfg["evaluation"]
    B = ec["block_size_m"]
    x0, y0 = grid.transform.c, grid.transform.f
    occ = occ.copy()
    occ["block"] = [f"{int((x - x0) // B)}_{int((y0 - y) // B)}" for x, y in zip(occ.x, occ.y)]
    rng = np.random.default_rng(cfg["project"]["seed"] + 101)
    blocks = list(rng.permutation(np.asarray(sorted(occ.block.unique()), dtype=object)))
    target = ec["holdout_fraction"] * len(occ)
    hold, n = [], 0
    for b in blocks:
        if n >= target:
            break
        hold.append(b)
        n += int((occ.block == b).sum())
    occ["split"] = np.where(occ.block.isin(hold), "holdout", "calibration")
    if len(blocks) < 4:
        log.warning("Only %d spatial blocks contain occurrences: the held-out test is weak. Reduce block_size_m.",
                    len(blocks))
    if (occ.split == "calibration").sum() < 5 or (occ.split == "holdout").sum() < 5:
        log.warning("Fewer than 5 occurrences in one split (calibration %d, held-out %d): results will be noisy.",
                    (occ.split == "calibration").sum(), (occ.split == "holdout").sum())
    return occ


# =========================================================================== statistics
def auc_vs_background(bg_sorted, vals):
    """P(occurrence value > background value), ties counted half."""
    lo = np.searchsorted(bg_sorted, vals, "left")
    hi = np.searchsorted(bg_sorted, vals, "right")
    return float(np.mean((lo + hi) / 2.0 / len(bg_sorted))) if len(vals) else np.nan


def auc_ci(bg_sorted, vals, n_boot, rng):
    if len(vals) < 3:
        return np.nan, np.nan
    lo = np.searchsorted(bg_sorted, vals, "left")
    hi = np.searchsorted(bg_sorted, vals, "right")
    p = (lo + hi) / 2.0 / len(bg_sorted)
    b = [p[rng.integers(0, len(p), len(p))].mean() for _ in range(n_boot)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def mw_p(auc, n1, n2):
    """One-sided Mann-Whitney p (normal approximation). Ignores spatial autocorrelation: optimistic."""
    if not np.isfinite(auc) or n1 < 1:
        return np.nan
    sd = np.sqrt((n1 + n2 + 1) / (12.0 * n1 * n2))
    return float(1 - norm.cdf((auc - 0.5) / sd))


def wofe(oriented_bg, oriented_occ, frac):
    """Weights of Evidence for the binary pattern 'top frac of the oriented layer' (pixels as unit cells)."""
    thr = np.quantile(oriented_bg, 1 - frac)
    N, nD = len(oriented_bg), len(oriented_occ)
    nB = int((oriented_bg >= thr).sum())
    nBD = int((oriented_occ >= thr).sum())
    a, b, c, d = nBD, nB - nBD, nD - nBD, N - nB - (nD - nBD)
    a_, c_ = (a if a > 0 else 0.5), (c if c > 0 else 0.5)
    pBD, pBnD = a_ / nD, max(b, 0.5) / max(N - nD, 1)
    wp = np.log(pBD / pBnD)
    wm = np.log((c_ / nD) / max((1 - pBnD), 1e-12))
    C = wp - wm
    s = np.sqrt(1 / a_ + 1 / max(b, 0.5) + 1 / c_ + 1 / max(d, 0.5))
    return dict(W_plus=float(wp), W_minus=float(wm), contrast=float(C), studentized_C=float(C / s),
                occ_in_pattern=a, area_frac=float(nB / N))


# =========================================================================== per-layer evidence
def layer_evidence(cfg, store, domain, calib, sets):
    ec = cfg["evaluation"]
    fe_al, fe_en, fe_st = U.d_fe(cfg, "aligned"), U.d_fe(cfg, "engineered"), U.d_fe(cfg, "store")
    cand = pd.read_excel(fe_st / "feature_dictionary.xlsx", sheet_name="all_candidates", index_col=0).fillna("")
    cand = cand[cand.feature_set.isin(sets) & cand.direction.isin(["high", "low", "band"])]
    stacks = {}
    for p in (fe_al / "aligned_layers.tif", fe_en / "engineered_stack.tif"):
        if p.exists():
            arr, names, _ = U.read_stack(p)
            for i, n in enumerate(names):
                stacks[n] = (p, i)
    cache = {}

    def get(n):
        p, i = stacks[n]
        if p not in cache:
            cache[p] = U.read_stack(p)[0]
        return cache[p][i]
    rng = np.random.default_rng(cfg["project"]["seed"] + 5)
    flat_dom = np.flatnonzero(domain.ravel())
    occ_flat = calib.flat.values
    rows = []
    for n, m in cand.iterrows():
        if n not in stacks:
            continue
        a = get(n).ravel()
        v_bg = a[flat_dom]
        ok = np.isfinite(v_bg)
        v_occ = a[occ_flat]
        okc = np.isfinite(v_occ)
        if ok.sum() < 100 or okc.sum() < 3:
            continue
        v_bg, v_occ = v_bg[ok], v_occ[okc]
        if m.direction == "high":
            ob, oo = v_bg, v_occ
        elif m.direction == "low":
            ob, oo = -v_bg, -v_occ
        else:                                                   # band: closeness of the percentile to the centre
            c = float(m.center) if str(m.center) not in ("", "nan") else cfg["models"]["fuzzy_gamma"]["band"]["center"]
            srt = np.sort(v_bg)
            qb = (rankdata(v_bg) - 0.5) / len(v_bg)
            qo = np.searchsorted(srt, v_occ) / len(srt)
            ob, oo = -np.abs(qb - c), -np.abs(qo - c)
        bs = np.sort(ob)
        auc = auc_vs_background(bs, oo)
        lo, hi = auc_ci(bs, oo, ec["n_boot"], rng)
        d = dict(feature_set=m.feature_set, layer=n, family=m.family, direction=m.direction,
                 label=m.get("label", ""), selection_status=m.get("status", ""), selection_reason=m.get("reason", ""),
                 spectral_verdict=m.get("spectral_verdict", ""), n_calibration=int(len(oo)),
                 auc=auc, auc_ci_low=lo, auc_ci_high=hi, p_mannwhitney_one_sided=mw_p(auc, len(oo), len(ob)),
                 auc_if_direction_reversed=1 - auc)
        best = None
        for f in ec["wofe_top_fractions"]:
            w = wofe(ob, oo, f)
            for k, v in w.items():
                d[f"top{int(f * 100)}_{k}"] = v
            if best is None or w["studentized_C"] > best[1]:
                best = (f, w["studentized_C"])
        d["best_wofe_top_fraction"] = best[0]
        d["best_studentized_C (optimistic: threshold chosen on the same data)"] = best[1]
        d["reading"] = ("supports declared direction" if lo > 0.5 else
                        ("CONTRADICTS declared direction" if hi < 0.5 else "no clear signal"))
        rows.append(d)
    df = pd.DataFrame(rows).sort_values(["feature_set", "auc"], ascending=[True, False])
    return df


# =========================================================================== map-level evaluation
def map_metrics(cfg, pct, domain, occ, rng, parts=("calibration", "holdout")):
    ec = cfg["evaluation"]
    bg = pct[domain & np.isfinite(pct)]
    bs = np.sort(bg)
    out, curves = {}, {}
    for part in parts:
        o = occ[occ.split == part]
        v = pct.ravel()[o.flat.values]
        v = v[np.isfinite(v)]
        key = "success" if part == "calibration" else "holdout"
        if len(v) == 0:
            continue
        auc = auc_vs_background(bs, v)
        lo, hi = auc_ci(bs, v, ec["n_boot"], rng)
        out[f"{key}_n"] = int(len(v))
        out[f"{key}_auc"] = auc
        out[f"{key}_auc_ci_low"], out[f"{key}_auc_ci_high"] = lo, hi
        null = bg[rng.integers(0, len(bg), (ec["n_null"], len(v)))]
        out[f"{key}_auc_null95"] = float(np.percentile(null.mean(1), 100 * ec["signal_rule"]["null_quantile"]))
        out[f"{key}_auc_p_vs_random"] = float((null.mean(1) >= auc).mean())
        for f in ec["capture_fractions"]:
            cap = float((v >= 1 - f).mean())
            nc = (null >= 1 - f).mean(1)
            boot = [(v[rng.integers(0, len(v), len(v))] >= 1 - f).mean() for _ in range(ec["n_boot"])]
            tag = f"capture_{int(round(f * 100))}"
            out[f"{key}_{tag}"] = cap
            out[f"{key}_{tag}_ci_low"] = float(np.percentile(boot, 2.5))
            out[f"{key}_{tag}_null95"] = float(np.percentile(nc, 100 * ec["signal_rule"]["null_quantile"]))
            out[f"{key}_{tag}_p_vs_random"] = float((nc >= cap).mean())
        area = np.linspace(0, 1, 201)
        curves[part] = (area, np.array([(v >= 1 - a).mean() for a in area]))
    return out, curves


def _verdict(cfg, d):
    sr = cfg["evaluation"]["signal_rule"]
    fmax = max(cfg["evaluation"]["capture_fractions"])
    tag = f"capture_{int(round(fmax * 100))}"
    if "holdout_auc" not in d:
        return False
    return bool(d.get("holdout_auc_ci_low", 0) > sr["min_auc_ci_low"] and
                d.get(f"holdout_{tag}", 0) > d.get(f"holdout_{tag}_null95", 1))


# =========================================================================== orchestrator
def run_evaluation(cfg, reset_freeze=False):
    from .compare import load_maps, run_layer_count, sensor_summary, write_report
    from .features import Store
    ev = U.d_stage(cfg, "evaluation")
    store = Store(cfg)
    sets = [s for s in U.run_sets(cfg) if store.features_for(s)]
    domain = store.domain(sets[0])
    grid = store.grid
    occ_path = U.rpath(cfg, cfg["inputs"].get("known_occurrences"))
    if occ_path is None or not Path(occ_path).exists():
        log.warning("No known occurrences (inputs.known_occurrences): evaluation skipped. The comparison can only "
                    "describe stability and agreement until ground truth (or field results) exist.")
        write_report(cfg)
        return None
    occ_hash = U.file_hash(occ_path)
    lock_p, split_p = ev / "FREEZE.lock.json", ev / "occurrence_split.csv"
    if reset_freeze and lock_p.exists():
        arch = ev / f"FREEZE.lock.archived_{datetime.now():%Y%m%d_%H%M%S}.json"
        lock_p.rename(arch)
        log.warning("Freeze reset: previous lock archived as %s. Results after this are no longer a blind test "
                    "of the held-out set.", arch.name)
    lock = U.read_json(lock_p) if lock_p.exists() else None

    occ, rep = load_occurrences(cfg, grid, domain)
    if occ is None or len(occ) < 6:
        log.error("Too few usable occurrences (%s). Evaluation skipped.", rep)
        return None
    if lock:
        if lock["occurrence_file_hash"] != occ_hash:
            raise RuntimeError("The occurrence file changed after the freeze. Restore it, or run with --reset-freeze "
                               "(this ends the blind held-out test and is logged).")
        sp = pd.read_csv(split_p)
        occ = occ.merge(sp[["flat", "split", "block"]], on="flat", how="left")
        if occ.split.isna().any():
            raise RuntimeError("Occurrences no longer match the frozen split (domain or masks changed?). Use "
                               "--reset-freeze only if you accept losing the blind held-out test.")
    else:
        occ = spatial_split(cfg, occ, grid)
        occ.to_csv(split_p, index=False)
    rep.update(calibration=int((occ.split == "calibration").sum()), holdout=int((occ.split == "holdout").sum()),
               blocks=int(occ.block.nunique()))
    U.jdump(rep, ev / "occurrence_report.json")
    log.info("Occurrences: %s", rep)
    try:
        import geopandas as gpd
        gpd.GeoDataFrame(occ, geometry=gpd.points_from_xy(occ.x, occ.y), crs=grid.crs).to_file(
            ev / "occurrences_used.gpkg", driver="GPKG")
    except Exception as e:  # pragma: no cover
        log.debug("gpkg write failed: %s", e)

    # per-layer evidence (calibration only) ------------------------------------------------------
    calib = occ[occ.split == "calibration"]
    le = layer_evidence(cfg, store, domain, calib, sets)
    le.to_csv(ev / "layer_evidence.csv", index=False)
    with pd.ExcelWriter(ev / "layer_evidence.xlsx", engine="openpyxl") as xw:
        pd.DataFrame({"README": [
            "Per-layer evidence against CALIBRATION occurrences only (held-out occurrences are never used here).",
            "auc: P(occurrence pixel more favourable than a random domain pixel) in the DECLARED direction. 0.5 = no signal.",
            "reading: 'CONTRADICTS declared direction' = the 95% CI is below 0.5: the layer's direction is probably wrong.",
            "WofE: binary pattern = top 10/20/30% of the layer. Contrast C = W+ - W-; studentized C = C / s(C).",
            "Pixels are unit cells and nearby occurrences are spatially correlated: p-values and studentized C are optimistic.",
            "best_studentized_C picks the threshold on the same data: use it to rank layers, not as a significance test.",
            "Use this table to refine directions/weights. Then re-run models; the map is scored on the held-out set only."]}
        ).to_excel(xw, sheet_name="README", index=False)
        le.to_excel(xw, sheet_name="all_layers", index=False)
        for s, g in le.groupby("feature_set"):
            g.to_excel(xw, sheet_name=f"{U.set_dir(cfg, s)}"[:31], index=False)
    log.info("Per-layer evidence: %d layers evaluated", len(le))

    # map-level ---------------------------------------------------------------------------------------
    maps, _ = load_maps(cfg, sets=sets)
    fps = {s: U.read_json(maps[s]["leaf"] / "fingerprint.json")["fingerprint"] for s in maps}
    first = lock is None
    rng = np.random.default_rng(cfg["project"]["seed"] + 9)
    rows, curves, revisions = [], {}, []
    for s in sets:
        if s not in maps:
            continue
        revised = (not first) and lock["fingerprints"].get(s) != fps[s]
        parts = ("holdout",) if revised else ("calibration", "holdout")
        m, cv = map_metrics(cfg, maps[s]["pct"], domain, occ, rng, parts)
        curves[s] = cv
        d = dict(set=s, n_layers=maps[s]["metrics"]["n_features"], status="REVISED: held-out only" if revised else "frozen",
                 fingerprint=fps[s], **m)
        pts = maps[s]["points"]
        if len(pts):
            dist = np.hypot(pts.Easting.values[:, None] - occ.x.values[None], pts.Northing.values[:, None] - occ.y.values[None])
            d["points_within_buffer_of_any_occurrence"] = float((dist.min(1) <= cfg["inputs"]["known_buffer_m"]).mean())
        d["carries_signal"] = _verdict(cfg, d)
        rows.append(d)
        if revised and fps[s] not in (lock.get("consulted", {}).get(s) or []):
            revisions.append(dict(time=datetime.now().isoformat(timespec="seconds"), set=s,
                                  frozen_fingerprint=lock["fingerprints"].get(s), new_fingerprint=fps[s], **m))
    comp = pd.DataFrame(rows)
    lbl = sensor_summary(cfg, {s: maps[s] for s in comp.set})
    comp = comp.merge(lbl[["set", "mc_stability_spearman", "mean_pct_sd_points", "drop_one_top_drivers"]], on="set", how="left")
    lead = ["set", "status", "n_layers", "carries_signal", "holdout_n", "holdout_auc", "holdout_auc_ci_low",
            "holdout_auc_ci_high", "holdout_auc_null95", "holdout_capture_5", "holdout_capture_5_null95",
            "holdout_capture_10", "holdout_capture_10_null95", "success_n", "success_auc", "success_capture_10"]
    comp = comp[[c for c in lead if c in comp] + [c for c in comp if c not in lead]]
    comp.to_csv(ev / "sensor_comparison.csv", index=False)
    with pd.ExcelWriter(ev / "sensor_comparison.xlsx", engine="openpyxl") as xw:
        comp.to_excel(xw, sheet_name="comparison", index=False)
        pd.DataFrame([rep]).to_excel(xw, sheet_name="occurrences", index=False)

    # curves -----------------------------------------------------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, part, title in zip(axs, ("calibration", "holdout"), ("Success-rate (calibration)", "Prediction-rate (held-out)")):
        for s, cv in curves.items():
            if part in cv:
                ax.plot(cv[part][0], cv[part][1], label=s)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="random")
        for f in cfg["evaluation"]["capture_fractions"]:
            ax.axvline(f, color="grey", lw=0.5, ls=":")
        ax.set(xlabel="fraction of area, most favourable first", ylabel="fraction of occurrences captured", title=title)
        ax.legend()
    fig.savefig(ev / "success_prediction_curves.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # equal layer count (calibration only) -----------------------------------------------------------
    try:
        run_layer_count(cfg, store, sets, ev, occ_flat=calib.flat.values, tag="_calibration")
    except Exception as e:
        log.warning("Layer-count test with occurrences failed: %s", e)

    # freeze bookkeeping ------------------------------------------------------------------------------
    if first:
        lock = dict(created=datetime.now().isoformat(timespec="seconds"), occurrence_file=str(occ_path),
                    occurrence_file_hash=occ_hash, split_file_hash=U.file_hash(split_p), fingerprints=fps,
                    holdout_consults=1)
        U.jdump(lock, lock_p)
        log.info("Weights FROZEN (FREEZE.lock.json). Later changes are scored on the held-out set only.")
    elif revisions:
        lock["holdout_consults"] = int(lock.get("holdout_consults", 1)) + 1
        for r in revisions:
            lock.setdefault("consulted", {}).setdefault(r["set"], []).append(r["new_fingerprint"])
        U.jdump(lock, lock_p)
        rl = ev / "revision_log.csv"
        pd.DataFrame(revisions).to_csv(rl, mode="a", header=not rl.exists(), index=False)
        log.warning("Revised set(s) %s scored on the held-out set only (consult #%d, logged in revision_log.csv).",
                    [r["set"] for r in revisions], lock["holdout_consults"])
        if lock["holdout_consults"] > cfg["evaluation"]["max_holdout_consults"]:
            log.warning("The held-out set has been consulted %d times: repeated tuning against it makes it a "
                        "calibration set. Treat the field test as the real check.", lock["holdout_consults"])
    write_report(cfg)
    log.info("Sensor comparison:\n%s", comp[[c for c in lead if c in comp]].to_string(index=False))
    return comp
