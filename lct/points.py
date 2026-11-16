"""Point extraction and per-run output writing (rasters, GPKG, Excel, plots, config).

Points are extracted from the RANK-PERCENTILE map (score_pct.tif) so every feature set is ranked on the
same 0-1 scale over the same pixels, with the same number of points.
"""
from __future__ import annotations

import platform
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import ndimage as ndi

from . import utils as U

log = U.log


# =========================================================================== extraction
def nan_gaussian(a, sigma):
    """Gaussian smoothing that ignores NaN and never creates values outside the original valid area."""
    ok = np.isfinite(a)
    num = ndi.gaussian_filter(np.where(ok, a, 0.0).astype(np.float64), sigma)
    den = ndi.gaussian_filter(ok.astype(np.float64), sigma)
    return np.where(ok & (den > 0.2), num / np.maximum(den, 1e-9), np.nan).astype(np.float32)


def extract_points(score2d, sd2d, domain, grid, pc) -> pd.DataFrame:
    """Local maxima -> top-fraction -> greedy minimum-separation suppression -> rank by mean - k*SD."""
    sc = np.where(domain & np.isfinite(score2d), score2d, -np.inf)
    mx = ndi.maximum_filter(sc, size=pc["local_max_window_px"], mode="nearest")
    thr = np.quantile(score2d[domain & np.isfinite(score2d)], 1 - pc["top_fraction"])
    r, c = np.nonzero(np.isfinite(sc) & (sc >= mx) & (sc >= thr))
    order = np.argsort(-score2d[r, c], kind="stable")
    x, y = grid.xy(r, c)
    ax, ay, keep = [], [], []
    for i in order:
        if ax and np.min(np.hypot(np.asarray(ax) - x[i], np.asarray(ay) - y[i])) < pc["min_separation_m"]:
            continue
        keep.append(i)
        ax.append(x[i])
        ay.append(y[i])
        if pc.get("max_points") and len(keep) >= pc["max_points"]:
            break
    keep = np.asarray(keep, int)
    df = pd.DataFrame(dict(row=r[keep], col=c[keep], Easting=x[keep], Northing=y[keep],
                           Score_pct=score2d[r[keep], c[keep]], Score_pct_SD=sd2d[r[keep], c[keep]]))
    df["Lower_bound_score"] = df.Score_pct - pc["rank_k_sd"] * df.Score_pct_SD
    df = df.sort_values("Lower_bound_score", ascending=False).reset_index(drop=True)
    df["Rank"] = np.arange(1, len(df) + 1)
    return df


def add_geographic(df, grid):
    from pyproj import Transformer
    t = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
    lon, lat = t.transform(df.Easting.values, df.Northing.values)
    df["Lat"], df["Lon"] = lat, lon
    df["UTM_zone"], df["EPSG"] = U.utm_label(grid.crs), grid.crs.to_epsg()
    return df


# =========================================================================== tables
def _reason(row, feats, fam_share, pc, unc_label, flags_on):
    ph = pc["direction_phrases"]
    parts = [f"{ph[d]} {lab} (value at {pct:.0f}th percentile; {sh:.0f}% of the signal)"
             for lab, d, pct, sh in feats if sh > 0]
    fams = ", ".join(f"{f} ({s:.0f}%)" for f, s in fam_share[:2] if s > 0)
    txt = (f"Rank {int(row.Rank)}, percentile {100 * row.Score_pct:.1f} +/- {100 * row.Score_pct_SD:.1f} "
           f"({unc_label} uncertainty).")
    if parts:
        txt += " Main drivers: " + "; ".join(parts) + "."
    if fams:
        txt += f" Strongest evidence families: {fams}."
    if flags_on:
        txt += " Caution: " + ", ".join(flags_on) + "."
    return txt


def build_tables(cfg, store, res, pts):
    pc = cfg["points"]
    prep, grid = res.prep, store.grid
    W = grid.width
    pos = np.searchsorted(prep.flat_idx, pts.row.values * W + pts.col.values)
    C = res.contrib_fn(pos) if len(pos) else np.zeros((0, len(prep.features)))
    Qp = prep.Q[pos] * 100.0
    feats = np.array(prep.features)
    fam = prep.meta.family.values
    dirs = prep.meta.direction.values
    labels = [store.dic.loc[f, "label"] or f for f in prep.features]
    pos_sum = np.clip(C, 0, None).sum(1, keepdims=True)
    share = np.where(pos_sum > 0, 100 * C / np.where(pos_sum > 0, pos_sum, 1), 0)
    fams = list(dict.fromkeys(fam))
    famC = np.column_stack([C[:, fam == f].sum(1) for f in fams]) if len(pos) else np.zeros((0, len(fams)))
    fam_pos = np.clip(famC, 0, None).sum(1, keepdims=True)
    famS = np.where(fam_pos > 0, 100 * famC / np.where(fam_pos > 0, fam_pos, 1), 0)

    out = pts.copy()
    out.insert(0, "Point_ID", [f"{res.model}-{res.set_name}-{i:03d}" for i in out.Rank])
    out.insert(1, "Model", res.model)
    out.insert(2, "Feature_Set", res.set_name)
    out.insert(3, "N_features", len(prep.features))
    out = add_geographic(out, grid)
    sd_q = np.quantile(out.Score_pct_SD, [1 / 3, 2 / 3]) if len(out) else [0, 0]
    unc = np.where(out.Score_pct_SD <= sd_q[0], "low", np.where(out.Score_pct_SD <= sd_q[1], "moderate", "high"))
    out["Uncertainty_class"] = unc

    for n, a in store.flags.items():                         # outcrop / cover / TIR flags
        out[f"flag_{n}"] = a[out.row.values, out.col.values]
    for n in pc.get("context_layers", []) or []:             # e.g. distance to lineament
        p = U.d_fe(cfg, "engineered") / "engineered_stack.tif"
        a = U.read_band_by_name(p, n) if p.exists() else None
        if a is not None:
            out[n] = a[out.row.values, out.col.values]
    for k, v in res.extra.get("point_columns", {}).items():
        out[k] = v[pos]

    n_top = pc["top_n_features"]
    flag_cols = [c for c in out.columns if c.startswith("flag_")]
    reasons, fam_cols = [], {}
    for i in range(len(out)):
        order = np.argsort(-C[i])[:n_top]
        for k, j in enumerate(order, 1):
            out.loc[i, f"Top{k}_feature"] = feats[j]
            out.loc[i, f"Top{k}_label"] = labels[j]
            out.loc[i, f"Top{k}_family"] = fam[j]
            out.loc[i, f"Top{k}_direction"] = dirs[j]
            out.loc[i, f"Top{k}_value_pctile"] = Qp[i, j]
            out.loc[i, f"Top{k}_contribution"] = C[i, j]
            out.loc[i, f"Top{k}_share_pct"] = share[i, j]
        fs = sorted(zip(fams, famS[i]), key=lambda t: -t[1])
        on = [c.replace("flag_", "").replace("_", " ") for c in flag_cols if out.loc[i, c]]
        reasons.append(_reason(out.iloc[i], [(labels[j], dirs[j], Qp[i, j], share[i, j]) for j in order],
                               fs, pc, unc[i], on))
    for k, f in enumerate(fams):
        out[f"Fam_{f}_share_pct"] = famS[:, k]
    out["Reasoning"] = reasons

    fam_tab = out[["Point_ID", "Rank"]].copy()
    for k, f in enumerate(fams):
        fam_tab[f"{f}_contribution"] = famC[:, k]
        fam_tab[f"{f}_share_pct"] = famS[:, k]
    long = pd.DataFrame(dict(Point_ID=np.repeat(out.Point_ID.values, len(feats)),
                             feature=np.tile(feats, len(out)), label=np.tile(labels, len(out)),
                             family=np.tile(fam, len(out)), direction=np.tile(dirs, len(out)),
                             value_pctile=Qp.ravel(), contribution=C.ravel(), share_pct=share.ravel()))
    return out, fam_tab, long, fams, famC


# =========================================================================== writers
def _write_excel(path, readme, points, fam_tab, field_cols, drop_cols):
    pts = points.drop(columns=[c for c in drop_cols if c in points.columns])
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        pd.DataFrame({"README": readme}).to_excel(xw, sheet_name="README_Method", index=False)
        pts.to_excel(xw, sheet_name="Points", index=False)
        fam_tab.to_excel(xw, sheet_name="Family_Contributions", index=False)
        ft = pts[["Point_ID", "Easting", "Northing", "Lat", "Lon"]].copy()
        for c in field_cols:
            ft[c] = ""
        ft.to_excel(xw, sheet_name="Field_Template", index=False)
        for ws in xw.book.worksheets:
            ws.freeze_panes = "A2"
            for col in ws.columns:
                w = max(len(str(col[0].value or "")), 8)
                ws.column_dimensions[col[0].column_letter].width = min(w + 2, 60)
        xw.book["README_Method"].column_dimensions["A"].width = 140


def _plots(leaf, res, score2d, domain, pts, long, weights):
    pd_ = leaf / "plots"
    pd_.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(np.where(domain, score2d, np.nan), cmap="viridis", vmin=0, vmax=1)
    ax.scatter(pts.col, pts.row, s=14, facecolor="none", edgecolor="red", linewidth=0.8)
    ax.set_title(f"{res.model} | {res.set_name}: score and {len(pts)} points")
    ax.set_xticks([]), ax.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.8, label="favourability percentile")
    fig.savefig(pd_ / "score_map.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    top = weights.sort_values("weight", ascending=False).head(25)
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(top) + 1.5))
    ax.barh(top.feature[::-1], top.weight[::-1])
    ax.set_title("Feature weights / importance (top 25)")
    fig.savefig(pd_ / "feature_weights.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    if len(long):
        m = long.groupby("family").share_pct.mean().sort_values()
        fig, ax = plt.subplots(figsize=(6, 0.4 * len(m) + 1.5))
        ax.barh(m.index, m.values)
        ax.set_title("Mean share of signal by evidence family (extracted points)")
        fig.savefig(pd_ / "family_contribution.png", dpi=140, bbox_inches="tight")
        plt.close(fig)


def write_leaf(cfg, store, res) -> dict:
    import geopandas as gpd
    grid, prep = store.grid, res.prep
    leaf = U.d_leaf(cfg, res.model, res.set_name)
    H, W = grid.shape

    def to2d(v):
        f = np.full(H * W, np.nan, np.float32)
        f[prep.flat_idx] = v
        return f.reshape(H, W)
    raw2d = to2d(res.score_mean)
    U.write_raster(leaf / "score_raw.tif", raw2d, grid, descriptions=["raw fuzzy gamma score (not comparable across sets)"])
    U.write_raster(leaf / "score_raw_sd.tif", to2d(res.score_sd), grid)
    pct = res.pct_mean if res.pct_mean is not None else U.pct_rank(res.score_mean)
    psd = res.pct_sd if res.pct_sd is not None else np.zeros_like(pct)
    score2d, sd2d = to2d(pct), to2d(psd)
    sig = cfg["points"].get("smooth_sigma_px", 0)
    if sig and sig > 0:        # pixel-level ratio noise is not a pegmatite-scale signal: smooth before ranking
        U.write_raster(leaf / "score_pct_unsmoothed.tif", score2d, grid)
        sm = nan_gaussian(score2d, sig)
        ok = prep.domain & np.isfinite(sm)
        score2d = np.full((H, W), np.nan, np.float32)
        score2d[ok] = U.pct_rank(sm[ok])                    # re-rank so the smoothed map is still a percentile
        sd2d = nan_gaussian(sd2d, sig)
    U.write_raster(leaf / "score_pct.tif", score2d, grid, descriptions=["rank percentile of favourability (0-1)"])
    U.write_raster(leaf / "score_pct_sd.tif", sd2d, grid, descriptions=["Monte-Carlo SD of the rank percentile"])
    tier = np.zeros((H, W), np.uint8)
    qs = sorted(cfg["points"]["tier_quantiles"])
    for k, q in enumerate(qs, 1):
        tier[score2d >= q] = k
    tier[~np.isfinite(score2d)] = 0
    U.write_raster(leaf / "tier.tif", tier, grid, nodata=0, dtype="uint8")

    pts = extract_points(score2d, sd2d, prep.domain, grid, cfg["points"])
    out, fam_tab, long, fams, famC = build_tables(cfg, store, res, pts)

    out.to_csv(leaf / "points.csv", index=False)
    long.to_csv(leaf / "contributions.csv", index=False)
    if len(out):
        g = gpd.GeoDataFrame(out.drop(columns=["row", "col"]), geometry=gpd.points_from_xy(out.Easting, out.Northing),
                             crs=grid.crs)
        g.to_file(leaf / "prospective_points.gpkg", driver="GPKG")
    ex = res.extra
    readme = [
        f"Model {res.model} | feature set: {res.set_name} | project: {cfg['project']['name']}",
        "Scores are RANK PERCENTILES of favourability for LCT pegmatite (0-1) over the common analysis area, not probabilities.",
        f"Features ({len(prep.features)}): " + ", ".join(prep.features),
        "No labels were used to build this map. Known occurrences are used only in the separate evaluation stage.",
        f"Contribution meaning: {ex.get('contribution_note', '')}. Contributions explain the model score, not geological cause.",
        "Reasoning text is a pointer for the field geologist, not proof. Feature directions are declared in feature_dictionary.xlsx.",
        "Rank is by Lower_bound_score = Score_pct - k*Score_pct_SD (k=%s). Score_pct_SD = Monte-Carlo SD of the percentile." % cfg["points"]["rank_k_sd"],
        f"Points: local maxima in the top {cfg['points']['top_fraction'] * 100:.0f}% of the scorable area, >= {cfg['points']['min_separation_m']} m apart.",
        f"Scorable pixels: {prep.flat_idx.size} ({100 * prep.flat_idx.size / max(int(store.aoi.sum()), 1):.1f}% of AOI). Masked pixels are never scored.",
        "Flag_* columns mark low-confidence ground (e.g. partial vegetation cover); a point on cover may not show outcrop.",
        f"Stability vs ensemble mean: Spearman={res.stability.get('mean_spearman', float('nan')):.3f}, top-overlap={res.stability.get('mean_top_overlap', float('nan')):.3f}.",
    ]
    _write_excel(leaf / "Prospective_Points.xlsx", readme, out, fam_tab, cfg["points"]["field_columns"], ["row", "col"])
    _plots(leaf, res, score2d, prep.domain, pts, long, res.weights)

    ok = np.isfinite(score2d)
    metrics = dict(model=res.model, feature_set=res.set_name, n_features=len(prep.features),
                   n_families=int(prep.meta.family.nunique()), features=prep.features,
                   n_scorable_pixels=int(prep.flat_idx.size), n_points=int(len(out)),
                   raw_score_mean=float(np.mean(res.score_mean)), raw_score_sd=float(np.std(res.score_mean)),
                   mean_pct_sd_all=float(np.nanmean(sd2d[ok])),
                   mean_pct_sd_points=float(out.Score_pct_SD.mean()) if len(out) else None,
                   stability=res.stability, runtime_s=round(res.runtime_s, 1),
                   extra={k: v for k, v in ex.items() if k not in ("point_columns", "spec")},
                   top_weights=res.weights.sort_values("weight", ascending=False).head(10).to_dict("records"))
    U.jdump(metrics, leaf / "metrics.json")
    res.weights.to_csv(leaf / "feature_weights.csv", index=False)
    public = {k: v for k, v in cfg.items() if not k.startswith("_")}
    (leaf / "run_config.yaml").write_text(yaml.safe_dump(dict(
        config=public, features=prep.features, seed=cfg["project"]["seed"], created=datetime.now().isoformat(timespec="seconds"),
        python=sys.version.split()[0], platform=platform.platform()), sort_keys=False, allow_unicode=True), encoding="utf-8")
    pos = np.searchsorted(prep.flat_idx, pts.row.values * W + pts.col.values) if len(pts) else np.zeros(0, int)
    return dict(model=res.model, set=res.set_name, n_features=len(prep.features), n_points=int(len(out)),
                leaf=str(leaf), point_pos=pos)
