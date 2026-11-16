"""Stage 1 - label-free feature engineering, organised by FEATURE SET (one set per data source).

spectral check -> QA/inventory -> alignment -> masks/flags -> engineered features -> lineament provenance ->
audit & de-duplication SEPARATELY per set -> feature store. No labels are used anywhere in this module.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from scipy import ndimage as ndi
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata, skew

from . import utils as U

log = U.log


# =========================================================================== discovery / metadata
def sensor_to_set(cfg, sensor):
    """Feature set that takes this sensor's layers ('' when no set uses it)."""
    for s, spec in cfg["feature_sets"].items():
        if sensor in (spec.get("sensors") or []):
            return s
    return ""


def discover_layers(cfg) -> pd.DataFrame:
    pats = cfg["inputs"].get("raster_patterns", ["*.tif"])
    rows = []
    for gname, spec in cfg["inputs"]["raster_groups"].items():
        folder = U.rpath(cfg, spec["path"])
        if not folder.exists():
            log.warning("Raster group %s: folder not found: %s", gname, folder)
            continue
        files = sorted({p for pat in pats for p in folder.glob(pat)})
        for p in files:
            sensor = spec.get("sensor", gname)
            rows.append(dict(name=p.stem, path=str(p), group=gname, sensor=sensor,
                             feature_set=sensor_to_set(cfg, sensor)))
    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError("No rasters discovered. Check inputs.raster_groups in config.yaml")
    dup = df[df.duplicated("name", keep=False)]
    if len(dup):
        raise ValueError(f"Duplicate layer names across groups: {sorted(dup.name.unique())}")
    return df.set_index("name", drop=False)


def check_metadata(cfg) -> dict:
    """Parse acquisition dates from the metadata text files; flag ASTER scenes after the SWIR failure."""
    q = cfg["qa"]
    fail = pd.Timestamp(q["aster_swir_failure_date"])
    out = {}
    for key, rel in (cfg["inputs"].get("metadata_files") or {}).items():
        p = U.rpath(cfg, rel)
        if not Path(p).exists():
            out[key] = {"found": False}
            continue
        txt = Path(p).read_text(errors="ignore")
        dates = sorted(set(re.findall(q["date_regex"], txt)))
        entry = {"found": True, "dates": dates}
        if key.upper() == str(q["swir_sensor_key"]).upper() and dates:
            late = [d for d in dates if pd.Timestamp(d) >= fail]
            entry["swir_valid"] = len(late) == 0
            if late:
                log.warning("ASTER scenes acquired on/after %s: SWIR bands 4-9 are unreliable: %s", fail.date(), late)
            else:
                log.info("ASTER acquisition %s is before the SWIR failure (%s): SWIR bands usable.", dates, fail.date())
        out[key] = entry
    return out


def load_layer_table(cfg) -> pd.DataFrame:
    p = U.rpath(cfg, cfg["inputs"].get("layer_table"))
    if p is None or not Path(p).exists():
        log.warning("No layer_table found: families/directions rely on name_rules only.")
        return pd.DataFrame(columns=["combination", "target"])
    df = pd.read_csv(p, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Unique Name": "name", "Combination": "combination", "Target": "target"})
    for c in ("name", "combination", "target"):
        df[c] = df[c].astype(str).str.strip()
    return df.drop_duplicates("name").set_index("name")


_BAND_RE = re.compile(r"(?<![\d.])(\d{1,2})(?![\d.])")


def parse_bands(expr: str):
    """Band numbers in a combination string (labels before ':' and decimal constants are ignored)."""
    expr = str(expr).split(":")[-1]
    return sorted({int(m) for m in _BAND_RE.findall(expr)})


def classify_layers(cfg, layers: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    lc = cfg["layers"]
    excl = [re.compile(p, re.I) for p in lc.get("exclude_regex", [])]
    cat = [re.compile(p, re.I) for p in lc.get("categorical_regex", [])]
    name_rules = [(re.compile(r["match"], re.I), r) for r in cfg.get("name_rules", [])]
    target_rules = [(re.compile(r["match"], re.I), r) for r in cfg.get("target_rules", [])]
    dflt = cfg["defaults"]
    rows = []
    for name, r in layers.iterrows():
        combo = table.loc[name, "combination"] if name in table.index else ""
        target = table.loc[name, "target"] if name in table.index else ""
        bands = lc.get("band_overrides", {}).get(name) or (parse_bands(combo) if combo else [])
        is_tir_sensor = r.sensor == lc.get("tir_sensor", "ASTER")
        tir = bool(is_tir_sensor and any(b >= lc["tir_band_min"] for b in bands))
        mixed = bool(is_tir_sensor and tir and any(b < lc["tir_band_min"] for b in bands))
        rule, src = None, "default"
        for rx, rr in name_rules:
            if rx.search(name):
                rule, src = rr, "name_rule"
                break
        if rule is None and target:
            for rx, rr in target_rules:
                if rx.search(target):
                    rule, src = rr, "target_rule"
                    break
        rule = rule or {}
        weight = float(rule.get("weight", dflt["weight"])) * (lc["tir_weight_multiplier"] if tir else 1.0)
        reason = ""
        if any(rx.search(name) for rx in excl):
            reason = "excluded_by_regex"
        elif mixed and lc.get("exclude_mixed_units", True):
            reason = "mixed_units (reflectance and emissivity bands in one ratio)"
        label = rule.get("label") or (f"{name} ({combo}: {target})" if combo else name)
        rows.append(dict(name=name, path=r.path, group=r.group, feature_set=r.feature_set, sensor=r.sensor,
                         combination=combo, target=target, bands=",".join(map(str, bands)), tir=tir,
                         mixed_units=mixed, family=rule.get("family", dflt["family"]),
                         direction=rule.get("direction", dflt["direction"]), weight=weight,
                         center=rule.get("center", np.nan), width=rule.get("width", np.nan),
                         rule_source=src, label=label, categorical=any(rx.search(name) for rx in cat),
                         excluded_reason=reason, source="raster"))
    return pd.DataFrame(rows).set_index("name", drop=False)


# =========================================================================== alignment / inventory
def _stripe_ratio(a):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dr = np.nanmedian(np.abs(np.diff(a, axis=0)))
        dc = np.nanmedian(np.abs(np.diff(a, axis=1)))
    return float(dr / (dc + 1e-12)) if np.isfinite(dr) and np.isfinite(dc) else np.nan


def align_layers(cfg, cls: pd.DataFrame, grid: U.Grid, aoi: np.ndarray):
    """Read every usable layer onto the master grid and build the QA inventory."""
    lc, qc = cfg["layers"], cfg["qa"]
    aligned, rows = {}, []
    support = None
    sup = lc.get("tir_support_layer")
    if sup and sup in cls.index:
        support = U.read_to_grid(cls.loc[sup, "path"], grid, categorical=True)
    for name, r in cls.iterrows():
        info = dict(layer=name, group=r.group, family=r.family, direction=r.direction,
                    excluded_reason=r.excluded_reason, flags="")
        flags = []
        with rasterio.open(r.path) as src:
            info.update(crs=str(src.crs), native_res=round(abs(src.transform.a), 4), bands=src.count,
                        dtype=src.dtypes[0], nodata=src.nodata)
            if src.crs is not None and src.crs == grid.crs:
                ox = ((src.transform.c - grid.transform.c) / grid.res) % 1
                oy = ((src.transform.f - grid.transform.f) / grid.res) % 1
                info["grid_offset_px"] = round(float(max(min(ox, 1 - ox), min(oy, 1 - oy))), 3)
                if info["grid_offset_px"] > qc["grid_offset_flag_px"]:
                    flags.append("offset_from_master_grid(resampled)")
            else:
                flags.append("crs_differs(reprojected)")
            multiband = src.count > 1
        if multiband and lc.get("exclude_multiband", True) and not r.excluded_reason:
            cls.loc[name, "excluded_reason"] = "multiband"
            info["excluded_reason"] = "multiband"
        if info["excluded_reason"]:
            info["flags"] = ";".join(flags)
            rows.append(info)
            continue
        a = U.read_to_grid(r.path, grid, categorical=bool(r.categorical))
        if r.tir and support is not None:
            a = np.where(support > 0, a, np.nan).astype(np.float32)
        v = a[aoi & np.isfinite(a)]
        info["valid_pct_in_aoi"] = round(100 * v.size / max(int(aoi.sum()), 1), 2)
        if v.size:
            info.update(min=float(v.min()), max=float(v.max()), mean=float(v.mean()), std=float(v.std()))
            info["stripe_ratio"] = round(_stripe_ratio(np.where(aoi, a, np.nan)), 3)
            if v.std() == 0:
                flags.append("constant")
            sr = info["stripe_ratio"]
            if np.isfinite(sr) and (sr > qc["stripe_ratio_flag"] or sr < 1 / qc["stripe_ratio_flag"]):
                flags.append("possible_striping")
        if info["valid_pct_in_aoi"] < 100 * cfg["audit"]["min_valid_fraction"]:
            flags.append("low_valid_fraction")
        info["flags"] = ";".join(flags)
        rows.append(info)
        aligned[name] = a
    return aligned, pd.DataFrame(rows)


def get_layer(cfg, cls, grid, aligned, name):
    """Layer on the master grid, from memory if aligned, else read it (mask layers are not features)."""
    if name in aligned:
        return aligned[name]
    if name not in cls.index:
        raise KeyError(f"Layer '{name}' referenced in config but not found in the input folders")
    return U.read_to_grid(cls.loc[name, "path"], grid, categorical=bool(cls.loc[name, "categorical"]))


def build_masks_and_flags(cfg, cls, grid, aligned, aoi):
    masks, mmeta = {}, []
    for m in cfg["masks"].get("rules", []):
        a = get_layer(cfg, cls, grid, aligned, m["layer"])
        ok = np.isfinite(a) & aoi
        thr = float(np.nanpercentile(a[ok], m["value"])) if m["mode"] == "percentile" else float(m["value"])
        mk = U.apply_op(np.where(np.isfinite(a), a, np.nan), m["op"], thr) & np.isfinite(a) & aoi
        if m.get("buffer_px"):      # mixed pixels next to masked ground distort ratios and local anomalies
            mk = ndi.binary_dilation(mk, iterations=int(m["buffer_px"])) & aoi
        masks[m["name"]] = mk
        mmeta.append(dict(name=m["name"], layer=m["layer"], op=m["op"], threshold=thr,
                          applies_to=m.get("applies_to", list(cfg["feature_sets"])),
                          pct_of_aoi_masked=round(100 * mk.sum() / max(aoi.sum(), 1), 2)))
    for m in cfg["masks"].get("external", []) or []:
        p = U.rpath(cfg, m["path"])
        if not Path(p).exists():
            log.warning("External mask not found: %s", p)
            continue
        a = U.read_to_grid(p, grid, categorical=True)
        mk = np.isin(a, m["mask_values"]) & aoi
        masks[m["name"]] = mk
        mmeta.append(dict(name=m["name"], layer=str(p), op="in", threshold=m["mask_values"],
                          applies_to=m.get("applies_to", list(cfg["feature_sets"])),
                          pct_of_aoi_masked=round(100 * mk.sum() / max(aoi.sum(), 1), 2)))
    flags = {}
    for f in cfg.get("flags", []) or []:
        a = get_layer(cfg, cls, grid, aligned, f["layer"])
        ok = np.isfinite(a) & aoi
        thr = float(np.nanpercentile(a[ok], f["value"])) if f["mode"] == "percentile" else float(f["value"])
        flags[f["name"]] = U.apply_op(np.where(np.isfinite(a), a, np.nan), f["op"], thr) & np.isfinite(a)
    for m in mmeta:
        log.info("Mask %-14s masks %5.1f%% of AOI (threshold %s)", m["name"], m["pct_of_aoi_masked"], m["threshold"])
    return masks, mmeta, flags


# =========================================================================== engineered features
def _load_vector(cfg, key, grid):
    import geopandas as gpd
    spec = cfg["engineer"]["vector_sources"][key]
    p = U.rpath(cfg, spec["path"])
    if not Path(p).exists():
        log.warning("Vector source '%s' not found (%s): its features are skipped.", key, p)
        return None
    g = gpd.read_file(p)
    if g.crs is None:
        g = g.set_crs(spec.get("assume_crs", cfg["inputs"].get("assume_vector_crs", "EPSG:4326")))
    g = g.to_crs(grid.crs)
    return g[~g.geometry.isna() & ~g.geometry.is_empty]


def _segments(gdf):
    from shapely.geometry import LineString
    segs = []
    for geom in gdf.geometry:
        for ln in (list(geom.geoms) if hasattr(geom, "geoms") else [geom]):
            if ln.geom_type != "LineString":
                continue
            c = np.asarray(ln.coords)[:, :2]
            for a, b in zip(c[:-1], c[1:]):
                dx, dy = b - a
                if dx == 0 and dy == 0:
                    continue
                segs.append((LineString([a, b]), float(np.degrees(np.arctan2(dx, dy)) % 180)))
    return segs


def _burn(geoms, grid):
    geoms = list(geoms)
    if not geoms:
        return np.zeros(grid.shape, np.uint8)
    return rasterize([(g, 1) for g in geoms], out_shape=grid.shape, transform=grid.transform, fill=0,
                     all_touched=True, dtype="uint8")


def _line_intersections(gdf):
    from shapely.strtree import STRtree
    lines = [ln for g in gdf.geometry for ln in (list(g.geoms) if hasattr(g, "geoms") else [g])
             if ln.geom_type == "LineString"]
    tree, pts = STRtree(lines), []
    for i, ln in enumerate(lines):
        for j in tree.query(ln):
            if j <= i:
                continue
            x = ln.intersection(lines[j])
            if x.is_empty:
                continue
            if x.geom_type == "Point":
                pts.append(x)
            elif x.geom_type == "MultiPoint":
                pts.extend(x.geoms)
    return pts


def vector_features(cfg, grid):
    out, meta, cache = {}, [], {}
    dflt = cfg["defaults"]
    for spec in cfg["engineer"].get("vector_features", []) or []:
        src = spec["source"]
        fset = spec.get("feature_set", cfg["engineer"]["vector_sources"].get(src, {}).get("feature_set", "DEM"))
        if not fset:
            log.info("Vector feature %s skipped (source '%s' has feature_set: null)", spec["name"], src)
            continue
        if src not in cache:
            cache[src] = _load_vector(cfg, src, grid)
        gdf = cache[src]
        if gdf is None or gdf.empty:
            continue
        t, sigma = spec["type"], spec.get("radius_m", 0) / grid.res / 2.0
        if t == "distance":
            b = _burn(gdf.geometry, grid)
            if not b.any():
                log.warning("%s: no %s features inside the grid; skipped", spec["name"], src)
                continue
            arr = ndi.distance_transform_edt(b == 0, sampling=grid.res).astype(np.float32)
        elif t == "density":
            arr = ndi.gaussian_filter(_burn(gdf.geometry, grid).astype(np.float32), sigma)
        elif t == "orientation_density":
            lo, hi = spec["az_min"], spec["az_max"]
            segs = [g for g, az in _segments(gdf) if (lo <= az <= hi if lo <= hi else (az >= lo or az <= hi))]
            arr = ndi.gaussian_filter(_burn(segs, grid).astype(np.float32), sigma)
        elif t == "intersection_density":
            pts = _line_intersections(gdf)
            cnt = np.zeros(grid.shape, np.float32)
            if pts:
                r, c = grid.rc([p.x for p in pts], [p.y for p in pts])
                ok = (r >= 0) & (r < grid.height) & (c >= 0) & (c < grid.width)
                np.add.at(cnt, (r[ok], c[ok]), 1.0)
            arr = ndi.gaussian_filter(cnt, sigma)
        else:
            raise ValueError(f"Unknown vector feature type: {t}")
        out[spec["name"]] = arr.astype(np.float32)
        meta.append(dict(name=spec["name"], feature_set=fset, family=spec["family"],
                         direction=spec.get("direction", dflt["direction"]), weight=spec.get("weight", dflt["weight"]),
                         center=spec.get("center", np.nan), width=spec.get("width", np.nan),
                         label=spec.get("label", spec["name"]), tir=False, combination="", target="",
                         source="vector", excluded_reason="", categorical=False, sensor=f"vector:{src}", group="vector"))
    return out, meta


def focal_zscore(a, size, min_frac):
    valid = np.isfinite(a)
    a0 = np.where(valid, a, 0.0).astype(np.float64)
    n = ndi.uniform_filter(valid.astype(np.float64), size, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        m1 = ndi.uniform_filter(a0, size, mode="nearest") / n
        var = ndi.uniform_filter(a0 * a0, size, mode="nearest") / n - m1 ** 2
        floor = (1e-3 * np.nanstd(a)) ** 2
        z = (a - m1) / np.sqrt(np.maximum(var, floor))
    z[(n < min_frac) | ~valid] = np.nan
    return z.astype(np.float32)


def local_anomaly_features(cfg, cls, aligned, grid):
    lc = cfg["engineer"].get("local_anomaly", {})
    out, meta = {}, []
    if not lc.get("enabled", False):
        return out, meta
    for name, a in aligned.items():
        r = cls.loc[name]
        if (r.sensor not in lc.get("apply_to_sensors", ["ASTER", "S2"]) or not r.feature_set
                or r.direction not in ("high", "low") or r.family not in lc["apply_to_families"]
                or (lc.get("exclude_tir", True) and r.tir)):
            continue
        for w in lc["windows_m"]:
            size = max(3, int(round(w / grid.res)) // 2 * 2 + 1)
            new = f"{name}__lz{int(w)}"
            out[new] = focal_zscore(a, size, lc["min_valid_fraction"])
            d = r.to_dict()
            d.update(name=new, source="local_anomaly", label=f"{r.label} - local anomaly ({int(w)} m window)")
            meta.append(d)
    return out, meta


# =========================================================================== audit / selection
def _priority_key(cfg, meta, vf):
    sc = cfg["selection"]
    fam_rank = {f: i for i, f in enumerate(sc.get("family_priority", []))}
    pref = [re.compile(p) for p in sc.get("prefer_regex", [])]

    def key(n):
        m = meta.loc[n]
        pr = next((i for i, p in enumerate(pref) if p.search(n)), len(pref))
        return (fam_rank.get(m.family, len(fam_rank)), bool(m.tir), m.source != "raster", pr, -vf[n], n)
    return key


def _eff_dim(rho, fracs=(0.9, 0.95)):
    ev = np.sort(np.clip(np.linalg.eigvalsh(rho), 0, None))[::-1]
    cum = np.cumsum(ev) / ev.sum()
    d = {f"n_components_{int(f * 100)}pct": int(np.searchsorted(cum, f) + 1) for f in fracs}
    d["participation_ratio"] = float(ev.sum() ** 2 / (ev ** 2).sum())
    d["n_layers"] = int(len(ev))
    return d


def audit_select(cfg, pool, meta, domain, outdir: Path, veg=None):
    ac, sc = cfg["audit"], cfg["selection"]
    rng = np.random.default_rng(cfg["project"]["seed"])
    names = list(pool)
    vf = {n: float(np.isfinite(pool[n][domain]).mean()) for n in names}
    status = {}
    for n in names:
        v = pool[n][domain & np.isfinite(pool[n])]
        if meta.loc[n, "direction"] == "neutral":
            status[n] = "context_only (direction=neutral)"
        elif vf[n] < ac["min_valid_fraction"]:
            status[n] = f"low_valid_fraction ({vf[n]:.2f})"
        elif v.size == 0 or np.nanstd(v) == 0:
            status[n] = "constant"
    cand = [n for n in names if n not in status]
    if len(cand) < 2:
        raise RuntimeError("Fewer than 2 scoring candidates. Check direction rules and masks.")
    idx = np.flatnonzero(domain.ravel())
    take = rng.choice(idx, size=min(len(idx), ac["max_samples"]), replace=False)
    Xs = np.column_stack([pool[n].ravel()[take] for n in cand]).astype(np.float64)
    med = np.nanmedian(Xs, axis=0)
    Xs = np.where(np.isfinite(Xs), Xs, med)
    Rk = rankdata(Xs, axis=0)
    rho = np.nan_to_num(np.corrcoef(Rk, rowvar=False))
    veg_corr = {}
    if veg:                                   # vegetation contamination: Spearman vs the NDVI of the SAME sensor/date
        rvs = {}
        for k, arr in veg.items():
            v = arr.ravel()[take].astype(float)
            rvs[k] = rankdata(np.where(np.isfinite(v), v, np.nanmedian(v)))
        for i, n in enumerate(cand):
            k = meta.loc[n, "sensor"]
            if k in rvs:
                veg_corr[n] = float(np.nan_to_num(np.corrcoef(Rk[:, i], rvs[k])[0, 1]))
    vt = ac.get("vegetation_corr_flag", 0.6)
    pos = {n: i for i, n in enumerate(cand)}
    key = _priority_key(cfg, meta, vf)
    rep_rows = []
    prot = [re.compile(p) for p in sc.get("protect", []) or []]
    is_prot = lambda n: any(p.search(n) for p in prot)

    # 1. exact monotone duplicates / reciprocals (|Spearman| ~ 1)
    parent = {n: n for n in cand}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    ii, jj = np.where(np.triu(np.abs(rho) >= ac["rank_duplicate_threshold"], 1))
    for i, j in zip(ii, jj):
        parent[find(cand[j])] = find(cand[i])
    groups = {}
    for n in cand:
        groups.setdefault(find(n), []).append(n)
    for g in groups.values():
        if len(g) < 2:
            continue
        rep = sorted(g, key=key)[0]
        for n in g:
            if n == rep:
                continue
            r = rho[pos[n], pos[rep]]
            kind = "reciprocal(direction-conflict-check)" if r < 0 else "monotone_duplicate"
            status[n] = f"duplicate_of:{rep} ({kind}, rho={r:.4f})"
            rep_rows.append(dict(layer=n, duplicate_of=rep, rho=r, kind=kind))
    pd.DataFrame(rep_rows).to_csv(outdir / "rank_duplicates.csv", index=False)

    # 1b. layers dominated by vegetation (dry litter/woodland in the SWIR) are not rock evidence
    if ac.get("vegetation_action", "flag") == "drop":
        for n in cand:
            if n not in status and abs(veg_corr.get(n, 0.0)) >= vt:
                status[n] = f"vegetation_contaminated (|rho vs NDVI|={abs(veg_corr[n]):.2f})"

    # 2. collinear clusters inside each family (keeps family coverage)
    surv = [n for n in cand if n not in status]
    clus_rows = []
    for fam in sorted({meta.loc[n, "family"] for n in surv}):
        mem = [n for n in surv if meta.loc[n, "family"] == fam]
        if len(mem) < 2:
            clus_rows.append(dict(family=fam, cluster=f"{fam}-1", layer=mem[0], representative=mem[0], rho_to_rep=1.0))
            continue
        ix = [pos[n] for n in mem]
        dist = 1 - np.abs(rho[np.ix_(ix, ix)])
        dist = np.clip((dist + dist.T) / 2, 0, None)
        np.fill_diagonal(dist, 0)
        labs = fcluster(linkage(squareform(dist, checks=False), method="average"),
                        t=1 - ac["corr_threshold"], criterion="distance")
        for c in np.unique(labs):
            cm = [m for m, l in zip(mem, labs) if l == c]
            rep = sorted(cm, key=key)[0]
            for n in cm:
                r = abs(rho[pos[n], pos[rep]])
                clus_rows.append(dict(family=fam, cluster=f"{fam}-{c}", layer=n, representative=rep, rho_to_rep=r))
                if n != rep:
                    status[n] = f"collinear_with:{rep} (|rho|={r:.2f})"
    pd.DataFrame(clus_rows).to_csv(outdir / "clusters.csv", index=False)

    # 3. iterative VIF pruning (never removes the last feature of a family)
    sel = [n for n in cand if n not in status]
    vif_rows = []
    min_fam = sc.get("min_per_family", 1)
    while len(sel) > 2:
        ix = [pos[n] for n in sel]
        vif = np.diag(np.linalg.pinv(rho[np.ix_(ix, ix)]))
        drop = None
        for k in np.argsort(-vif):
            if vif[k] <= ac["vif_threshold"]:
                break
            if is_prot(sel[k]):
                continue
            fam = meta.loc[sel[k], "family"]
            if sum(meta.loc[n, "family"] == fam for n in sel) > min_fam:
                drop, vmax = sel[k], float(vif[k])
                break
        if drop is None:
            break
        status[drop] = f"vif_pruned (VIF={vmax:.1f})"
        vif_rows.append(dict(step=len(vif_rows), dropped=drop, vif=vmax))
        sel.remove(drop)
    caps = sc.get("max_per_family") or {}
    for fam in list(dict.fromkeys(meta.loc[n, "family"] for n in sel)):
        cap = caps.get(fam, caps.get("default"))
        mem = [n for n in sel if meta.loc[n, "family"] == fam]
        while cap and len(mem) > cap:
            ixm = [pos[n] for n in mem]
            sub = np.abs(rho[np.ix_(ixm, ixm)])
            np.fill_diagonal(sub, 0)
            red = sub.mean(1)
            red[[is_prot(n) for n in mem]] = -1.0
            if red.max() < 0:
                break
            worst = mem[int(np.argmax(red))]
            status[worst] = f"family_cap ({fam} <= {cap}; most redundant removed)"
            mem.remove(worst)
            sel.remove(worst)
    ix = [pos[n] for n in sel]
    final_vif = np.diag(np.linalg.pinv(rho[np.ix_(ix, ix)]))
    pd.DataFrame(dict(layer=sel, VIF=final_vif)).to_csv(outdir / "vif_selected.csv", index=False)
    pd.DataFrame(vif_rows).to_csv(outdir / "vif_steps.csv", index=False)
    pd.DataFrame(rho, index=cand, columns=cand).to_csv(outdir / "spearman_candidates.csv")
    U.jdump({"all_candidates": _eff_dim(rho), "selected": _eff_dim(rho[np.ix_(ix, ix)]),
             "note": "Effective independent dimensions of the spectral/terrain evidence (Spearman eigenvalues)."},
            outdir / "dimensionality.json")
    report = pd.DataFrame([dict(layer=n, family=meta.loc[n, "family"], direction=meta.loc[n, "direction"],
                                feature_set=meta.loc[n, "feature_set"], valid_fraction=vf[n],
                                veg_corr=veg_corr.get(n, np.nan),
                                status="SELECTED" if n in sel else "dropped",
                                reason="" if n in sel else status.get(n, "")) for n in names])
    report.to_csv(outdir / "selection_report.csv", index=False)
    vflag = [n for n in sel if abs(veg_corr.get(n, 0)) >= vt]
    if vflag:
        log.warning("%d selected features track vegetation (|rho vs NDVI| >= %.2f): %s", len(vflag), vt, ", ".join(vflag[:10]) + (" ..." if len(vflag) > 10 else ""))
    log.info("Audit [%s]: %d layers -> %d candidates -> %d selected (%d independent dims at 90%% variance)",
             outdir.name, len(names), len(cand), len(sel), _eff_dim(rho[np.ix_(ix, ix)])["n_components_90pct"])
    return sel, report


def maybe_transform(arr, domain, skew_thr):
    v = arr[domain & np.isfinite(arr)]
    if v.size > 50 and v.min() >= 0 and skew(v) > skew_thr:
        return np.log1p(arr).astype(np.float32), "log1p"
    return arr, "none"


# =========================================================================== lineament provenance
def _edge_pct(a, aoi):
    """Percentile rank of gradient magnitude (Sobel) inside the AOI."""
    from scipy.stats import rankdata
    a = np.where(np.isfinite(a), a, np.nanmedian(a[np.isfinite(a)]) if np.isfinite(a).any() else 0.0)
    a = ndi.gaussian_filter(a.astype(np.float64), 1.0)
    g = np.hypot(ndi.sobel(a, 0), ndi.sobel(a, 1))
    out = np.full(a.shape, np.nan, np.float32)
    out[aoi] = (rankdata(g[aoi]) - 0.5) / aoi.sum()
    return out


def lineament_provenance(cfg, cls, grid, aligned, aoi, outdir):
    """Report what the lineament features are made of: declared source + an edge-coincidence heuristic.

    Lineaments drawn on a DEM/hillshade sit on DEM breaks; lineaments drawn on imagery sit on image edges.
    The mean edge-strength percentile along the lineaments (0.5 = no relation) is computed for each.
    """
    vs = cfg["engineer"].get("vector_sources", {}).get("lineaments")
    pc = cfg["engineer"].get("provenance_check", {}) or {}
    if not vs:
        return {}
    rep = dict(declared_derived_from=vs.get("derived_from", "unknown"), assigned_feature_set=vs.get("feature_set"),
               path=str(U.rpath(cfg, vs["path"])))
    xml = Path(str(rep["path"]) + ".xml")
    if xml.exists():                                     # ArcGIS metadata often names the source raster
        txt = xml.read_text(errors="ignore").lower()
        rep["metadata_keywords"] = [k for k in ("dem", "srtm", "alos", "hillshade", "sentinel", "landsat", "aster",
                                                "imagery", "manual", "digitis", "digitiz") if k in txt]
    g = _load_vector(cfg, "lineaments", grid)
    if g is None or g.empty:
        rep["note"] = "lineament file not found"
        U.jdump(rep, outdir / "lineament_provenance.json")
        return rep
    lin = _burn(g.geometry, grid).astype(bool) & aoi
    rep["n_lineament_pixels"] = int(lin.sum())
    scores = {}
    try:
        scores["DEM"] = get_layer(cfg, cls, grid, aligned, pc.get("dem_layer", "DEM_utm"))
    except KeyError as e:
        log.warning("Provenance check: %s", e)
    for sensor, lname in (pc.get("image_layers") or {}).items():
        try:
            scores[sensor] = get_layer(cfg, cls, grid, aligned, lname)
        except KeyError as e:
            log.warning("Provenance check: %s", e)
    rng = np.random.default_rng(cfg["project"]["seed"])
    for k, a in scores.items():
        e = _edge_pct(a, aoi)
        on = float(np.nanmean(e[lin])) if lin.any() else np.nan
        null = [float(np.nanmean(np.roll(np.roll(e, int(rng.integers(20, 200)), 0), int(rng.integers(20, 200)), 1)[lin]))
                for _ in range(50)]                         # shifted lineaments keep their geometry, lose position
        rep[f"edge_pct_on_lineaments_{k}"] = on
        rep[f"edge_pct_shifted_null95_{k}"] = float(np.nanpercentile(null, 95))
    dem_e = rep.get("edge_pct_on_lineaments_DEM", np.nan)
    img = {k: v for k, v in rep.items() if k.startswith("edge_pct_on_lineaments_") and not k.endswith("_DEM")}
    best_img = max(img.values()) if img else np.nan
    # a source "explains" the lineaments only if they sit on its edges more than shifted copies of themselves do
    above = {k.replace("edge_pct_on_lineaments_", ""): v > rep[k.replace("on_lineaments", "shifted_null95")]
             for k, v in rep.items() if k.startswith("edge_pct_on_lineaments_")}
    dem_ok = above.get("DEM", False)
    img_ok = any(v for k, v in above.items() if k != "DEM")
    rep["sources_with_edge_coincidence_above_null"] = [k for k, v in above.items() if v]
    if rep["declared_derived_from"] in ("DEM", "imagery"):
        rep["conclusion"] = f"declared: {rep['declared_derived_from']}"
    elif dem_ok and not img_ok:
        rep["conclusion"] = "likely DEM-derived (heuristic)"
    elif img_ok and not dem_ok:
        rep["conclusion"] = "likely imagery-derived (heuristic)"
    else:
        rep["conclusion"] = "undetermined: ask whoever digitised them, then set derived_from"
    imagery = rep["declared_derived_from"] == "imagery" or "imagery" in rep["conclusion"]
    if imagery and rep["assigned_feature_set"] == "DEM":
        rep["warning"] = ("Lineaments look imagery-derived but feed the DEM set: the DEM-only run is not DEM-only. "
                          "Set engineer.vector_sources.lineaments.feature_set to the image set or null.")
        log.warning(rep["warning"])
    elif rep["conclusion"].startswith("undetermined"):
        log.warning("Lineament provenance undetermined (DEM edge pct %.2f vs image %.2f). Set derived_from in config.",
                    dem_e, best_img)
    else:
        log.info("Lineament provenance: %s (DEM edge pct %.2f, best image %.2f)", rep["conclusion"], dem_e, best_img)
    U.jdump(rep, outdir / "lineament_provenance.json")
    return rep


# =========================================================================== store
class Store:
    """Reader for the stored feature sets (what the model stage consumes)."""

    def __init__(self, cfg):
        d, da = U.d_fe(cfg, "store"), U.d_fe(cfg, "aligned")
        self.arrays, self.names, self.grid = U.read_stack(d / "ALL_features.tif")
        self.dic = pd.read_excel(d / "feature_dictionary.xlsx", sheet_name="features", index_col=0).fillna("")
        marr, mnames, _ = U.read_stack(da / "masks.tif") if (da / "masks.tif").exists() else (np.zeros((0,) + self.grid.shape), [], None)
        self.masks = {n: marr[i] > 0 for i, n in enumerate(mnames)}
        self.mask_applies = {m["name"]: m["applies_to"] for m in U.read_json(da / "masks_meta.json")} if (da / "masks_meta.json").exists() else {}
        self.aoi = U.read_stack(da / "aoi_mask.tif")[0][0] > 0
        farr, fnames, _ = U.read_stack(da / "flags.tif") if (da / "flags.tif").exists() else (np.zeros((0,) + self.grid.shape), [], None)
        self.flags = {n: farr[i] > 0 for i, n in enumerate(fnames)}
        self.cfg = cfg
        self._common = None

    def features_for(self, set_name):
        return [n for n in self.dic.index if self.dic.loc[n, "feature_set"] == set_name]

    def own_domain(self, set_name):
        """AOI minus the masks that apply to this set (its own vegetation date, water...)."""
        dom = self.aoi.copy()
        for n, mk in self.masks.items():
            if set_name in self.mask_applies.get(n, []):
                dom &= ~mk
        return dom

    def common_domain(self, sets):
        """Pixels every set can score: AOI minus every set's masks, and every selected feature finite."""
        dom = self.aoi.copy()
        for s in sets:
            dom &= self.own_domain(s)
            f = self.features_for(s)
            if f:
                dom &= np.all(np.isfinite(self.stack(f)), axis=0)
        return dom

    def domain(self, set_name):
        if self.cfg.get("comparison", {}).get("common_domain", True):
            if self._common is None:
                sets = [s for s in U.run_sets(self.cfg) if self.features_for(s)]
                self._common = self.common_domain(sets)
            return self._common
        return self.own_domain(set_name)

    def stack(self, names):
        return self.arrays[[self.names.index(n) for n in names]]


# =========================================================================== orchestrator
def _apply_spectral(cfg, cls, spec):
    """Attach spectral verdicts; optionally move flagged layers to the suggested generic family."""
    if spec is None or spec.empty:
        return cls
    sp = spec.set_index("layer")
    cls["spectral_verdict"] = [sp.loc[n, "verdict"] if n in sp.index else "" for n in cls.index]
    cls["spectral_note"] = [sp.loc[n, "note"] if n in sp.index else "" for n in cls.index]
    if cfg["spectral_check"].get("apply_relabel") and "suggested_family" in sp.columns:
        for n in cls.index:
            if n in sp.index and sp.loc[n, "verdict"] == "relabel_generic" and isinstance(sp.loc[n, "suggested_family"], str):
                old = cls.loc[n, "family"]
                cls.loc[n, "family"] = sp.loc[n, "suggested_family"]
                cls.loc[n, "label"] = f"{cls.loc[n, 'label']} [relabelled: generic {sp.loc[n, 'suggested_family']}]"
                log.info("Spectral relabel: %s %s -> %s", n, old, sp.loc[n, "suggested_family"])
    return cls


def run_feature_engineering(cfg):
    fe_qa, fe_al, fe_au = U.d_fe(cfg, "qa"), U.d_fe(cfg, "aligned"), U.d_fe(cfg, "audit")
    fe_en, fe_se, fe_st = U.d_fe(cfg, "engineered"), U.d_fe(cfg, "selected"), U.d_fe(cfg, "store")
    sets = U.run_sets(cfg)

    # 1.1 QA ------------------------------------------------------------------------------
    meta_check = check_metadata(cfg)
    U.jdump(meta_check, fe_qa / "metadata_check.json")
    layers = discover_layers(cfg)
    table = load_layer_table(cfg)
    cls = classify_layers(cfg, layers, table)
    unmapped = cls[(cls.rule_source == "default") & (cls.excluded_reason == "")]
    if len(unmapped):
        log.warning("%d layers matched no family/direction rule (treated as neutral/context): %s",
                    len(unmapped), ", ".join(unmapped.name[:8]) + (" ..." if len(unmapped) > 8 else ""))

    # 1.0 spectral check (physics, before any image statistics) ------------------------------
    from .spectral import run_spectral_check
    spec = run_spectral_check(cfg, table, cls)
    cls = _apply_spectral(cfg, cls, spec)

    # 1.2 alignment ---------------------------------------------------------------------------
    grid = U.build_master_grid(cfg)
    aoi = U.aoi_mask(cfg, grid)
    aligned, inv = align_layers(cfg, cls, grid, aoi)
    inv.to_csv(fe_qa / "layer_inventory.csv", index=False)
    cls.to_csv(fe_qa / "layer_classification.csv", index=False)
    bad = inv[inv["flags"].fillna("") != ""]
    if len(bad):
        log.warning("QA flags on %d layers (see layer_inventory.csv)", len(bad))
    log.info("Aligned %d layers to the master grid (%d excluded)", len(aligned), int((cls.excluded_reason != "").sum()))

    # 1.3 masks and flags -------------------------------------------------------------------
    masks, mmeta, flags = build_masks_and_flags(cfg, cls, grid, aligned, aoi)
    U.write_raster(fe_al / "aoi_mask.tif", aoi.astype("uint8"), grid, nodata=0, dtype="uint8")
    if masks:
        U.write_raster(fe_al / "masks.tif", np.stack([m.astype("uint8") for m in masks.values()]), grid,
                       descriptions=list(masks), nodata=0, dtype="uint8")
    if flags:
        U.write_raster(fe_al / "flags.tif", np.stack([m.astype("uint8") for m in flags.values()]), grid,
                       descriptions=list(flags), nodata=0, dtype="uint8")
    U.jdump(mmeta, fe_al / "masks_meta.json")
    stack_names = list(aligned)
    U.write_raster(fe_al / "aligned_layers.tif", np.stack([aligned[n] for n in stack_names]), grid, descriptions=stack_names)

    # 1.5 engineered features + lineament provenance ------------------------------------------
    vec, vmeta = vector_features(cfg, grid)
    lan, lmeta = local_anomaly_features(cfg, cls, aligned, grid)
    eng = {**vec, **lan}
    if eng:
        U.write_raster(fe_en / "engineered_stack.tif", np.stack(list(eng.values())), grid, descriptions=list(eng))
    log.info("Engineered %d vector features and %d local-anomaly features", len(vec), len(lan))
    lineament_provenance(cfg, cls, grid, aligned, aoi, fe_qa)

    # 1.4 audit: SEPARATELY per feature set, on the analysis domain ---------------------------
    meta_cols = ["feature_set", "sensor", "family", "direction", "weight", "center", "width", "label", "tir",
                 "combination", "target", "source"]
    for c in ("spectral_verdict", "spectral_note"):
        if c in cls.columns:
            meta_cols.append(c)
    vl = pd.DataFrame(vmeta + lmeta).set_index("name") if (vmeta or lmeta) else pd.DataFrame(columns=meta_cols)
    for c in meta_cols:
        if c not in vl.columns:
            vl[c] = ""
    meta = pd.concat([cls.loc[list(aligned), meta_cols], vl[meta_cols]])
    pool = {**aligned, **eng}
    union = np.zeros(grid.shape, bool)
    for m, mm in zip(masks.values(), mmeta):
        if set(mm["applies_to"]) & set(sets):
            union |= m
    audit_dom = aoi & ~union
    if audit_dom.sum() < 1000:
        log.warning("Few unmasked pixels (%d) for the audit; using the full AOI instead.", int(audit_dom.sum()))
        audit_dom = aoi
    veg = {}
    for sensor, lname in (cfg["audit"].get("vegetation_layers") or {}).items():
        try:
            veg[sensor] = get_layer(cfg, cls, grid, aligned, lname)
        except KeyError as e:
            log.warning("Vegetation check for %s skipped: %s", sensor, e)
    sel, reports = [], []
    for s in sets:
        sub = {n: a for n, a in pool.items() if meta.loc[n, "feature_set"] == s}
        if not sub:
            log.error("Feature set %s: no layers (sensors %s). It will not be run.", s, cfg["feature_sets"][s]["sensors"])
            continue
        d = fe_au / U.set_dir(cfg, s)
        d.mkdir(parents=True, exist_ok=True)
        try:
            ss, rep = audit_select(cfg, sub, meta.loc[list(sub)], audit_dom, d, veg)
        except RuntimeError as e:
            log.error("Feature set %s: %s It will not be run.", s, e)
            continue
        sel += ss
        reports.append(rep)
    if not sel:
        raise RuntimeError("No features selected in any set.")
    report = pd.concat(reports, ignore_index=True)
    report.to_csv(fe_au / "selection_report_all_sets.csv", index=False)

    # 1.6/1.7 transforms, store --------------------------------------------------------------------
    arrays, rows = [], []
    for n in sel:
        a, tr = maybe_transform(pool[n], audit_dom, cfg["transforms"]["auto_log_skew"])
        a = np.where(aoi, a, np.nan).astype(np.float32)
        arrays.append(a)
        m = meta.loc[n].to_dict()
        m.update(name=n, transform=tr, veg_corr=float(report.set_index("layer").loc[n, "veg_corr"]),
                 valid_fraction=float(np.isfinite(a[aoi]).mean()))
        rows.append(m)
    dic = pd.DataFrame(rows).set_index("name")
    stack = np.stack(arrays)
    U.write_raster(fe_st / "ALL_features.tif", stack, grid, descriptions=sel)
    for s in sets:
        ix = [i for i, n in enumerate(sel) if dic.loc[n, "feature_set"] == s]
        if ix:
            U.write_raster(fe_st / f"{U.set_dir(cfg, s)}_features.tif", stack[ix], grid, descriptions=[sel[i] for i in ix])
    full = meta.join(report.set_index("layer")[["status", "reason", "valid_fraction", "veg_corr"]], rsuffix="_audit")
    with pd.ExcelWriter(fe_st / "feature_dictionary.xlsx") as xw:
        dic.to_excel(xw, sheet_name="features")
        full.to_excel(xw, sheet_name="all_candidates")
        pd.DataFrame(mmeta).to_excel(xw, sheet_name="masks", index=False)
        inv.to_excel(xw, sheet_name="qa_inventory", index=False)
        if spec is not None and len(spec):
            spec.to_excel(xw, sheet_name="spectral_check", index=False)
    dic.to_csv(fe_se / "selected_features.csv")
    if cfg["store"].get("write_tables", False):
        ok = aoi & np.all(np.isfinite(stack), axis=0)
        r, c = np.nonzero(ok)
        x, y = grid.xy(r, c)
        for s in sets:
            ix = [i for i, n in enumerate(sel) if dic.loc[n, "feature_set"] == s]
            if not ix:
                continue
            tab = pd.DataFrame(stack[ix][:, r, c].T, columns=[sel[i] for i in ix])
            tab.insert(0, "northing", y)
            tab.insert(0, "easting", x)
            try:
                tab.to_parquet(fe_st / f"{U.set_dir(cfg, s)}_table.parquet", index=False)
            except Exception:
                tab.to_csv(fe_st / f"{U.set_dir(cfg, s)}_table.csv.gz", index=False)
    counts = {s: int((dic.feature_set == s).sum()) for s in sets}
    U.jdump(dict(features_per_set=counts, masks=mmeta), fe_st / "feature_sets_summary.json")
    log.info("Feature store written: %d features (%s)", len(sel), ", ".join(f"{k}={v}" for k, v in counts.items()))
    return dict(n_selected=len(sel), per_set=counts, grid=(grid.height, grid.width, grid.res))
