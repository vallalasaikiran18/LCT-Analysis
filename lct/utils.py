"""Shared helpers: configuration, paths, grid handling, raster I/O.

Nothing project-specific lives here. Every name, threshold and folder comes from config.yaml.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import yaml
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.transform import Affine, array_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject

log = logging.getLogger("lct")


# --------------------------------------------------------------------------- config / paths
def setup_logging(log_file=None, level="INFO"):
    handlers = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))
    logging.basicConfig(level=getattr(logging, str(level).upper(), logging.INFO),
                        format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%H:%M:%S", handlers=handlers, force=True)


def load_config(path) -> dict:
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    base = path.parent
    cfg["_path"], cfg["_dir"] = str(path), str(base)
    for key, tag in (("input_root", "_in"), ("output_root", "_out")):
        p = Path(cfg["paths"][key])
        cfg[tag] = str(p if p.is_absolute() else (base / p).resolve())
    return cfg


def rpath(cfg, p):
    """Resolve a config path: absolute -> as is; else under input_root; else next to config.yaml."""
    if p is None:
        return None
    p = Path(p)
    if p.is_absolute():
        return p
    for base in (cfg["_in"], cfg["_dir"]):
        c = Path(base) / p
        if c.exists():
            return c
    return Path(cfg["_dir"]) / p


def d_stage(cfg, key, *sub, create=True) -> Path:
    p = Path(cfg["_out"]) / cfg["layout"][key]
    for s in sub:
        p = p / s
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def d_fe(cfg, key) -> Path:
    return d_stage(cfg, "features", cfg["layout"]["fe_sub"][key])


def run_sets(cfg):
    """Feature sets to run, in order (e.g. ASTER, S2, DEM)."""
    sets = list(cfg.get("run", {}).get("sets") or cfg["feature_sets"])
    bad = [s for s in sets if s not in cfg["feature_sets"]]
    if bad:
        raise KeyError(f"run.sets names unknown feature_sets: {bad}")
    return sets


def set_dir(cfg, set_name):
    return cfg["feature_sets"][set_name].get("dir", set_name)


def d_leaf(cfg, model, set_name, create=True) -> Path:
    return d_stage(cfg, "models", cfg["layout"]["model_dirs"][model], set_dir(cfg, set_name), create=create)


def pct_rank(x):
    """Rank percentile in (0, 1) of a 1-D array (ties averaged)."""
    from scipy.stats import rankdata
    x = np.asarray(x)
    return ((rankdata(x) - 0.5) / max(len(x), 1)).astype(np.float32)


def fast_pct(x):
    """Rank percentile without tie handling (fast; for Monte Carlo loops on continuous scores)."""
    x = np.asarray(x)
    r = np.empty(len(x), np.float64)
    r[np.argsort(x, kind="stable")] = np.arange(len(x))
    return (r + 0.5) / max(len(x), 1)


def fingerprint(obj) -> str:
    """Stable short hash of a JSON-serialisable object (used to freeze weights)."""
    import hashlib
    txt = json.dumps(obj, sort_keys=True, default=_default)
    return hashlib.sha1(txt.encode("utf-8")).hexdigest()[:16]


def file_hash(path) -> str:
    import hashlib
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _default(o):
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def jdump(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=_default), encoding="utf-8")


def utm_label(crs: CRS):
    epsg = crs.to_epsg()
    if epsg and 32601 <= epsg <= 32660:
        return f"{epsg - 32600}N"
    if epsg and 32701 <= epsg <= 32760:
        return f"{epsg - 32700}S"
    return None


# --------------------------------------------------------------------------- grid
@dataclass
class Grid:
    crs: CRS
    transform: Affine
    width: int
    height: int

    @property
    def shape(self):
        return (self.height, self.width)

    @property
    def res(self):
        return abs(self.transform.a)

    def xy(self, rows, cols):
        rows, cols = np.asarray(rows, float), np.asarray(cols, float)
        x = self.transform.a * (cols + 0.5) + self.transform.c
        y = self.transform.e * (rows + 0.5) + self.transform.f
        return x, y

    def rc(self, x, y):
        col, row = ~self.transform * (np.asarray(x, float), np.asarray(y, float))
        return np.floor(row).astype(int), np.floor(col).astype(int)

    @classmethod
    def from_raster(cls, path):
        with rasterio.open(path) as s:
            return cls(s.crs, s.transform, s.width, s.height)


def load_aoi(cfg, crs):
    p = rpath(cfg, cfg["inputs"].get("aoi"))
    if p is None or not Path(p).exists():
        return None
    import geopandas as gpd
    g = gpd.read_file(p)
    if g.crs is None:
        g = g.set_crs(cfg["inputs"].get("assume_vector_crs", "EPSG:4326"))
    return g.to_crs(crs)


def build_master_grid(cfg) -> Grid:
    g = cfg["grid"]
    ref = rpath(cfg, g["reference"]) if g.get("reference") else None
    if ref is None:
        for c in g.get("reference_candidates", []):
            p = rpath(cfg, c)
            if p.exists():
                ref = p
                break
    if ref is None or not Path(ref).exists():
        raise FileNotFoundError("No grid reference raster found. Set grid.reference or grid.reference_candidates.")
    with rasterio.open(ref) as src:
        crs, transform, w, h, bounds = src.crs, src.transform, src.width, src.height, src.bounds
    tcrs = CRS.from_user_input(g["target_crs"]) if g.get("target_crs") else crs
    if tcrs is None or tcrs.is_geographic:
        raise ValueError("Master grid must be a projected (metric) CRS. Set grid.target_crs (e.g. EPSG:32736).")
    if tcrs != crs or g.get("resolution"):
        transform, w, h = calculate_default_transform(crs, tcrs, w, h, *bounds, resolution=g.get("resolution"))
    grid = Grid(tcrs, transform, int(w), int(h))
    aoi = load_aoi(cfg, grid.crs)
    if aoi is not None and g.get("crop_to_aoi", True):
        minx, miny, maxx, maxy = aoi.total_bounds
        col0 = max(int(np.floor((minx - transform.c) / transform.a)), 0)
        col1 = min(int(np.ceil((maxx - transform.c) / transform.a)), grid.width)
        row0 = max(int(np.floor((maxy - transform.f) / transform.e)), 0)
        row1 = min(int(np.ceil((miny - transform.f) / transform.e)), grid.height)
        if col1 > col0 and row1 > row0:
            grid = Grid(tcrs, transform * Affine.translation(col0, row0), col1 - col0, row1 - row0)
    log.info("Master grid: %s | %d x %d px | %.1f m | reference=%s", grid.crs.to_string(), grid.width,
             grid.height, grid.res, Path(ref).name)
    return grid


def aoi_mask(cfg, grid) -> np.ndarray:
    aoi = load_aoi(cfg, grid.crs)
    if aoi is None:
        return np.ones(grid.shape, bool)
    return rasterize([(geom, 1) for geom in aoi.geometry], out_shape=grid.shape, transform=grid.transform,
                     fill=0, dtype="uint8").astype(bool)


def load_points_file(cfg, key, grid):
    """Optional point layer (known occurrences). Shapefile/GPKG, or CSV/XLSX with lon/lat columns."""
    p = rpath(cfg, cfg["inputs"].get(key))
    if p is None or not Path(p).exists():
        return None
    import geopandas as gpd
    import pandas as pd
    if Path(p).suffix.lower() in (".csv", ".txt", ".xlsx", ".xls"):
        df = pd.read_excel(p) if Path(p).suffix.lower() in (".xlsx", ".xls") else pd.read_csv(p)
        oc = cfg["inputs"].get("occurrence_columns", {"lon": "lon", "lat": "lat", "crs": "EPSG:4326"})
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[oc["lon"]], df[oc["lat"]]), crs=oc["crs"])
    else:
        gdf = gpd.read_file(p)
        if gdf.crs is None:
            gdf = gdf.set_crs(cfg["inputs"].get("assume_vector_crs", "EPSG:4326"))
    return gdf.to_crs(grid.crs)


# --------------------------------------------------------------------------- raster I/O
def _native_res_m(src):
    r = abs(src.transform.a)
    return r * 111320.0 if (src.crs is not None and src.crs.is_geographic) else r


def read_to_grid(path, grid: Grid, categorical=False, band=1) -> np.ndarray:
    """Read one band onto the master grid. Area-average when downsampling, bilinear when upsampling."""
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"{path} has no CRS")
        nodata = src.nodata
        if nodata is None and np.issubdtype(np.dtype(src.dtypes[band - 1]), np.floating):
            nodata = np.nan
        if categorical:
            rs = Resampling.nearest
        else:
            rs = Resampling.average if _native_res_m(src) < grid.res * 0.99 else Resampling.bilinear
        dst = np.full(grid.shape, np.nan, dtype=np.float32)
        reproject(source=rasterio.band(src, band), destination=dst, src_transform=src.transform,
                  src_crs=src.crs, src_nodata=nodata, dst_transform=grid.transform, dst_crs=grid.crs,
                  dst_nodata=np.nan, resampling=rs, init_dest_nodata=True)
    dst[~np.isfinite(dst)] = np.nan
    dst[dst < -1e30] = np.nan          # un-flagged ArcGIS float nodata (-3.4e38)
    return dst


def write_raster(path, arr, grid: Grid, descriptions=None, nodata=np.nan, dtype="float32"):
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[None]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2], count=arr.shape[0],
                       dtype=dtype, crs=grid.crs, transform=grid.transform, nodata=nodata,
                       compress="deflate") as dst:
        dst.write(arr.astype(dtype))
        if descriptions:
            dst.descriptions = tuple(str(d) for d in descriptions)


def read_stack(path):
    with rasterio.open(path) as s:
        arr = s.read().astype(np.float32)
        names = [d if d else f"band_{i + 1}" for i, d in enumerate(s.descriptions)]
        grid = Grid(s.crs, s.transform, s.width, s.height)
    return arr, names, grid


def read_band_by_name(path, name):
    with rasterio.open(path) as s:
        for i, d in enumerate(s.descriptions, start=1):
            if d == name:
                return s.read(i).astype(np.float32)
    return None


def apply_op(a, op, v):
    return {">": a > v, ">=": a >= v, "<": a < v, "<=": a <= v, "==": a == v, "!=": a != v}[op]
