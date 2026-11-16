"""Create a small synthetic dataset with the same folder/layer names as the real project (smoke test only)."""
import sys
from pathlib import Path
import numpy as np, rasterio
from rasterio.transform import from_origin
from scipy import ndimage as ndi
import geopandas as gpd
from shapely.geometry import LineString

root = Path(sys.argv[1] if len(sys.argv) > 1 else "synthetic_input")
n = int(sys.argv[2]) if len(sys.argv) > 2 else 160
rng = np.random.default_rng(1)
tr = from_origin(417000, 8147000, 30, 30)
def field(s=6): 
    a = ndi.gaussian_filter(rng.normal(size=(n, n)), s); return (a - a.mean()) / a.std()
def save(p, a, dtype="float32", nodata=-9999.0):
    p.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(p, "w", driver="GTiff", height=n, width=n, count=1, dtype=dtype, crs="EPSG:32736", transform=tr, nodata=nodata) as d:
        d.write(np.where(np.isfinite(a), a, nodata).astype(dtype), 1)
base = [field() for _ in range(5)]          # latent factors: AlOH, felsic, iron, mafic, vegetation
dem = 1200 + 40 * field(10) + 10 * field(2)
D = root / "DEM Features"
save(D/"DEM_utm.tif", dem); save(D/"dem_clip.tif", dem); save(D/"fill.tif", dem); save(D/"dem_smooth.tif", ndi.gaussian_filter(dem, 1))
for k in (3, 11, 31, 500): save(D/f"mean_{k}.tif", ndi.uniform_filter(dem, min(k, 41)))
gy, gx = np.gradient(dem, 30); slope = np.degrees(np.arctan(np.hypot(gx, gy)))
save(D/"slope.tif", slope); save(D/"rugged.tif", ndi.generic_filter(dem, np.std, size=3) if n < 200 else slope * 2)
for k in (3, 11, 31): save(D/f"TPI_{k}.tif", dem - ndi.uniform_filter(dem, k))
save(D/"residual_500.tif", dem - ndi.uniform_filter(dem, 41)); save(D/"relief_250.tif", ndi.maximum_filter(dem, 9) - ndi.minimum_filter(dem, 9))
save(D/"relief_500.tif", ndi.maximum_filter(dem, 17) - ndi.minimum_filter(dem, 17)); save(D/"Curveture.tif", ndi.laplace(dem))
save(D/"flowacc.tif", np.exp(2 * field(3)) * 10); save(D/"streamdensity.tif", np.abs(field(4)))
save(D/"flowdir.tif", rng.integers(1, 255, (n, n)).astype("float32")); save(D/"HS_315_45.tif", dem)
lines = [LineString([(417000 + rng.uniform(0, n*30), 8147000 - rng.uniform(0, n*30)),
                     (417000 + rng.uniform(0, n*30), 8147000 - rng.uniform(0, n*30))]) for _ in range(25)]
(D/"Lineaments").mkdir(parents=True, exist_ok=True); (D/"Stream to feature").mkdir(parents=True, exist_ok=True)
gpd.GeoDataFrame(geometry=lines, crs="EPSG:32736").to_file(D/"Lineaments"/"Lineaments.shp")
gpd.GeoDataFrame(geometry=lines[:6], crs="EPSG:32736").to_file(D/"Stream to feature"/"stream to feature.shp")
A = root/"Remote Sensing"/"ASTER"
def mix(w): return sum(wi * b for wi, b in zip(w, base)) + 0.5 * field(1.5)
save(A/"TIR_SUPPORT_MASK.tif", np.ones((n, n)), "uint8", 0)
for i in range(1, 24):
    f = mix(rng.normal(size=5)); save(A/f"AST-BR-{i:02d}.tif", 1 + 0.1 * f)
for i, j in ((23, 4), (18, 6), (21, 8)):    # exact duplicates, as in the real data
    save(A/f"AST-BR-{i:02d}.tif", rasterio.open(A/f"AST-BR-{j:02d}.tif").read(1))
save(A/"AST-BR-14.tif", 1 / rasterio.open(A/"AST-BR-07.tif").read(1))   # reciprocal
for i in range(1, 18):
    f = mix(rng.normal(size=5)); 
    if i in (4, 7, 8, 9, 10, 11, 12, 17): f = np.kron(ndi.zoom(f[::3, ::3], 1)[:n//3 + 1, :n//3 + 1], np.ones((3, 3)))[:n, :n]  # blocky TIR
    save(A/f"AST-IDX-{i:02d}.tif", 1 + 0.1 * f)
save(A/"AST-NDVI.tif", 0.2 + 0.1 * base[4]); save(A/"AST-MNDWI.tif", -0.6 + 0.05 * base[4])
S = root/"Remote Sensing"/"Sentinel-2"
for i in range(1, 9): save(S/f"S2-BR-{i:02d}.tif", 1 + 0.2 * mix(rng.normal(size=5)))
for i in range(1, 9): save(S/f"S2-EXT-{i:02d}.tif", 0.02 + 0.01 * np.abs(mix(rng.normal(size=5))))
save(S/"S2-IDX-01.tif", 1.8 + 0.1 * mix(rng.normal(size=5)))
for k, (a, b) in {"01": ("S2A-BR-01", "S2-BR-03"), "02": ("S2A-BR-02", "S2-BR-07"), "04": ("S2A-BR-04", "S2-BR-01")}.items():
    save(S/f"{a}.tif", rasterio.open(S/f"{b}.tif").read(1))
save(S/"S2A-BR-03.tif", 1 + 0.2 * mix(rng.normal(size=5)))
save(S/"S2-NDVI.tif", 0.28 + 0.12 * base[4]); save(S/"S2-MNDWI.tif", -0.6 + 0.05 * base[4] + 0.03 * (field(2) > 2.2))
# RGB (3-band) file to prove it is ignored
p = A/"AST-RGB-01.tif"
with rasterio.open(p, "w", driver="GTiff", height=n, width=n, count=3, dtype="uint8", crs="EPSG:32736", transform=tr) as d: d.write(np.zeros((3, n, n), "uint8"))
M = root/"Raw Data Information"; M.mkdir(parents=True, exist_ok=True)
(M/"ASTER Metadata.txt").write_text("Acquisition time  : 2005-09-21T08:09:57.970000Z\n")
(M/"Sentinel-2 Metadata.txt").write_text("SENSING_TIME        : 2026-08-05T08:14:59.844199Z\n")
print("synthetic data in", root)

# ---------------------------------------------------------------- synthetic occurrences (smoke test only)
import pandas as pd
from pyproj import Transformer
p = np.exp(1.5 * base[0] + 0.8 * base[1]); p[:5] = p[-5:] = 0; p[:, :5] = p[:, -5:] = 0
idx = rng.choice(n * n, 70, replace=False, p=(p / p.sum()).ravel())
rr, cc = np.unravel_index(idx, (n, n))
x, y = 417000 + (cc + 0.5) * 30, 8147000 - (rr + 0.5) * 30
lon, lat = Transformer.from_crs("EPSG:32736", "EPSG:4326", always_xy=True).transform(x, y)
(root / "Occurrences").mkdir(exist_ok=True)
pd.DataFrame(dict(name=[f"occ{i}" for i in range(len(x))], lon=lon, lat=lat)).to_csv(root / "Occurrences" / "occurrences.csv", index=False)

# ---------------------------------------------------------------- SYNTHETIC spectral library (NOT real USGS spectra)
L = root / "Spectral Library" / "splib07a"; L.mkdir(parents=True, exist_ok=True)
wl = np.round(np.linspace(0.35, 2.5, 2151), 4)
(L / "splib07a_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt").write_text("Wavelengths\n" + "\n".join(map(str, wl)))
def spec(level, feats):
    r = level * np.ones_like(wl)
    for c, d, w in feats: r *= 1 - d * np.exp(-0.5 * ((wl - c) / w) ** 2)
    return r
lib = {"Muscovite": spec(0.55, [(1.41, .2, .01), (2.20, .35, .012), (2.35, .15, .015)]),
       "Lepidolite": spec(0.60, [(1.41, .15, .01), (2.195, .30, .012), (2.35, .12, .015)]),
       "Kaolinite": spec(0.70, [(1.40, .2, .01), (2.165, .25, .01), (2.205, .30, .01)]),
       "Chlorite": spec(0.35, [(0.95, .2, .1), (2.25, .3, .015), (2.33, .35, .02)]),
       "Quartz": spec(0.80, [(1.9, .03, .02)]), "Albite": spec(0.72, [(1.9, .05, .02)]),
       "Spodumene": spec(0.70, [(1.0, .08, .1), (1.9, .06, .02), (2.2, .03, .02)])}
for k, v in lib.items():
    for j in range(2):
        vv = v * (1 + 0.02 * (j - 0.5)) + rng.normal(0, 0.002, v.size)
        (L / f"splib07a_{k}_SYN{j}_ASDFRa_AREF.txt").write_text(f"{k} synthetic sample {j}\n" + "\n".join(f"{t:.5f}" for t in vv))
print("synthetic occurrences and SYNTHETIC spectral library written")
