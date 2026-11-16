"""Physical check of the band-ratio layers against library spectra (no image data used).

For every ratio/index in layer_table.csv:
  1. the formula is compiled into a function of band reflectances,
  2. each library spectrum is resampled to the sensor's bands (Gaussian centre/FWHM, or a supplied SRF file),
  3. the index is evaluated for every mineral sample,
  4. a verdict says whether the labelled target is separable from every other mineral, only from other
     classes (-> relabel as a generic proxy), or not at all; plus the areal fraction of target needed in a
     pixel of host rock before the index moves noticeably (pegmatite dykes are sub-pixel).

TIR (ASTER bands 10-14) cannot be checked with a 0.35-2.5 um reflectance library; such layers are reported
as such rather than guessed. Image-normalised indices (norm(...)) are reported as not evaluable.
"""
from __future__ import annotations

import fnmatch
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import utils as U

log = U.log

# Band centre and FWHM in nm. ASTER: VNIR/SWIR band-pass limits from the ASTER instrument description.
# Sentinel-2A: central wavelength and bandwidth from the ESA Sentinel-2 MSI technical guide.
SRF_GAUSS = {
    "ASTER": {"1": (556, 80), "2": (661, 60), "3": (807, 80), "4": (1656, 100), "5": (2167, 40),
              "6": (2209, 40), "7": (2262, 50), "8": (2336, 70), "9": (2400, 70)},
    "S2": {"1": (443, 21), "2": (492, 66), "3": (560, 36), "4": (665, 31), "5": (704, 15), "6": (740, 15),
           "7": (783, 20), "8": (833, 106), "8A": (865, 21), "9": (945, 20), "10": (1374, 31),
           "11": (1614, 91), "12": (2202, 175)},
}
TIR_BANDS = {"ASTER": {"10", "11", "12", "13", "14"}}
SENSOR_BANDS = {"ASTER": set(SRF_GAUSS["ASTER"]) | TIR_BANDS["ASTER"], "S2": set(SRF_GAUSS["S2"])}


# =========================================================================== library
def _read_numbers(path):
    rows = []
    for line in Path(path).read_text(errors="ignore").splitlines():
        parts = re.split(r"[,\s;]+", line.strip())
        try:
            vals = [float(p) for p in parts if p != ""]
        except ValueError:
            continue                                   # header / text line
        if vals:
            rows.append(vals)
    return rows


def _to_nm(wl):
    wl = np.asarray(wl, float)
    return wl * 1000.0 if np.nanmax(wl) < 100 else wl


def load_library(cfg):
    """{mineral: [(sample_name, wavelength_nm, reflectance), ...]} and a list of missing minerals."""
    sc = cfg["spectral_check"]
    d = U.rpath(cfg, sc["library_dir"])
    if d is None or not Path(d).exists():
        return {}, list(sc["minerals"]), f"library folder not found: {d}"
    files = sorted(p for p in Path(d).rglob("*") if p.is_file())
    wl_common = None
    if sc.get("wavelength_file"):
        wf = [p for p in files if p.name == sc["wavelength_file"]]
        if wf:
            wl_common = _to_nm([r[0] for r in _read_numbers(wf[0])])
    lib, missing = {}, []
    for mineral, spec in sc["minerals"].items():
        hits = [p for p in files if fnmatch.fnmatch(p.name.lower(), spec["pattern"].lower())
                and p.name != sc.get("wavelength_file")]
        samples = []
        for p in hits:
            rows = _read_numbers(p)
            if not rows:
                continue
            if len(rows[0]) >= 2:                                   # two-column file
                arr = np.array([r[:2] for r in rows if len(r) >= 2])
                wl, rf = _to_nm(arr[:, 0]), arr[:, 1]
            elif wl_common is not None and len(rows) == len(wl_common):
                wl, rf = wl_common, np.array([r[0] for r in rows])
            else:
                log.warning("Spectral check: cannot pair %s with a wavelength axis; skipped", p.name)
                continue
            rf = np.where((rf < -1) | (rf > 5), np.nan, rf)       # USGS deleted-channel marker -1.23e34
            if np.isfinite(rf).sum() > 50:
                samples.append((p.stem, wl, rf))
        if samples:
            lib[mineral] = samples
        else:
            missing.append(mineral)
    return lib, missing, ""


def load_srf(cfg):
    """Optional measured SRFs: CSV with wavelength_nm and columns ASTER_<band> / S2_<band> (e.g. S2_8A)."""
    p = cfg["spectral_check"].get("srf_file")
    if not p:
        return None
    p = U.rpath(cfg, p)
    if not Path(p).exists():
        log.warning("SRF file not found (%s): using Gaussian band models", p)
        return None
    df = pd.read_csv(p)
    wl = _to_nm(df.iloc[:, 0].values)
    out = {}
    for c in df.columns[1:]:
        sensor, _, band = c.partition("_")
        out.setdefault(sensor, {})[band.upper().lstrip("B")] = (wl, df[c].values.astype(float))
    return out


def resample(wl, rf, sensor, srf=None):
    """Band-averaged reflectance for every reflective band of the sensor."""
    out = {}
    ok = np.isfinite(rf)
    for band, (c, fwhm) in SRF_GAUSS[sensor].items():
        if srf and sensor in srf and band in srf[sensor]:
            swl, sr = srf[sensor][band]
            w = np.interp(wl, swl, sr, left=0, right=0)
        else:
            s = fwhm / 2.3548
            w = np.exp(-0.5 * ((wl - c) / s) ** 2)
        w = np.where(ok, w, 0.0)
        out[band] = float(np.sum(w * np.nan_to_num(rf)) / w.sum()) if w.sum() > 1e-9 else np.nan
    return out


# =========================================================================== formula compiler
_TOKEN = re.compile(r"(?<![\w.])B?(\d{1,2}A?)(?![\w.])", re.I)
_SAFE = re.compile(r"^[\d\s.+\-*/()]*$")


def compile_index(combo, sensor, bands_override=None):
    """Return (function(band_dict) -> value, bands_used, note). function is None when not evaluable."""
    e = str(combo).split(":")[-1].strip()
    if not e:
        return None, [], "no formula"
    if "norm(" in e.lower():
        return None, [], "image-normalised index (norm): needs scene statistics, not evaluable on library spectra"
    e = re.sub(r"(?<=[\d)\s])[xX](?=[\s\d(])", "*", e)
    e = (e.replace("×", "*").replace("−", "-").replace("–", "-").replace("²", "**SQ")
          .replace("[", "(").replace("]", ")").replace("lg", "LOG"))
    valid = SENSOR_BANDS[sensor]
    allow = {str(b) for b in bands_override} if bands_override else None
    used = []

    def sub(m):
        tok = m.group(1).upper()
        if tok in valid and (allow is None or tok in allow):
            used.append(tok)
            return f"V{len(used) - 1}"
        return m.group(0)                                          # a constant (e.g. the 2 in '2 x 13')
    expr = _TOKEN.sub(sub, e)
    expr = expr.replace("SQ", "2")
    check = re.sub(r"V\d+", "1", expr).replace("LOG", "")
    if not _SAFE.match(check):
        return None, used, f"could not parse formula '{combo}'"
    names = list(used)
    tir = TIR_BANDS.get(sensor, set())
    if any(b in tir for b in names):
        return None, sorted(set(names)), "uses TIR bands: needs an emissivity library (e.g. ECOSTRESS), not checked"
    code = compile(expr.replace("LOG", "log10"), "<index>", "eval")

    def f(b):
        env = {f"V{i}": b.get(n, np.nan) for i, n in enumerate(names)}
        env["log10"] = np.log10
        with np.errstate(all="ignore"):
            try:
                return float(eval(code, {"__builtins__": {}}, env))
            except (ZeroDivisionError, ValueError, OverflowError):
                return np.nan
    return f, sorted(set(names), key=lambda t: (len(t), t)), ""


# =========================================================================== verdict
def _targets(cfg, target_text):
    out = []
    for rx, minerals in (cfg["spectral_check"].get("target_map") or {}).items():
        if re.search(rx, str(target_text), re.I):
            out += [m for m in minerals if m not in out]
    return out


def evaluate_layer(cfg, name, sensor, combo, target, direction, lib, srf, override=None):
    sc = cfg["spectral_check"]
    cls_of = {m: s.get("class", m) for m, s in sc["minerals"].items()}
    row = dict(layer=name, sensor=sensor, combination=combo, target=target, direction=direction)
    f, bands, note = compile_index(combo, sensor, override)
    row["bands"] = ",".join(bands)
    if f is None:
        row.update(verdict="not_checked", note=note)
        return row, None
    vals = {m: np.array([f(resample(wl, rf, sensor, srf)) for _, wl, rf in samples]) for m, samples in lib.items()}
    vals = {m: v[np.isfinite(v)] for m, v in vals.items()}
    vals = {m: v for m, v in vals.items() if v.size}
    if len(vals) < 2:
        row.update(verdict="not_checked", note="fewer than 2 library minerals evaluable")
        return row, vals
    sign = -1.0 if direction == "low" else 1.0
    o = {m: sign * v for m, v in vals.items()}                  # oriented: higher = favourable
    allv = np.concatenate(list(o.values()))
    rng = float(allv.max() - allv.min())
    med = {m: float(np.median(v)) for m, v in o.items()}
    ranking = sorted(med, key=lambda m: -med[m])
    row["favours"] = ", ".join(ranking[:3])
    row.update({f"value_{m}": float(np.median(v)) for m, v in vals.items()})
    tg = [t for t in _targets(cfg, target) if t in o]
    row["target_minerals"] = ", ".join(tg)
    if rng <= 1e-12:
        row.update(verdict="non_diagnostic", note="index is flat across the library at this sensor's bands")
        return row, vals
    if direction in ("neutral", "band"):
        row.update(verdict="not_checked", note=f"direction '{direction}': no favourable tail to test")
        return row, vals
    if not tg:
        row.update(verdict="target_not_in_library",
                   note=f"target '{target}' has no library mineral (add it to spectral_check.minerals/target_map)")
        return row, vals
    margin = sc["margin_frac"] * rng
    t_worst = min(float(o[t].min()) for t in tg)                 # least favourable target sample
    confusers = [m for m in o if m not in tg and float(o[m].max()) >= t_worst - margin]
    row["margin"] = margin / rng
    row["confused_with"] = ", ".join(confusers)
    t_cls = {cls_of.get(t) for t in tg}
    c_cls = {cls_of.get(c) for c in confusers}
    all_med = float(np.median(allv))
    if all(med[t] < all_med for t in tg):
        verdict = "direction_conflict"
        note = "the target scores LOW on this index: the declared direction favours other minerals"
    elif not confusers:
        verdict, note = "discriminates", "target beats every other library mineral by the margin"
    elif len(t_cls) == 1 and c_cls <= t_cls:
        cl = next(iter(t_cls))
        verdict = "relabel_generic"
        note = f"separates the {cl} class but not {', '.join(tg)} within it"
        row["suggested_family"] = sc.get("class_family", {}).get(cl, cl)
    else:
        verdict, note = "non_diagnostic", "favours minerals of other classes as much as the target"
    row.update(verdict=verdict, note=note)

    bg = [m for m in sc.get("mixing_background", []) if m in lib]   # sub-pixel test
    if bg and verdict in ("discriminates", "relabel_generic"):
        wl0 = lib[tg[0]][0][1]

        def med_spec(m):
            return np.nanmedian(np.vstack([np.interp(wl0, wl, rf) for _, wl, rf in lib[m]]), axis=0)
        T, B = np.nanmedian(np.vstack([med_spec(t) for t in tg]), axis=0), np.nanmean(np.vstack([med_spec(m) for m in bg]), axis=0)
        vb = sign * f(resample(wl0, B, sensor, srf))
        fd = np.nan
        for frac in np.linspace(0.05, 1.0, 20):
            if sign * f(resample(wl0, frac * T + (1 - frac) * B, sensor, srf)) - vb >= margin:
                fd = float(frac)
                break
        row["min_areal_fraction_detectable"] = fd
    return row, vals


# =========================================================================== orchestrator
def run_spectral_check(cfg, table: pd.DataFrame, cls: pd.DataFrame | None = None):
    """Writes spectral_check.csv; returns it (empty DataFrame when the library is unavailable)."""
    sc = cfg.get("spectral_check") or {}
    if not sc.get("enabled", False):
        return pd.DataFrame()
    out = U.d_fe(cfg, "spectral")
    lib, missing, err = load_library(cfg)
    status = dict(library_found=not err, error=err, minerals_found={m: len(v) for m, v in lib.items()},
                  minerals_missing=missing, srf="file" if sc.get("srf_file") else "gaussian centre/FWHM")
    U.jdump(status, out / "spectral_library_status.json")
    if err or len(lib) < 2:
        log.warning("Spectral check skipped: %s", err or f"only {len(lib)} library minerals found ({missing} missing)")
        return pd.DataFrame()
    if missing:
        log.warning("Spectral check: no library spectra for %s (layers targeting them are reported as such)", missing)
    srf = load_srf(cfg)
    overrides = cfg["layers"].get("band_overrides", {}) or {}
    rows, curves = [], {}
    for name, r in table.iterrows():
        sensor = "ASTER" if name.upper().startswith("AST") else ("S2" if name.upper().startswith("S2") else None)
        if cls is not None and name in cls.index:
            sensor = cls.loc[name, "sensor"] if cls.loc[name, "sensor"] in SRF_GAUSS else sensor
            direction = cls.loc[name, "direction"]
        else:
            direction = "high"
        if sensor not in SRF_GAUSS:
            continue
        row, vals = evaluate_layer(cfg, name, sensor, r["combination"], r["target"], direction, lib, srf,
                                   overrides.get(name))
        rows.append(row)
        if vals:
            curves[name] = vals
    df = pd.DataFrame(rows)
    df.to_csv(out / "spectral_check.csv", index=False)
    rs = []                                                            # resampled library (for inspection)
    for m, samples in lib.items():
        for sname, wl, rf in samples:
            for sensor in SRF_GAUSS:
                rs.append(dict(mineral=m, sample=sname, sensor=sensor, **{f"B{k}": v for k, v in resample(wl, rf, sensor, srf).items()}))
    pd.DataFrame(rs).to_csv(out / "library_resampled_to_bands.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    for m, samples in lib.items():
        _, wl, rf = samples[0]
        ax.plot(wl, rf, label=m, lw=0.9)
    for sensor, ls in (("ASTER", "-"), ("S2", ":")):
        for b, (c, w) in SRF_GAUSS[sensor].items():
            if c > 1500:
                ax.axvspan(c - w / 2, c + w / 2, alpha=0.08 if sensor == "ASTER" else 0.05,
                           color="k" if sensor == "ASTER" else "b")
    ax.set(xlim=(1400, 2500), xlabel="wavelength (nm)", ylabel="reflectance",
           title="Library spectra in the SWIR (grey = ASTER bands 4-9, blue = S2 B11/B12)")
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(out / "library_swir.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    if len(df):
        summ = df.groupby(["sensor", "verdict"]).size().unstack(fill_value=0)
        summ.to_csv(out / "spectral_check_summary.csv")
        log.info("Spectral check (%d layers):\n%s", len(df), summ.to_string())
    return df
