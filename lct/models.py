"""Stage 2 - the fuzzy gamma model (M1), run independently on each feature set (ASTER, S2, DEM).

M1 is knowledge-driven: per-layer fuzzy memberships (linear ramps on rank percentile, or a Gaussian band),
aggregated inside each evidence family by layer weight, then combined across families with AHP weights
through the fuzzy gamma operator. Monte-Carlo jitter of the memberships, family weights and gamma gives the
uncertainty. Every set is scored on the same pixels (common domain), and results are reported as rank
percentiles so that sets with different numbers of layers are compared on the same scale.

M2 (Isolation Forest) and M3 (TOPSIS + CRITIC) are kept for reference; they are not run by default.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import IsolationForest

from . import utils as U

log = U.log
EPS = 1e-6


# =========================================================================== data preparation
@dataclass
class Prepared:
    set_name: str
    features: list
    meta: pd.DataFrame
    flat_idx: np.ndarray
    X: np.ndarray            # (N, F) feature values
    Q: np.ndarray            # (N, F) rank percentile of raw values in (0, 1)
    domain: np.ndarray       # 2-D bool, pixels scored (identical for every set when common_domain is on)


def rank_pct(X):
    N, F = X.shape
    Q = np.empty((N, F), np.float32)
    for j in range(F):
        Q[:, j] = (rankdata(X[:, j]) - 0.5) / N
    return Q


def prepare(cfg, store, set_name) -> Prepared:
    feats = store.features_for(set_name)
    if not feats:
        raise RuntimeError(f"No features for set '{set_name}'")
    dom = store.domain(set_name)
    arr = store.stack(feats)
    ok = dom & np.all(np.isfinite(arr), axis=0)
    flat = np.flatnonzero(ok.ravel())
    if flat.size < 100:
        raise RuntimeError(f"Set '{set_name}': only {flat.size} scorable pixels (masks too aggressive?)")
    X = arr.reshape(len(feats), -1)[:, flat].T.astype(np.float32)
    meta = store.dic.loc[feats].copy()
    mc = cfg["models"]["fuzzy_gamma"]["band"]
    meta["center"] = pd.to_numeric(meta["center"], errors="coerce").fillna(mc["center"])
    meta["width"] = pd.to_numeric(meta["width"], errors="coerce").fillna(mc["width"])
    meta["weight"] = pd.to_numeric(meta["weight"], errors="coerce").fillna(1.0)
    log.info("[%s] %d features | %d scorable pixels (%.1f%% of AOI)", set_name, len(feats), flat.size,
             100 * flat.size / max(int(store.aoi.sum()), 1))
    return Prepared(set_name, feats, meta, flat, X, rank_pct(X), ok)


@dataclass
class ModelResult:
    model: str
    set_name: str
    prep: Prepared
    score_mean: np.ndarray           # raw model score (not comparable across sets)
    score_sd: np.ndarray
    contrib_fn: object
    weights: pd.DataFrame
    pct_mean: np.ndarray = None      # rank percentile of score_mean (comparable across sets)
    pct_sd: np.ndarray = None        # Monte-Carlo SD of each pixel's rank percentile
    stability: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)
    runtime_s: float = 0.0


# =========================================================================== shared helpers
class Welford:
    def __init__(self, n):
        self.n, self.mean, self.m2 = 0, np.zeros(n, np.float64), np.zeros(n, np.float64)

    def add(self, x):
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)

    @property
    def sd(self):
        return np.sqrt(self.m2 / max(self.n - 1, 1))


def stability(runs_sub, ref_sub, top_fraction):
    """Spearman and top-fraction overlap (Jaccard) of each run against the ensemble mean (on a subsample)."""
    rr = rankdata(ref_sub)
    ranks = rankdata(runs_sub, axis=1)
    sp = np.array([np.corrcoef(r, rr)[0, 1] for r in ranks])
    k = max(int(round(top_fraction * runs_sub.shape[1])), 1)
    ref_top = set(np.argsort(-ref_sub)[:k])
    jac = []
    for r in runs_sub:
        t = set(np.argsort(-r)[:k])
        jac.append(len(t & ref_top) / len(t | ref_top))
    return dict(n_runs=int(runs_sub.shape[0]), mean_spearman=float(np.nanmean(sp)), min_spearman=float(np.nanmin(sp)),
                mean_top_overlap=float(np.mean(jac)), min_top_overlap=float(np.min(jac)))


def top_jaccard(a, b, frac):
    k = max(int(round(frac * len(a))), 1)
    ta, tb = set(np.argpartition(-a, k - 1)[:k]), set(np.argpartition(-b, k - 1)[:k])
    return len(ta & tb) / len(ta | tb)


def _norm(w):
    w = np.asarray(w, float)
    s = w.sum()
    return w / s if s > 0 else np.full(len(w), 1.0 / max(len(w), 1))


def _oriented(prep: Prepared):
    """Favourability-oriented values: higher = more favourable (band features -> closeness to centre)."""
    X, Q = prep.X.astype(np.float64), prep.Q
    O = np.empty_like(X)
    for j, (d, c) in enumerate(zip(prep.meta.direction, prep.meta.center)):
        O[:, j] = X[:, j] if d == "high" else (-X[:, j] if d == "low" else -np.abs(Q[:, j] - c))
    return O


# =========================================================================== M1 fuzzy gamma
RI = {1: 0, 2: 0, 3: .58, 4: .9, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49, 11: 1.51, 12: 1.48}


def ahp_weights(families, acfg):
    k = len(families)
    if k == 1:
        return np.array([1.0]), 0.0
    scores = acfg.get("priority_scores", {})
    A = np.ones((k, k))
    for i, fi in enumerate(families):
        for j, fj in enumerate(families):
            A[i, j] = np.clip(scores.get(fi, 1.0) / scores.get(fj, 1.0), 1 / 9, 9)
    for a, b, v in acfg.get("pairwise", []) or []:
        if a in families and b in families:
            i, j = families.index(a), families.index(b)
            A[i, j], A[j, i] = v, 1.0 / v
    vals, vecs = np.linalg.eig(A)
    m = int(np.argmax(vals.real))
    w = _norm(np.abs(vecs[:, m].real))
    ci = (vals[m].real - k) / (k - 1) if k > 2 else 0.0
    cr = ci / RI.get(k, 1.5) if RI.get(k, 1.5) > 0 else 0.0
    return w, float(cr)


def _membership(Q, dirs, centers, widths, lo, hi):
    M = np.empty(Q.shape, np.float64)
    for j, d in enumerate(dirs):
        q = Q[:, j]
        if d == "high":
            m = (q - lo[j]) / (hi[j] - lo[j])
        elif d == "low":
            m = ((1 - q) - lo[j]) / (hi[j] - lo[j])
        else:
            m = np.exp(-0.5 * ((q - centers[j]) / widths[j]) ** 2)
        M[:, j] = np.clip(m, 0, 1)
    return M


def _gamma(Fk, w, gamma):
    Fk = np.clip(Fk, EPS, 1 - EPS)
    P = np.exp(np.log(Fk) @ w)                      # weighted geometric mean (fuzzy product)
    S = 1.0 - np.exp(np.log1p(-Fk) @ w)             # weighted fuzzy algebraic sum
    return P ** (1 - gamma) * S ** gamma


@dataclass
class M1Spec:
    """Everything that defines one fuzzy gamma model on (a subset of) a prepared feature set."""
    idx: np.ndarray                 # columns of prep.Q used
    features: list
    dirs: list
    cen: np.ndarray
    wid: np.ndarray
    lw: np.ndarray                  # layer weights (within family)
    fam_of: np.ndarray              # family index of each used column
    fams: list
    wf: np.ndarray                  # AHP family weights
    cr: float
    lo: np.ndarray
    hi: np.ndarray
    gamma: float


def build_m1(cfg, prep: Prepared, cols=None) -> M1Spec:
    mc = cfg["models"]["fuzzy_gamma"]
    idx = np.arange(len(prep.features)) if cols is None else np.asarray(cols, int)
    meta = prep.meta.iloc[idx]
    fams = list(dict.fromkeys(meta.family.values))
    wf, cr = ahp_weights(fams, mc["ahp"])
    F = len(idx)
    return M1Spec(idx=idx, features=[prep.features[i] for i in idx], dirs=list(meta.direction),
                  cen=meta.center.values.astype(float), wid=meta.width.values.astype(float),
                  lw=meta.weight.values.astype(float), fam_of=np.array([fams.index(f) for f in meta.family.values]),
                  fams=fams, wf=wf, cr=cr, lo=np.full(F, mc["membership"]["lo"]), hi=np.full(F, mc["membership"]["hi"]),
                  gamma=float(np.mean(mc["gamma_range"])))


def _G(spec: M1Spec, lw):
    G = np.zeros((len(spec.idx), len(spec.fams)))
    for k in range(len(spec.fams)):
        sel = spec.fam_of == k
        tot = lw[sel].sum()
        if tot > 0:
            G[sel, k] = lw[sel] / tot
    return G


def m1_score(prep: Prepared, spec: M1Spec, lw=None, wf=None, lo=None, hi=None, gamma=None, rows=None):
    """Fuzzy gamma score for every pixel (or the given rows). Any argument left None uses the spec value."""
    lw = spec.lw if lw is None else np.asarray(lw, float)
    wf = spec.wf if wf is None else np.asarray(wf, float)
    lo = spec.lo if lo is None else lo
    hi = spec.hi if hi is None else hi
    gamma = spec.gamma if gamma is None else gamma
    Q = prep.Q[:, spec.idx] if rows is None else prep.Q[rows][:, spec.idx]
    G = _G(spec, lw)
    wf = _norm(wf * (G.sum(0) > 0))                 # a family whose layers all have weight 0 drops out
    M = _membership(Q, spec.dirs, spec.cen, spec.wid, lo, hi)
    return _gamma(M @ G, wf, gamma)


def m1_feature_weights(spec: M1Spec):
    """Effective weight of each layer = AHP family weight x within-family layer share."""
    return (_G(spec, spec.lw) @ spec.wf)


def run_m1(cfg, prep: Prepared) -> ModelResult:
    t0 = time.time()
    mc, seed = cfg["models"]["fuzzy_gamma"], cfg["project"]["seed"]
    spec = build_m1(cfg, prep)
    if spec.cr > 0.1:
        log.warning("[%s] AHP consistency ratio %.2f > 0.10: revisit ahp.pairwise", prep.set_name, spec.cr)
    F, N = len(spec.idx), prep.X.shape[0]
    rng = np.random.default_rng(seed)
    sub = rng.choice(N, min(N, cfg["validation"]["stability_sample"]), replace=False)
    n_mc, mj, wj = mc["n_mc"], mc["membership"]["jitter"], mc["weight_jitter"]
    runs = np.empty((n_mc, len(sub)), np.float32)
    acc, accp = Welford(N), Welford(N)
    for r in range(n_mc):
        lo = np.clip(spec.lo + rng.uniform(-mj, mj, F), 0.0, 0.9)
        hi = np.minimum(np.maximum(spec.hi + rng.uniform(-mj, mj, F), lo + 0.05), 1.0)
        wf = _norm(spec.wf * rng.uniform(1 - wj, 1 + wj, len(spec.wf)))
        s = m1_score(prep, spec, wf=wf, lo=lo, hi=hi, gamma=rng.uniform(*mc["gamma_range"]))
        acc.add(s)
        accp.add(U.fast_pct(s))
        runs[r] = s[sub]
    mean, sd = acc.mean.astype(np.float32), acc.sd.astype(np.float32)
    fw = m1_feature_weights(spec)
    dirs, cen, wid = spec.dirs, spec.cen, spec.wid

    def contrib(pos):
        return _membership(prep.Q[pos], dirs, cen, wid, spec.lo, spec.hi) * fw[None, :]

    weights = pd.DataFrame(dict(feature=prep.features, family=prep.meta.family.values, direction=dirs,
                                layer_weight=spec.lw, family_weight=spec.wf[spec.fam_of], weight=fw))
    return ModelResult("M1", prep.set_name, prep, mean, sd, contrib, weights,
                       pct_mean=U.pct_rank(mean), pct_sd=accp.sd.astype(np.float32),
                       stability=stability(runs, mean[sub], cfg["validation"]["top_fraction"]),
                       extra=dict(ahp_consistency_ratio=spec.cr, gamma_range=mc["gamma_range"], n_mc=n_mc,
                                  family_weights=dict(zip(spec.fams, map(float, spec.wf))), spec=spec,
                                  contribution_note="weighted fuzzy membership (final score is a non-additive gamma combination)"),
                       runtime_s=time.time() - t0)


# =========================================================================== one-at-a-time sensitivity
def oat_sensitivity(cfg, res: ModelResult, point_pos, leaf):
    """Perturb one layer weight (or one family weight) at a time; measure how the map and the points move.

    Reported per perturbation: Spearman with the baseline (subsample), Jaccard of the top-fraction pixels,
    the share of the baseline points that stay in the top fraction, and their mean percentile drop.
    """
    sc = cfg["models"]["fuzzy_gamma"].get("sensitivity", {}) or {}
    prep, spec = res.prep, res.extra["spec"]
    tf = cfg["points"]["top_fraction"]
    N = prep.X.shape[0]
    rng = np.random.default_rng(cfg["project"]["seed"] + 7)
    sub = rng.choice(N, min(N, sc.get("sample", 20000)), replace=False)
    base = m1_score(prep, spec)
    bp = U.fast_pct(base)
    rb = rankdata(base[sub])
    point_pos = np.asarray(point_pos, int)

    def measure(s):
        p = U.fast_pct(s)
        d = dict(spearman_vs_base=float(np.corrcoef(rb, rankdata(s[sub]))[0, 1]),
                 top_jaccard_vs_base=float(top_jaccard(base, s, tf)))
        if point_pos.size:
            d["points_still_in_top"] = float((p[point_pos] >= 1 - tf).mean())
            d["points_mean_pct_drop"] = float(100 * (bp[point_pos] - p[point_pos]).mean())
        return d

    rows = []
    for j, f in enumerate(spec.features):
        for fac in sc.get("layer_factors", [0.0, 0.5, 1.5]):
            lw = spec.lw.copy()
            lw[j] *= fac
            rows.append(dict(level="layer", item=f, family=spec.fams[spec.fam_of[j]], factor=fac,
                             **measure(m1_score(prep, spec, lw=lw))))
    for k, fam in enumerate(spec.fams):
        for fac in sc.get("family_factors", [0.0, 0.5, 1.5]):
            if len(spec.fams) == 1 and fac == 0:
                continue
            wf = spec.wf.copy()
            wf[k] *= fac
            rows.append(dict(level="family", item=fam, family=fam, factor=fac, **measure(m1_score(prep, spec, wf=wf))))
    df = pd.DataFrame(rows)
    df["impact"] = 1 - df["top_jaccard_vs_base"]
    df.to_csv(leaf / "sensitivity_oat.csv", index=False)
    drop = df[(df.level == "layer") & (df.factor == 0.0)].sort_values("impact", ascending=False)
    if len(drop):
        fig, ax = plt.subplots(figsize=(7, 0.3 * min(len(drop), 25) + 1.5))
        top = drop.head(25)[::-1]
        ax.barh(top["item"], top["impact"])
        ax.set(xlabel="1 - top-5% Jaccard when the layer is dropped",
               title=f"{res.set_name}: which layers drive the top area")
        (leaf / "plots").mkdir(exist_ok=True)
        fig.savefig(leaf / "plots" / "sensitivity_drop_one.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
    return df


# =========================================================================== layer-count experiment
def layer_count_experiment(cfg, prep: Prepared, k_values, n_subsets, rows=None, pos_rows=None, seed_offset=0):
    """Random k-layer subsets of one set: how the raw score distribution (and, with occurrences, the AUC)
    depends on the number of layers. rows = pixel subsample; pos_rows = occurrence pixels (calibration)."""
    rng = np.random.default_rng(cfg["project"]["seed"] + 11 + seed_offset)
    F = len(prep.features)
    out = []
    for k in k_values:
        if k > F or k < 1:
            continue
        reps = 1 if k == F else n_subsets
        for r in range(reps):
            cols = np.sort(rng.choice(F, k, replace=False)) if k < F else np.arange(F)
            spec = build_m1(cfg, prep, cols)
            s = m1_score(prep, spec, rows=rows)
            q = np.percentile(s, [5, 25, 50, 75, 95])
            d = dict(set=prep.set_name, k=k, rep=r, n_families=len(spec.fams), raw_mean=float(s.mean()),
                     raw_sd=float(s.std()), raw_iqr=float(q[3] - q[1]), raw_p95_minus_p50=float(q[4] - q[2]),
                     features="|".join(spec.features))
            if pos_rows is not None and len(pos_rows):
                sp = m1_score(prep, spec, rows=pos_rows)
                d["auc_calibration"] = float(np.mean(np.searchsorted(np.sort(s), sp) / len(s)))
            out.append(d)
    return pd.DataFrame(out)

# =========================================================================== M2 isolation forest
def _one_sided_z(prep: Prepared, mc):
    O = _oriented(prep)
    med = np.median(O, axis=0)
    mad = 1.4826 * np.median(np.abs(O - med), axis=0)
    sd = O.std(axis=0)
    scale = np.where(mad > 1e-12, mad, np.where(sd > 1e-12, sd, 1.0))
    return np.clip((O - med) / scale, mc["clip_min"], mc["clip_max"]).astype(np.float32)


def run_m2(cfg, prep: Prepared) -> ModelResult:
    t0 = time.time()
    mc, seed = cfg["models"]["isolation_forest"], cfg["project"]["seed"]
    Z = _one_sided_z(prep, mc)
    N, F = Z.shape
    rng = np.random.default_rng(seed)
    sub = rng.choice(N, min(N, cfg["validation"]["stability_sample"]), replace=False)
    runs = np.empty((mc["n_forests"], len(sub)), np.float32)
    acc, accp, kept = Welford(N), Welford(N), []
    for i in range(mc["n_forests"]):
        clf = IsolationForest(n_estimators=mc["n_estimators"], max_samples=min(mc["max_samples"], N),
                              max_features=mc["max_features"], random_state=int(rng.integers(1 << 31)),
                              n_jobs=cfg["project"]["n_jobs"]).fit(Z)
        s = -clf.score_samples(Z)
        acc.add(s)
        accp.add(U.fast_pct(s))
        runs[i] = s[sub]
        if len(kept) < mc["attr_n_forests"]:
            kept.append(clf)
    mean, sd = acc.mean.astype(np.float32), acc.sd.astype(np.float32)

    attribution = {"used": None}

    def contrib(pos):
        Zp = Z[pos]
        base = np.mean([-c.score_samples(Zp) for c in kept], axis=0)
        if mc.get("attribution", "auto") in ("auto", "shap"):
            try:
                import shap
                sv = np.mean([shap.TreeExplainer(c).shap_values(Zp) for c in kept], axis=0)
                tot = sv.sum(1)
                if np.std(tot) > 0 and np.std(base) > 0:
                    if np.corrcoef(tot, base)[0, 1] < 0:
                        sv = -sv                         # orient so positive = more anomalous/favourable
                    attribution["used"] = "shap"
                    return sv
            except Exception as e:  # shap missing / incompatible -> occlusion
                if mc.get("attribution") == "shap":
                    raise
                log.debug("SHAP unavailable (%s); using occlusion attribution", e)
        out = np.zeros((len(pos), F))
        for j in range(F):
            Zo = Zp.copy()
            Zo[:, j] = 0.0                                # neutral value = no favourable signal
            out[:, j] = base - np.mean([-c.score_samples(Zo) for c in kept], axis=0)
        attribution["used"] = "occlusion"
        return out

    weights = pd.DataFrame(dict(feature=prep.features, family=prep.meta.family.values,
                                direction=prep.meta.direction.values,
                                weight=np.asarray(Z.mean(0) / max(Z.mean(0).sum(), 1e-12))))
    res = ModelResult("M2", prep.set_name, prep, mean, sd, contrib, weights,
                      pct_mean=U.pct_rank(mean), pct_sd=accp.sd.astype(np.float32),
                      stability=stability(runs, mean[sub], cfg["validation"]["top_fraction"]),
                      extra=dict(n_forests=mc["n_forests"], attribution=attribution,
                           contribution_note="SHAP (or baseline-occlusion) on the forest, oriented so positive = raises favourability"),
                      runtime_s=time.time() - t0)
    return res


# =========================================================================== M3 TOPSIS + CRITIC
def run_m3(cfg, prep: Prepared) -> ModelResult:
    t0 = time.time()
    mc, seed = cfg["models"]["topsis_critic"], cfg["project"]["seed"]
    meta, X, Q = prep.meta, prep.X.astype(np.float64), prep.Q
    N, F = X.shape
    p_lo, p_hi = np.percentile(X, mc["clip_percentiles"][0], axis=0), np.percentile(X, mc["clip_percentiles"][1], axis=0)
    span = np.where(p_hi - p_lo > 1e-12, p_hi - p_lo, np.nan)
    R = np.empty((N, F), np.float32)
    for j, d in enumerate(meta.direction):
        if d == "band":
            R[:, j] = np.exp(-0.5 * ((Q[:, j] - meta.center.values[j]) / meta.width.values[j]) ** 2)
        else:
            u = np.clip((X[:, j] - p_lo[j]) / span[j], 0, 1) if np.isfinite(span[j]) else np.zeros(N)
            R[:, j] = u if d == "high" else 1 - u
    Rmax, Rmin = R.max(0), R.min(0)
    Dp2, Dm2 = (R - Rmax) ** 2, (R - Rmin) ** 2
    mod = meta.weight.values

    def critic(Rs):
        sd = Rs.std(0)
        C = np.nan_to_num(np.corrcoef(Rs, rowvar=False))
        w = sd * (1 - C).sum(0) * mod
        return _norm(w) if w.sum() > 0 else np.full(F, 1.0 / F)

    def closeness(w):
        dp, dm = np.sqrt(Dp2 @ (w ** 2)), np.sqrt(Dm2 @ (w ** 2))
        return dm / (dp + dm + 1e-12)

    rng = np.random.default_rng(seed)
    w0 = critic(R[rng.choice(N, min(N, mc["max_rows"]), replace=False)])
    sub = rng.choice(N, min(N, cfg["validation"]["stability_sample"]), replace=False)
    runs = np.empty((mc["n_boot"], len(sub)), np.float32)
    acc, accp = Welford(N), Welford(N)
    for b in range(mc["n_boot"]):
        rows = rng.choice(N, min(N, mc["max_rows"]), replace=True)
        w = _norm(critic(R[rows]) * rng.uniform(1 - mc["weight_jitter"], 1 + mc["weight_jitter"], F))
        s = closeness(w)
        acc.add(s)
        accp.add(U.fast_pct(s))
        runs[b] = s[sub]
    mean, sd = acc.mean.astype(np.float32), acc.sd.astype(np.float32)

    def contrib(pos):
        Rp = R[pos].astype(np.float64)
        return (w0 ** 2)[None, :] * (((Rp - Rmin) ** 2) - ((Rp - Rmax) ** 2))

    weights = pd.DataFrame(dict(feature=prep.features, family=meta.family.values, direction=meta.direction.values,
                                weight=w0))
    return ModelResult("M3", prep.set_name, prep, mean, sd, contrib, weights,
                       pct_mean=U.pct_rank(mean), pct_sd=accp.sd.astype(np.float32),
                       stability=stability(runs, mean[sub], cfg["validation"]["top_fraction"]),
                       extra=dict(n_boot=mc["n_boot"],
                                  contribution_note="signed decomposition of (D-)^2 - (D+)^2: positive = moves the pixel toward the ideal point"),
                       runtime_s=time.time() - t0)


RUNNERS = {"M1": run_m1, "M2": run_m2, "M3": run_m3}


# =========================================================================== orchestrator
def _fingerprint(cfg, res: ModelResult):
    """Everything that defines the scores of this run; changes after the freeze are detected from this."""
    keys = ("name_rules", "target_rules", "defaults", "masks", "selection", "audit", "transforms", "layers")
    eng = {k: v for k, v in cfg["engineer"].items() if k != "provenance_check"}
    return U.fingerprint(dict(cfg={k: cfg.get(k) for k in keys}, engineer=eng,
                              model=cfg["models"]["fuzzy_gamma"] if res.model == "M1" else cfg["models"],
                              comparison=cfg.get("comparison", {}).get("common_domain"),
                              weights=res.weights.round(6).to_dict("list")))


def run_models_stage(cfg, models=None, sets=None):
    from .features import Store
    from .points import write_leaf
    store = Store(cfg)
    models = models or list(cfg.get("run", {}).get("models") or ["M1"])
    sets = [s for s in (sets or U.run_sets(cfg)) if store.features_for(s)]
    skipped = [s for s in (U.run_sets(cfg)) if not store.features_for(s)]
    if skipped:
        log.warning("Sets with no selected features are skipped: %s", skipped)
    summary = []
    for s in sets:
        prep = prepare(cfg, store, s)
        for m in models:
            log.info("=== %s | %s ===", m, s)
            res = RUNNERS[m](cfg, prep)
            info = write_leaf(cfg, store, res)
            leaf = U.d_leaf(cfg, m, s)
            if m == "M1":
                oat_sensitivity(cfg, res, info.pop("point_pos"), leaf)
            else:
                info.pop("point_pos", None)
            fp = _fingerprint(cfg, res)
            U.jdump(dict(model=m, set=s, fingerprint=fp, features=prep.features), leaf / "fingerprint.json")
            info["fingerprint"] = fp
            summary.append(info)
            log.info("%s | %s: %d features | %d points | MC stability(Spearman)=%.3f | %.0fs", m, s,
                     len(prep.features), info["n_points"], res.stability.get("mean_spearman", float("nan")), res.runtime_s)
    U.jdump(summary, U.d_stage(cfg, "models") / "models_run_summary.json")
    return summary
