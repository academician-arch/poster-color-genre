# -*- coding: utf-8 -*-
r"""
Color-based genre signals in movie posters -- analysis pipeline
================================================================

Extracts interpretable colour features from the posters of the MM-IMDb dataset
(Arevalo et al. 2017) and relates them to IMDb genre labels.

Stages
  1) Raw-data audit      : order of the genre labels in the MM-IMDb JSON files (alphabetical?),
                           all-label vs. first-position frequencies, labels per film, co-occurrence.
  2) Genre definitions   : A0 = original first-listed-label sample with its stored feature values
                           A  = first-listed label (re-extracted features)
                           B  = MULTI-LABEL, all films (independent of label order -- MAIN ANALYSIS)
                           C  = films with exactly one genre label (independent of label order)
                           D1 = random-label rule (20 draws), D2 = rarest-label rule
  3) Per-genre results   : precision / recall / F1 / support, row-normalised confusion matrix,
                           class-weight and equal-n sensitivity checks.
  4) CIELAB              : all analyses repeated with CIELAB counterparts of the HSV features.
  5) Class-specific SHAP : signed contributions per genre, dependence plots, family-level
                           importance for correlated features.
  6) Robustness          : 10% border crop, 300x300 resize, grayscale weights, k = 4 vs. k = 5.
  7) H3                  : agreement between directional predictions and observed / SHAP directions;
                           pleasure-arousal-dominance indices from Valdez & Mehrabian (1994).

Usage
    pip install -r requirements.txt
    python poster_color_genre_analysis.py

    The working directory defaults to ./work next to this script; set the environment variable
    POSTER_GENRE_BASE_DIR to use another location. MM-IMDb (mmimdb.tar.gz, 8.7 GB) is downloaded
    from the Internet Archive on first use (resumable); an already extracted copy can be given
    through MMIMDB_DIR_OVERRIDE below, and a manually downloaded archive is picked up from
    <BASE_DIR>/mmimdb_official.

    DATA_SOURCE = "demo" runs every stage on a small synthetic dataset in a few minutes (smoke test).
    A full run on about 26,000 posters takes one to three hours; features are cached on disk.

Outputs
    <BASE_DIR>/outputs/tables    (xlsx + csv)
    <BASE_DIR>/outputs/figures   (300 dpi PNG)
    <BASE_DIR>/outputs/SUMMARY_REPORT.txt
    <BASE_DIR>/results.zip
"""

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import os

import json
import glob
import time
import zlib
import hashlib
import zipfile
import tarfile
import datetime
import platform
import warnings
import traceback
import itertools
import collections
import ssl
import shutil
import urllib.request
import urllib.error

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from joblib import Parallel, delayed
from PIL import Image

import sklearn
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import (StratifiedKFold, KFold, cross_validate,
                                     permutation_test_score, cross_val_predict,
                                     train_test_split)
from sklearn.metrics import (f1_score, balanced_accuracy_score, confusion_matrix,
                             classification_report, roc_auc_score, average_precision_score,
                             precision_recall_fscore_support)
from sklearn.inspection import permutation_importance
import skimage
from skimage import color as skcolor
from skimage.measure import shannon_entropy

warnings.filterwarnings("ignore")

# =============================================================================
# SETTINGS
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.environ.get("POSTER_GENRE_BASE_DIR", os.path.join(SCRIPT_DIR, "work"))

DATA_SOURCE = "official"       # "official" | "demo"
QUICK_TEST_MODE = False        # True: quick trial on a small random subset of the real data
SMALL = QUICK_TEST_MODE or DATA_SOURCE == "demo"

RNG_SEED = 42
N_JOBS = -1                    # parallel jobs (-1 = all cores)

# --- preprocessing / features ---
THUMBNAIL_SIZE = 150
K_DOMINANT = 5                 # number of k-means clusters for the dominant colour
SAMPLE_PX = 2000
BORDER_CROP = 0.10             # 10% from each edge

# --- sampling ---
TARGET_N_PER_GENRE = 60 if SMALL else 800     # cap per genre in the single-label designs
K_GENRES_ORIGINAL = 10                        # definition A: the 10 most frequent first-listed labels
MIN_FILMS_PER_GENRE_ML = 80 if SMALL else 800 # definition B: genres carried by at least this many films
MIN_N_SINGLE = 25 if SMALL else 150           # C/D2: minimum number of films per genre
QUICK_N_FILMS = 3000                          # number of films used in QUICK_TEST_MODE
N_DEMO_FILMS = 1500

# --- numbers of permutations ---
N_PERM_STATS = 199 if SMALL else 999          # Pillai / R^2 permutations (fast)
N_PERM_CLASSIFIER = 19 if SMALL else 199      # single-label classifier vs. chance
N_PERM_MULTILABEL = 19 if SMALL else 99       # multi-label, with refitting
PERM_ML_MAX_N = 1200 if SMALL else 8000       # films used in the multi-label permutation test
PERM_ML_TREES = 50 if SMALL else 100
N_RANDOM_LABEL_DRAWS = 5 if SMALL else 20     # D1
N_RANDOM_LABEL_CLF_DRAWS = 2 if SMALL else 5

# --- models ---
RF_TREES = 300                                # single-label Random Forest
ML_RF_TREES = 100 if SMALL else 300           # multi-label (binary) Random Forest
ML_RF_MAX_DEPTH = 12
ML_RF_MIN_LEAF = 10

# --- SHAP ---
SHAP_N_EXPLAIN = 150 if SMALL else 1000       # held-out observations explained (single-label)
SHAP_N_EXPLAIN_LAB = 100 if SMALL else 500
SHAP_N_EXPLAIN_ML = 150 if SMALL else 800     # multi-label, per genre model
FOCUS_GENRES = ["Horror", "Comedy", "Animation", "Fantasy"]

# --- which blocks to run ---
RUN_DEF_A = True               # first-listed-label definition with re-extracted features
RUN_PERM_FOR_A = False         # label-permutation test for definition A (slow; off by default)
RUN_LAB = True
RUN_SENSITIVITY_SUBSET = True
SENSITIVITY_N = 200 if SMALL else 1000
COMPUTE_ARCHIVE_HASH = True    # sha256 of mmimdb.tar.gz (8.7 GB, about 1 min)

MMIMDB_URL = "https://archive.org/download/mmimdb/mmimdb.tar.gz"
MMIMDB_DIR_OVERRIDE = r""      # path of an already extracted MM-IMDb folder (empty = search automatically)
DOWNLOAD_MAX_RETRIES = 60      # download retries (the download resumes where it stopped)

# --- folders ---
MMIMDB_OFFICIAL_DIR = os.path.join(BASE_DIR, "mmimdb_official")
MMIMDB_DEMO_DIR = os.path.join(BASE_DIR, "mmimdb_demo")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs" + ("_demo" if DATA_SOURCE == "demo" else ("_quick" if QUICK_TEST_MODE else "")))
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
TAB_DIR = os.path.join(OUTPUT_DIR, "tables")
CACHE_DIR = os.path.join(BASE_DIR, "cache" + ("_demo" if DATA_SOURCE == "demo" else ""))
ZIP_PATH = os.path.join(BASE_DIR, "results" + ("_demo" if DATA_SOURCE == "demo" else ("_quick" if QUICK_TEST_MODE else "")) + ".zip")
LOG_PATH = os.path.join(BASE_DIR, "run_log.txt")
ORIGINAL_FEATURES_CANDIDATES = [
    os.path.join(SCRIPT_DIR, "data", "first_listed_sample_features.xlsx"),
    os.path.join(BASE_DIR, "first_listed_sample_features.xlsx"),
]

IMDB_ORDER_EXCEPTIONS = ("Short", "Animation", "Documentary")

# --- feature sets ---
FEATURE_COLS = ["mean_hue_sin", "mean_hue_cos", "mean_saturation", "std_saturation",
                "mean_value", "std_value", "warm_cool_logodds", "colorfulness",
                "entropy", "contrast", "dom1_hue_sin", "dom1_hue_cos",
                "dom1_saturation", "dom1_value", "dom1_proportion"]
# CIELAB counterparts: V -> L*, S -> C*ab (chroma), hue -> h_ab
FEATURE_COLS_LAB = ["mean_hab_sin", "mean_hab_cos", "mean_chroma", "std_chroma",
                    "mean_L", "std_L", "warm_cool_logodds", "colorfulness",
                    "entropy", "contrast", "dom1_hab_sin", "dom1_hab_cos",
                    "dom1_chroma", "dom1_L", "dom1_proportion"]
HSV_TO_LAB = dict(zip(FEATURE_COLS, FEATURE_COLS_LAB))
LAB_ONLY_COLS = [c for c in FEATURE_COLS_LAB if c not in FEATURE_COLS]
CROP_COLS = ["crop_" + c for c in FEATURE_COLS]
CACHE_COLS = ["movie_id", "ok", "img_w", "img_h"] + FEATURE_COLS + LAB_ONLY_COLS + CROP_COLS

FEATURE_FAMILY = {
    "mean_value": "Value / lightness", "std_value": "Value / lightness", "dom1_value": "Value / lightness",
    "mean_L": "Value / lightness", "std_L": "Value / lightness", "dom1_L": "Value / lightness",
    "mean_saturation": "Saturation / chroma", "std_saturation": "Saturation / chroma", "dom1_saturation": "Saturation / chroma",
    "mean_chroma": "Saturation / chroma", "std_chroma": "Saturation / chroma", "dom1_chroma": "Saturation / chroma",
    "mean_hue_sin": "Hue", "mean_hue_cos": "Hue", "dom1_hue_sin": "Hue", "dom1_hue_cos": "Hue",
    "mean_hab_sin": "Hue", "mean_hab_cos": "Hue", "dom1_hab_sin": "Hue", "dom1_hab_cos": "Hue",
    "warm_cool_logodds": "Hue",
    "colorfulness": "Colorfulness",
    "entropy": "Complexity / contrast", "contrast": "Complexity / contrast",
    "dom1_proportion": "Dominant-colour share",
}

# Directional predictions for H3: (genre, HSV feature, expected sign, association, basis).
# They combine an assumed affective tone per genre with published colour-affect relations;
# they were fixed before the analysis was run but were not preregistered.
H3_PREDICTIONS = [
    ("Animation",   "colorfulness",      +1, "playfulness -> high colorfulness",        "genre-affect assumption + colour-affect literature"),
    ("Comedy",      "colorfulness",      +1, "playfulness -> high colorfulness",        "genre-affect assumption + colour-affect literature"),
    ("Family",      "colorfulness",      +1, "playfulness -> high colorfulness",        "genre-affect assumption + colour-affect literature"),
    ("Comedy",      "mean_value",        +1, "positive valence -> light/bright",        "Valdez & Mehrabian 1994 (pleasure ~ brightness)"),
    ("Documentary", "mean_saturation",   -1, "seriousness/realism -> low saturation",   "genre-affect assumption + colour-affect literature"),
    ("Drama",       "mean_saturation",   -1, "seriousness/realism -> low saturation",   "genre-affect assumption + colour-affect literature"),
    ("Crime",       "mean_saturation",   -1, "seriousness/realism -> low saturation",   "genre-affect assumption + colour-affect literature"),
    ("Action",      "mean_saturation",   +1, "arousal/danger -> high saturation",       "Valdez & Mehrabian 1994 (arousal ~ saturation)"),
    ("Horror",      "mean_value",        -1, "threat/fear -> dark",                     "genre-affect assumption + colour-affect literature"),
    ("Thriller",    "mean_value",        -1, "suspense -> dark",                        "genre-affect assumption + colour-affect literature"),
    ("Sci-Fi",      "warm_cool_logodds", -1, "design convention: technology -> cool hue",       "genre-affect assumption + colour-affect literature"),
    ("Sci-Fi",      "mean_value",        -1, "design convention: dark",                      "genre-affect assumption + colour-affect literature"),
    ("Romance",     "warm_cool_logodds", +1, "intimacy -> warm hue",                    "genre-affect assumption + colour-affect literature"),
    ("Romance",     "mean_value",        +1, "intimacy -> light/bright",                "genre-affect assumption + colour-affect literature"),
]

plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 300, "font.size": 11,
                     "axes.titlesize": 13, "axes.labelsize": 11})

TABLES = []      # (name, DataFrame, index_flag) -> for the combined workbook
SUMMARY = []     # lines of the summary report
FAILED = []      # stages that raised an error


# =============================================================================
# GENERAL HELPERS
# =============================================================================
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass


def banner(text):
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def note(line):
    """Print a line and keep it for the summary report."""
    print(line)
    SUMMARY.append(line)


def run_stage(name, fn, *args, critical=False, **kwargs):
    banner("STAGE: " + name)
    t0 = time.time()
    try:
        out = fn(*args, **kwargs)
        print("[DONE] " + name + " (" + "{:.1f}".format((time.time() - t0) / 60.0) + " min)")
        return out
    except Exception:
        print("[ERROR] " + name + " failed:")
        traceback.print_exc()
        FAILED.append(name)
        plt.close("all")
        if critical:
            raise
        return None


def _listlike_to_text(v):
    if isinstance(v, (list, tuple, set)):
        return "; ".join(str(x) for x in v)
    return v


def save_table(df, name, index=False, combine=True):
    d = df.copy()
    for c in d.columns:
        if d[c].dtype == object:
            d[c] = d[c].apply(_listlike_to_text)
    try:
        d.to_excel(os.path.join(TAB_DIR, name + ".xlsx"), index=index, engine="openpyxl")
    except Exception as e:
        print("  [WARNING] could not write xlsx (" + name + "): " + str(e))
    d.to_csv(os.path.join(TAB_DIR, name + ".csv"), index=index, encoding="utf-8")
    if combine:
        TABLES.append((name, d, index))
    return d


def save_fig(name):
    try:
        plt.tight_layout()
    except Exception:
        pass
    plt.savefig(os.path.join(FIG_DIR, name + ".png"), dpi=300, bbox_inches="tight")
    plt.close("all")


def id_to_int(mid):
    try:
        return int(str(mid))
    except Exception:
        return zlib.crc32(str(mid).encode("utf-8"))


def labels_of(df):
    """Return the label column as a plain numpy object array (compatible with pandas >= 3 string arrays)."""
    return np.array([str(v) for v in df["label"]], dtype=object)


def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


def safe_spearman(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if np.all(x == x[0]) or np.all(y == y[0]):
        return 0.0
    r = stats.spearmanr(x, y)[0]
    return 0.0 if not np.isfinite(r) else float(r)


def fmt(x, nd=3):
    try:
        return ("{:." + str(nd) + "f}").format(float(x))
    except Exception:
        return str(x)


# =============================================================================
# MULTIVARIATE LINEAR MODEL (Pillai's trace + per-feature R^2 / eta^2)
#   Single-label design: D = one-hot -> classical one-way MANOVA Pillai's trace and eta^2
#   Multi-label design: D = binary label matrix -> same statistic, independent of label order
# =============================================================================
def mv_prepare(X, D):
    Xc = X - X.mean(axis=0)
    Dc = D - D.mean(axis=0)
    U, s, _ = np.linalg.svd(Dc, full_matrices=False)
    r = int((s > s.max() * 1e-10).sum())
    Q = U[:, :r]
    T = Xc.T @ Xc
    Tinv = np.linalg.pinv(T)
    return Xc, Q, T, Tinv


def mv_stats(Xc, Q, T, Tinv):
    B = Q.T @ Xc
    H = B.T @ B
    pillai = float(np.trace(H @ Tinv))
    r2 = np.diag(H) / np.diag(T)
    return pillai, r2


def mv_permutation_test(X, D, n_perm, seed):
    Xc, Q, T, Tinv = mv_prepare(X, D)
    obs_pillai, obs_r2 = mv_stats(Xc, Q, T, Tinv)
    rng = np.random.default_rng(seed)
    count_p = 0
    count_r2 = np.zeros(X.shape[1])
    n = X.shape[0]
    for _ in range(n_perm):
        perm = rng.permutation(n)
        pl, r2 = mv_stats(Xc, Q[perm], T, Tinv)
        count_p += pl >= obs_pillai
        count_r2 += r2 >= obs_r2
    p_pillai = (1 + count_p) / (1 + n_perm)
    p_r2 = (1 + count_r2) / (1 + n_perm)
    return obs_pillai, p_pillai, obs_r2, p_r2


# =============================================================================
# COLOUR FEATURES
# =============================================================================
def colorfulness_index(rgb_img):
    img = rgb_img.astype(float)
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    rg, yb = R - G, 0.5 * (R + G) - B
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def gray_bt601(rgb_img):
    # direct weighted sum, ITU-R BT.601 weights (float, range 0-255)
    return np.dot(rgb_img[..., :3].astype(float), [0.2989, 0.5870, 0.1140])


def color_features(rgb_img, rng, k_dominant=K_DOMINANT, sample_px=SAMPLE_PX, with_lab=True):
    """Fifteen HSV-based colour features plus their CIELAB counterparts."""
    hsv = mcolors.rgb_to_hsv(rgb_img.astype(float) / 255.0)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    hue_rad = H * 2 * np.pi

    warm_mask = (H < (60 / 360)) | (H > (300 / 360))
    warm_frac = warm_mask.mean()
    cool_frac = 1 - warm_frac
    eps = 1e-3
    warm_cool_logodds = np.log((warm_frac + eps) / (cool_frac + eps))

    gray = gray_bt601(rgb_img)

    pixels = rgb_img.reshape(-1, 3).astype(float)
    idx = rng.choice(pixels.shape[0], size=min(sample_px, pixels.shape[0]), replace=False)
    km = KMeans(n_clusters=k_dominant, n_init=4, random_state=0).fit(pixels[idx])
    counts = np.bincount(km.labels_, minlength=k_dominant)
    top = int(np.argsort(-counts)[0])
    center = km.cluster_centers_[top]
    dom_hsv = mcolors.rgb_to_hsv((center / 255.0).reshape(1, 1, 3)).ravel()

    out = {
        "mean_hue_sin": float(np.mean(np.sin(hue_rad))), "mean_hue_cos": float(np.mean(np.cos(hue_rad))),
        "mean_saturation": float(S.mean()), "std_saturation": float(S.std()),
        "mean_value": float(V.mean()), "std_value": float(V.std()),
        "warm_cool_logodds": float(warm_cool_logodds),
        "colorfulness": colorfulness_index(rgb_img),
        "entropy": float(shannon_entropy(gray.astype(np.uint8))),
        "contrast": float(gray.std()),
        "dom1_hue_sin": float(np.sin(dom_hsv[0] * 2 * np.pi)), "dom1_hue_cos": float(np.cos(dom_hsv[0] * 2 * np.pi)),
        "dom1_saturation": float(dom_hsv[1]), "dom1_value": float(dom_hsv[2]),
        "dom1_proportion": float(counts[top] / counts.sum()),
    }
    if with_lab:
        lab = skcolor.rgb2lab(rgb_img.astype(float) / 255.0)   # sRGB, D65, 2-degree observer
        L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
        chroma = np.hypot(a, b)
        hab = np.arctan2(b, a)
        dom_lab = skcolor.rgb2lab((np.clip(center, 0, 255) / 255.0).reshape(1, 1, 3)).ravel()
        dom_hab = np.arctan2(dom_lab[2], dom_lab[1])
        out.update({
            "mean_L": float(L.mean()), "std_L": float(L.std()),
            "mean_chroma": float(chroma.mean()), "std_chroma": float(chroma.std()),
            "mean_hab_sin": float(np.mean(np.sin(hab))), "mean_hab_cos": float(np.mean(np.cos(hab))),
            "dom1_L": float(dom_lab[0]), "dom1_chroma": float(np.hypot(dom_lab[1], dom_lab[2])),
            "dom1_hab_sin": float(np.sin(dom_hab)), "dom1_hab_cos": float(np.cos(dom_hab)),
        })
    return out


def extract_one(mid, path, size, seed, crop_frac):
    """One poster: whole-image features (HSV + CIELAB) and HSV features after cropping 10% from each edge."""
    warnings.filterwarnings("ignore")
    try:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        full = np.array(img.resize((size, size)))
        dx, dy = int(round(w * crop_frac)), int(round(h * crop_frac))
        crop = np.array(img.crop((dx, dy, w - dx, h - dy)).resize((size, size)))
    except Exception:
        return {"movie_id": str(mid), "ok": 0}
    try:
        s = seed + id_to_int(mid) % (2 ** 31)
        f = color_features(full, np.random.default_rng(s), with_lab=True)
        fc = color_features(crop, np.random.default_rng(s), with_lab=False)
    except Exception:
        return {"movie_id": str(mid), "ok": 0}
    out = {"movie_id": str(mid), "ok": 1, "img_w": w, "img_h": h}
    out.update(f)
    for k, v in fc.items():
        out["crop_" + k] = v
    return out


def extract_sensitivity(mid, path, size, seed):
    """Subsample sensitivity: 300x300 resize, BT.709 grayscale weights, k=4 dominant colours."""
    warnings.filterwarnings("ignore")
    try:
        img = Image.open(path).convert("RGB")
        small = np.array(img.resize((size, size)))
        big = np.array(img.resize((size * 2, size * 2)))
    except Exception:
        return None
    s = seed + id_to_int(mid) % (2 ** 31)
    out = {"movie_id": str(mid)}
    f_big = color_features(big, np.random.default_rng(s), with_lab=True)
    for k, v in f_big.items():
        out["s300_" + k] = v
    g709 = np.dot(small[..., :3].astype(float), [0.2125, 0.7154, 0.0721])
    out["bt709_entropy"] = float(shannon_entropy(g709.astype(np.uint8)))
    out["bt709_contrast"] = float(g709.std())
    f_k4 = color_features(small, np.random.default_rng(s), k_dominant=4, with_lab=False)
    for k in ["dom1_hue_sin", "dom1_hue_cos", "dom1_saturation", "dom1_value", "dom1_proportion"]:
        out["k4_" + k] = f_k4[k]
    return out


# =============================================================================
# DATASET PREPARATION
# =============================================================================
def _open_url(req, timeout=60):
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.URLError as e:
        # if the system certificate store is incomplete, retry once with the certifi bundle
        if isinstance(getattr(e, "reason", None), ssl.SSLError):
            try:
                import certifi
            except ImportError:
                raise e
            ctx = ssl.create_default_context(cafile=certifi.where())
            return urllib.request.urlopen(req, timeout=timeout, context=ctx)
        raise


def download_resumable(url, dest, max_retries=DOWNLOAD_MAX_RETRIES):
    """Resumable download of a large file (dest + '.part'); retries when the connection drops."""
    part = dest + ".part"
    fails, total, last_print = 0, None, 0.0
    while True:
        have = os.path.getsize(part) if os.path.isfile(part) else 0
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (academic research script)"})
        if have > 0:
            req.add_header("Range", "bytes=" + str(have) + "-")
        try:
            resp = _open_url(req)
            mode = "ab"
            if have > 0 and resp.getcode() != 206:      # server ignored the Range header -> start over
                have, mode = 0, "wb"
            elif have == 0:
                mode = "wb"
            cr = resp.headers.get("Content-Range")
            if cr and "/" in cr and cr.split("/")[-1].strip().isdigit():
                total = int(cr.split("/")[-1])
            elif resp.headers.get("Content-Length"):
                total = have + int(resp.headers.get("Content-Length"))
            start_have = have
            try:
                with open(part, mode) as fh:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        fh.write(chunk)
                        have += len(chunk)
                        if time.time() - last_print > 2:
                            last_print = time.time()
                            if total:
                                print("\r  Downloading: {:5.1f}%  ({:6.2f} / {:5.2f} GB)".format(
                                    100.0 * have / total, have / 1e9, total / 1e9), end="", flush=True)
                            else:
                                print("\r  Downloading: {:6.2f} GB".format(have / 1e9), end="", flush=True)
            finally:
                resp.close()
                if have > start_have:
                    fails = 0
            if total is None or have >= total:
                break
            print("\n  Connection closed early; resuming...")
        except urllib.error.HTTPError as e:
            if e.code == 416 and have > 0:              # requested range not satisfiable = file already complete
                break
            fails += 1
            err = "HTTP " + str(e.code)
        except Exception as e:
            fails += 1
            err = type(e).__name__ + ": " + str(e)
        else:
            continue
        if fails > max_retries:
            raise RuntimeError("Download did not complete after " + str(max_retries) + " retries (last error: " + err + ").")
        wait = min(60, 5 * fails)
        print("\n  [WARNING] Download error (" + err + "). " + str(wait) + " s until the next attempt ("
              + str(fails) + "/" + str(max_retries) + ")...")
        time.sleep(wait)
    print()
    os.replace(part, dest)


def find_dataset_dir():
    """Locate the extracted MM-IMDb folder (first directory tree with at least 1000 json files)."""
    cands = []
    if MMIMDB_DIR_OVERRIDE:
        cands.append(MMIMDB_DIR_OVERRIDE)
    cands += [MMIMDB_OFFICIAL_DIR, os.path.join(BASE_DIR, "mmimdb"), os.path.join(BASE_DIR, "dataset")]
    for c in cands:
        if os.path.isdir(c):
            n = len(glob.glob(os.path.join(c, "**", "*.json"), recursive=True))
            if n >= 1000:
                return c, n
            if c == MMIMDB_DIR_OVERRIDE:
                print("  [WARNING] MMIMDB_DIR_OVERRIDE contains only " + str(n) + " json files: " + c)
    # general search under BASE_DIR (excluding output / cache folders)
    skip = ("outputs", "cache", "mmimdb_demo")
    try:
        for name in sorted(os.listdir(BASE_DIR)):
            full = os.path.join(BASE_DIR, name)
            if os.path.isdir(full) and not name.lower().startswith(skip) and full not in cands:
                n = len(glob.glob(os.path.join(full, "**", "*.json"), recursive=True))
                if n >= 1000:
                    return full, n
    except Exception:
        pass
    return None, 0


def find_archive():
    home = os.path.expanduser("~")
    for c in [os.path.join(MMIMDB_OFFICIAL_DIR, "mmimdb.tar.gz"), os.path.join(BASE_DIR, "mmimdb.tar.gz"),
              os.path.join(home, "Downloads", "mmimdb.tar.gz"), os.path.join(home, "Desktop", "mmimdb.tar.gz")]:
        if os.path.isfile(c):
            return c
    return None


def extract_archive(tar_path):
    free = shutil.disk_usage(MMIMDB_OFFICIAL_DIR).free
    if free < 10 * 1024 ** 3:
        print("  [WARNING] Only " + "{:.1f}".format(free / 1024 ** 3) + " GB of free disk space; extracting the archive needs about 9 GB.")
    print("Extracting archive: " + tar_path + " (may take 5-15 minutes)...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=MMIMDB_OFFICIAL_DIR)


MANUAL_DOWNLOAD_HELP = """
-------------------------------------------------------------------------------
DATASET NOT FOUND / DOWNLOAD FAILED. Three options:

 (1) If MM-IMDb is already extracted elsewhere (a folder with thousands of .json and .jpeg
     files), set in the SETTINGS section of this script
         MMIMDB_DIR_OVERRIDE = r"PATH TO THAT FOLDER"
     and run the script again.

 (2) Run the script again: the download RESUMES where it stopped (the partial file is kept).

 (3) Download manually (8.7 GB):
         {url}
     Put mmimdb.tar.gz into this folder (keep the file name):
         {dest}
     and run the script again; the script extracts the archive itself.
-------------------------------------------------------------------------------
"""


DEMO_BIAS = {
    "Action": (0.02, 0.65, 0.50), "Adventure": (0.10, 0.60, 0.60), "Animation": (0.28, 0.75, 0.72),
    "Biography": (0.08, 0.30, 0.50), "Comedy": (0.13, 0.60, 0.74), "Crime": (0.60, 0.30, 0.38),
    "Documentary": (0.12, 0.25, 0.50), "Drama": (0.08, 0.28, 0.47), "Family": (0.20, 0.65, 0.75),
    "Fantasy": (0.75, 0.50, 0.40), "Horror": (0.98, 0.40, 0.22), "Mystery": (0.62, 0.30, 0.30),
    "Romance": (0.93, 0.45, 0.70), "Sci-Fi": (0.55, 0.50, 0.38), "Thriller": (0.58, 0.35, 0.28),
    "Short": (0.30, 0.40, 0.55),
}
DEMO_POP = {"Drama": 30, "Comedy": 22, "Romance": 12, "Thriller": 12, "Crime": 9, "Action": 9,
            "Adventure": 7, "Horror": 8, "Documentary": 7, "Mystery": 5, "Sci-Fi": 5, "Fantasy": 5,
            "Family": 5, "Biography": 4, "Animation": 5, "Short": 1}


def build_demo_dataset(root, n_films, seed):
    """Write a small synthetic dataset in the MM-IMDb layout (json + jpeg) to disk."""
    ds = os.path.join(root, "dataset")
    os.makedirs(ds, exist_ok=True)
    if len(glob.glob(os.path.join(ds, "*.json"))) >= n_films:
        print("Demo dataset already exists; not regenerated.")
        return
    rng = np.random.default_rng(seed)
    names = list(DEMO_POP.keys())
    probs = np.array([DEMO_POP[g] for g in names], float)
    probs = probs / probs.sum()
    for i in range(n_films):
        k = int(rng.choice([1, 2, 3, 4], p=[0.2, 0.35, 0.3, 0.15]))
        gs = list(rng.choice(names, size=k, replace=False, p=probs))
        exc = [g for g in gs if g in IMDB_ORDER_EXCEPTIONS]
        rest = sorted(g for g in gs if g not in IMDB_ORDER_EXCEPTIONS)
        ordered = exc + rest
        if rng.random() < 0.02:
            ordered = list(rng.permutation(ordered))
        main = gs[int(rng.integers(len(gs)))]
        hue0, sat0, val0 = DEMO_BIAS[main]
        hue_c = (hue0 + rng.normal(0, 0.10)) % 1.0
        sat_c = float(np.clip(np.mean([DEMO_BIAS[g][1] for g in gs]) + rng.normal(0, 0.15), 0.05, 0.95))
        val_c = float(np.clip(np.mean([DEMO_BIAS[g][2] for g in gs]) + rng.normal(0, 0.15), 0.05, 0.95))
        hgt, wid = 180, 120
        h = rng.normal(hue_c, 0.06, size=(hgt, wid)) % 1.0
        s = np.clip(rng.normal(sat_c, 0.10, size=(hgt, wid)), 0, 1)
        v = np.clip(rng.normal(val_c, 0.10, size=(hgt, wid)), 0, 1)
        v[:12, :] = 0.05
        v[-12:, :] = 0.95
        s[-12:, :] = 0.05
        rgb = (mcolors.hsv_to_rgb(np.stack([h, s, v], axis=-1)) * 255).astype(np.uint8)
        mid = "{:07d}".format(100000 + i)
        Image.fromarray(rgb).save(os.path.join(ds, mid + ".jpeg"), "JPEG", quality=90)
        with open(os.path.join(ds, mid + ".json"), "w", encoding="utf-8") as fh:
            json.dump({"genres": ordered, "title": "Demo Film " + str(i),
                       "year": int(rng.integers(1930, 2016))}, fh)
    print(str(n_films) + " synthetic films written to: " + ds)


def prepare_dataset():
    if DATA_SOURCE == "demo":
        build_demo_dataset(MMIMDB_DEMO_DIR, N_DEMO_FILMS, RNG_SEED)
        return MMIMDB_DEMO_DIR
    if DATA_SOURCE != "official":
        raise ValueError("Unknown DATA_SOURCE: " + repr(DATA_SOURCE))
    os.makedirs(MMIMDB_OFFICIAL_DIR, exist_ok=True)
    found, n = find_dataset_dir()
    if found:
        print("Dataset found: " + found + " (" + str(n) + " json files). Download skipped.")
        return found
    print("No extracted dataset found (looked in: " + MMIMDB_OFFICIAL_DIR + ").")
    tar_path = os.path.join(MMIMDB_OFFICIAL_DIR, "mmimdb.tar.gz")
    arch = find_archive()
    if arch is not None:
        print("Archive found: " + arch + " ({:.2f} GB)".format(os.path.getsize(arch) / 1e9))
        try:
            extract_archive(arch)
        except (EOFError, tarfile.ReadError, OSError) as e:
            print("  [WARNING] Archive could not be extracted (" + type(e).__name__ + ": " + str(e) + "); the file may be incomplete.")
            if os.path.abspath(arch) == os.path.abspath(tar_path) and not os.path.isfile(tar_path + ".part"):
                os.replace(tar_path, tar_path + ".part")      # resume from the partial file
            arch = None
    if arch is None:
        print("Downloading: " + MMIMDB_URL + "  (about 8.7 GB; resumes if the connection drops)")
        download_resumable(MMIMDB_URL, tar_path)
        extract_archive(tar_path)
    found, n = find_dataset_dir()
    if not found:
        raise FileNotFoundError("Archive extracted but no json files found in: " + MMIMDB_OFFICIAL_DIR)
    print("Dataset ready: " + found + " (" + str(n) + " json).")
    return found


def load_metadata(root):
    json_files = sorted(glob.glob(os.path.join(root, "**", "*.json"), recursive=True))
    rows = []
    n_bad = 0
    t0 = time.time()
    for i, jf in enumerate(json_files):
        img_path = jf[:-5] + ".jpeg"
        if not os.path.isfile(img_path):
            continue
        try:
            with open(jf, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            n_bad += 1
            continue
        if not isinstance(data, dict):
            continue
        genres = data.get("genres", [])
        if not genres or not isinstance(genres, list):
            continue
        genres = [str(g) for g in genres]
        seen, uniq = set(), []
        for g in genres:
            if g not in seen:
                seen.add(g)
                uniq.append(g)
        year = data.get("year", None)
        try:
            year = int(year)
        except Exception:
            year = np.nan
        rows.append({"movie_id": os.path.splitext(os.path.basename(jf))[0],
                     "jpeg_path": img_path, "title": str(data.get("title", "")),
                     "year": year, "genres": uniq, "n_labels": len(uniq),
                     "first_label": uniq[0]})
        if (i + 1) % 5000 == 0:
            print("  " + str(i + 1) + " / " + str(len(json_files)) + " json files read ("
                  + "{:.0f}".format(time.time() - t0) + " s)")
    if not rows:
        raise FileNotFoundError(root + ": no valid json+jpeg pairs found.")
    meta = pd.DataFrame(rows)
    print("Total: " + str(len(meta)) + " films (json + jpeg + at least one genre label); unreadable json files: " + str(n_bad))
    return meta


# =============================================================================
# STAGE 0: ENVIRONMENT / IMPLEMENTATION DETAILS
# =============================================================================
def stage_environment():
    try:
        import shap as _shap
        shap_version = _shap.__version__
    except Exception:
        shap_version = "NOT INSTALLED"
    env = {
        "run_started": datetime.datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": __import__("scipy").__version__, "scikit-learn": sklearn.__version__,
        "scikit-image": skimage.__version__, "matplotlib": matplotlib.__version__,
        "Pillow": Image.__version__, "shap": shap_version,
        "settings": {"DATA_SOURCE": DATA_SOURCE, "QUICK_TEST_MODE": QUICK_TEST_MODE, "RNG_SEED": RNG_SEED,
                     "THUMBNAIL_SIZE": THUMBNAIL_SIZE, "K_DOMINANT": K_DOMINANT, "SAMPLE_PX": SAMPLE_PX,
                     "BORDER_CROP": BORDER_CROP, "TARGET_N_PER_GENRE": TARGET_N_PER_GENRE,
                     "MIN_FILMS_PER_GENRE_ML": MIN_FILMS_PER_GENRE_ML, "MIN_N_SINGLE": MIN_N_SINGLE,
                     "N_PERM_STATS": N_PERM_STATS, "N_PERM_CLASSIFIER": N_PERM_CLASSIFIER,
                     "N_PERM_MULTILABEL": N_PERM_MULTILABEL, "RF_TREES": RF_TREES,
                     "ML_RF": [ML_RF_TREES, ML_RF_MAX_DEPTH, ML_RF_MIN_LEAF]},
    }
    with open(os.path.join(OUTPUT_DIR, "run_environment.json"), "w", encoding="utf-8") as fh:
        json.dump(env, fh, indent=2)

    details = """IMPLEMENTATION DETAILS
======================

Image loading and resizing
  PIL.Image.open(path).convert("RGB").resize((150, 150))  -- Pillow default resampling filter
  for Image.resize (BICUBIC in Pillow >= 7.0); aspect ratio is NOT preserved (posters are
  squeezed to a square). Pillow version: {pillow}.

HSV conversion
  matplotlib.colors.rgb_to_hsv on RGB scaled to [0, 1]. Value = max(R, G, B); Saturation =
  (max - min) / max; Hue in [0, 1) mapped to radians by theta = 2*pi*H.
  NOTE: HSV Value is a computational (device-dependent) quantity, not perceived lightness.

CIELAB (sensitivity analysis)
  skimage.color.rgb2lab on sRGB scaled to [0, 1] (illuminant D65, 2-degree observer).
  L* = lightness, C*ab = sqrt(a*^2 + b*^2) (chroma), h_ab = atan2(b*, a*).

Grayscale conversion, entropy and contrast
  Grayscale: direct weighted sum, NOT a library function:
      Y = 0.2989*R + 0.5870*G + 0.1140*B      (ITU-R BT.601 luma weights, MATLAB rounding)
  Input: uint8 RGB, 0-255. Output: float64, range 0-255.
  Contrast: population standard deviation (numpy .std(), ddof = 0) of the float Y image.
  Entropy: Y is truncated to uint8 (numpy astype(uint8), i.e. floor -> 256 integer levels),
  then skimage.measure.shannon_entropy: probabilities = relative frequencies of the distinct
  integer levels present in the image; H = -sum(p * log2(p)) (base 2, bits; maximum = 8).
  scikit-image version: {skimage}.

Pseudocode
      rgb   = resize(read_rgb(poster), 150, 150)                # uint8
      Y     = 0.2989*rgb[:,:,0] + 0.5870*rgb[:,:,1] + 0.1140*rgb[:,:,2]
      contrast = std(Y)
      levels, counts = unique(floor(Y) as uint8)
      p = counts / sum(counts);  entropy = -sum(p * log2(p))

Dominant colour
  k-means in RGB space on a random sample of up to {px} pixels per poster; k = {k},
  n_init = 4, random_state = 0 (scikit-learn KMeans). The centre of the most populous cluster
  is the "primary dominant colour"; its share of the sampled pixels is dom1_proportion.
  Pixel subsampling uses numpy default_rng(seed = {seed} + IMDb id), so features are
  reproducible per poster and independent of processing order.

Warm-cool log-odds
  warm = hue < 60 deg or hue > 300 deg; log((p_warm + 0.001) / (p_cool + 0.001)).

SHAP
  shap.TreeExplainer(model) with the default tree-path-dependent algorithm
  (feature_perturbation = "tree_path_dependent"): no separate background data set; the
  reference expectation is the training-data distribution encoded in the trees' cover.
  Output scale for scikit-learn RandomForestClassifier: predicted class PROBABILITY
  (average of the trees' leaf class proportions). A SHAP value is therefore a change in the
  predicted probability of a genre relative to the mean predicted probability.
  Single-label designs: one multi-class forest, SHAP values kept per class (genre).
  Multi-label design: one binary forest per genre, SHAP values of the positive class.
  Correlated predictors: path-dependent TreeSHAP splits credit among correlated features in
  proportion to how the trees use them, so single-feature importances of correlated features
  (e.g. mean_value and dom1_value) are not separately identifiable. We therefore also report
  (i) the Spearman correlation matrix of the features and (ii) family-level importances,
  i.e. mean |sum of SHAP values within a feature family|.
  shap version: {shap}.
""".format(pillow=Image.__version__, skimage=skimage.__version__, px=SAMPLE_PX, k=K_DOMINANT,
           seed=RNG_SEED, shap=shap_version)
    with open(os.path.join(OUTPUT_DIR, "implementation_details.txt"), "w", encoding="utf-8") as fh:
        fh.write(details)
    print("Wrote run_environment.json and implementation_details.txt.")
    if shap_version == "NOT INSTALLED":
        note("!!! WARNING: the 'shap' package is not installed -> class-specific SHAP tables and figures were NOT produced. "
             "Run 'pip install shap' and start the script again (features are read from the cache).")
    return env


# =============================================================================
# STAGE 1: RAW-DATA AUDIT (label order, frequencies, co-occurrence)
# =============================================================================
def order_class(genres):
    if len(genres) < 2:
        return "single_label"
    if list(genres) == sorted(genres):
        return "alphabetical"
    exc = [g for g in genres if g in IMDB_ORDER_EXCEPTIONS]
    rest = [g for g in genres if g not in IMDB_ORDER_EXCEPTIONS]
    k = len(exc)
    if k > 0 and set(genres[:k]) == set(exc) and rest == sorted(rest):
        return "alphabetical_after_imdb_exceptions"
    return "other_order"


def stage_audit(root):
    meta = load_metadata(root)
    n = len(meta)

    # --- data provenance ---
    prov = {"n_films_with_json_jpeg_genre": n,
            "n_json_files": len(glob.glob(os.path.join(root, "**", "*.json"), recursive=True)),
            "n_jpeg_files": len(glob.glob(os.path.join(root, "**", "*.jpeg"), recursive=True)),
            "source_url": MMIMDB_URL if DATA_SOURCE == "official" else "synthetic demo data",
            "dataset_reference": "Arevalo et al. (2017), MM-IMDb; archive.org/details/mmimdb",
            "release_year_min": float(np.nanmin(meta["year"])) if meta["year"].notna().any() else np.nan,
            "release_year_max": float(np.nanmax(meta["year"])) if meta["year"].notna().any() else np.nan}
    tar_path = os.path.join(root, "mmimdb.tar.gz")
    if not os.path.isfile(tar_path):
        tar_path = find_archive() or tar_path
    if os.path.isfile(tar_path):
        prov["archive_bytes"] = os.path.getsize(tar_path)
        prov["archive_modified"] = datetime.datetime.fromtimestamp(os.path.getmtime(tar_path)).isoformat(timespec="seconds")
        if COMPUTE_ARCHIVE_HASH and not SMALL:
            print("Computing sha256 of the archive...")
            hsh = hashlib.sha256()
            with open(tar_path, "rb") as fh:
                for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
                    hsh.update(chunk)
            prov["archive_sha256"] = hsh.hexdigest()
    save_table(pd.DataFrame([{"field": k, "value": v} for k, v in prov.items()]), "R01_dataset_provenance")
    yr = meta["year"].dropna().astype(int).value_counts().sort_index()
    save_table(yr.rename_axis("release_year").reset_index(name="n_films"), "R01b_release_year_distribution")
    note("DATA: " + str(n) + " films; release years " + str(prov["release_year_min"]) + "-" + str(prov["release_year_max"])
         + " (the latest year is a lower bound on when the labels were collected).")

    # --- 1) label-order audit ---
    meta["order_class"] = meta["genres"].apply(order_class)
    meta["first_is_alpha_min"] = [g[0] == min(g) for g in meta["genres"]]
    meta["first_is_exception"] = [g[0] in IMDB_ORDER_EXCEPTIONS for g in meta["genres"]]
    multi = meta[meta["n_labels"] >= 2]
    oc = multi["order_class"].value_counts()
    rows = [{"quantity": "films (total)", "n": n, "share": 1.0},
            {"quantity": "films with a single genre label", "n": int((meta["n_labels"] == 1).sum()),
             "share": float((meta["n_labels"] == 1).mean())},
            {"quantity": "films with >= 2 genre labels", "n": len(multi), "share": len(multi) / n}]
    for k in ["alphabetical", "alphabetical_after_imdb_exceptions", "other_order"]:
        cnt = int(oc.get(k, 0))
        rows.append({"quantity": "multi-label films, genre array order: " + k, "n": cnt,
                     "share": cnt / max(len(multi), 1)})
    rows.append({"quantity": "multi-label films: first label = alphabetically first label",
                 "n": int(multi["first_is_alpha_min"].sum()), "share": float(multi["first_is_alpha_min"].mean())})
    rows.append({"quantity": "multi-label films: first label is an IMDb ordering exception (Short/Animation/Documentary)",
                 "n": int(multi["first_is_exception"].sum()), "share": float(multi["first_is_exception"].mean())})
    both = multi["first_is_alpha_min"] | multi["first_is_exception"]
    rows.append({"quantity": "multi-label films: first label explained by alphabetical rule OR IMDb exception",
                 "n": int(both.sum()), "share": float(both.mean())})
    save_table(pd.DataFrame(rows), "R02_genre_order_audit")
    share_alpha = oc.get("alphabetical", 0) / max(len(multi), 1)
    share_exc = oc.get("alphabetical_after_imdb_exceptions", 0) / max(len(multi), 1)
    note("LABEL ORDER: of " + str(len(multi)) + " multi-label films, the genre array is strictly alphabetical in " + fmt(100 * share_alpha, 1)
         + "% and, in a further " + fmt(100 * share_exc, 1)
         + "%, alphabetical after leading IMDb exceptions (Short/Animation/Documentary). First label "
         + "explained by the alphabetical rule OR an exception (%): " + fmt(100 * both.mean(), 1) + ".")
    ex = meta[meta["order_class"] == "other_order"].head(40)[["movie_id", "title", "year", "genres"]]
    save_table(ex, "R02b_examples_not_alphabetical")

    # --- 2) all-label and first-position frequencies ---
    all_counts = collections.Counter(itertools.chain.from_iterable(meta["genres"]))
    first_counts = meta["first_label"].value_counts()
    single_counts = meta.loc[meta["n_labels"] == 1, "first_label"].value_counts()
    freq = pd.DataFrame({"genre": list(all_counts.keys()), "n_all_labels": list(all_counts.values())})
    freq["n_first_position"] = freq["genre"].map(first_counts).fillna(0).astype(int)
    freq["n_single_label_films"] = freq["genre"].map(single_counts).fillna(0).astype(int)
    freq["share_first_given_present"] = freq["n_first_position"] / freq["n_all_labels"]
    freq["rank_all_labels"] = freq["n_all_labels"].rank(ascending=False, method="min").astype(int)
    freq["rank_first_position"] = freq["n_first_position"].rank(ascending=False, method="min").astype(int)
    freq = freq.sort_values("n_all_labels", ascending=False).reset_index(drop=True)
    freq["in_top10_all_labels"] = freq["rank_all_labels"] <= 10
    freq["in_top10_first_position"] = freq["rank_first_position"] <= 10
    save_table(freq, "R03_genre_frequency_all_vs_first_position")
    top_all = freq.sort_values("n_all_labels", ascending=False)["genre"].head(10).tolist()
    top_first = freq.sort_values("n_first_position", ascending=False)["genre"].head(10).tolist()
    note("GENRE FREQUENCY: top 10 by all labels = " + ", ".join(top_all))
    note("                 top 10 by first-listed label = " + ", ".join(top_first))

    fq = freq.head(24)
    xs = np.arange(len(fq))
    plt.figure(figsize=(12, 5))
    plt.bar(xs - 0.2, fq["n_all_labels"], width=0.4, label="All genre labels", color="#1f77b4")
    plt.bar(xs + 0.2, fq["n_first_position"], width=0.4, label="First-listed label only", color="#ff7f0e")
    plt.xticks(xs, fq["genre"], rotation=60, ha="right")
    plt.ylabel("Number of films")
    plt.title("Genre frequency in MM-IMDb: all labels vs. first-listed label")
    plt.legend()
    save_fig("R_fig01_genre_frequency_all_vs_first")

    # --- 3) labels per film ---
    lab = meta["n_labels"].value_counts().sort_index().rename_axis("n_genre_labels").reset_index(name="n_films")
    lab["share"] = lab["n_films"] / n
    save_table(lab, "R04_labels_per_film")
    note("LABELS PER FILM: mean " + fmt(meta["n_labels"].mean(), 2) + " genre labels; share of single-label films (%): "
         + fmt(100 * (meta["n_labels"] == 1).mean(), 1) + ".")

    # --- 4) co-occurrence ---
    gl = freq["genre"].tolist()
    gi = {g: i for i, g in enumerate(gl)}
    M = np.zeros((len(gl), len(gl)), dtype=int)
    for gs in meta["genres"]:
        ids = [gi[g] for g in gs]
        for a in ids:
            for b in ids:
                M[a, b] += 1
    co = pd.DataFrame(M, index=gl, columns=gl)
    save_table(co.rename_axis("genre").reset_index(), "R05_cooccurrence_counts")
    diag = np.diag(M).astype(float)
    cond = M / np.maximum(diag[:, None], 1)
    save_table(pd.DataFrame(cond, index=gl, columns=gl).rename_axis("row_genre__P(col|row)").reset_index(),
               "R06_cooccurrence_conditional_probability")
    jac = M / np.maximum(diag[:, None] + diag[None, :] - M, 1)
    save_table(pd.DataFrame(jac, index=gl, columns=gl).rename_axis("genre").reset_index(), "R07_cooccurrence_jaccard")

    k = min(len(gl), 20)
    plt.figure(figsize=(11, 9))
    cc = cond[:k, :k].copy()
    np.fill_diagonal(cc, np.nan)
    im = plt.imshow(cc, cmap="viridis", vmin=0, vmax=np.nanmax(cc))
    plt.xticks(range(k), gl[:k], rotation=60, ha="right")
    plt.yticks(range(k), gl[:k])
    plt.colorbar(im, label="P(column genre | row genre)")
    plt.title("Genre co-occurrence: share of row-genre films that also carry the column genre")
    for i in range(k):
        for j in range(k):
            if i != j and cc[i, j] >= 0.25:
                plt.text(j, i, "{:.2f}".format(cc[i, j]), ha="center", va="center", fontsize=6, color="white")
    save_fig("R_fig02_genre_cooccurrence")
    return meta


# =============================================================================
# STAGE 2: FEATURE EXTRACTION (all films, parallel, cached)
# =============================================================================
def stage_features(meta):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, "features_all.csv")
    params_path = os.path.join(CACHE_DIR, "features_params.json")
    params = {"size": THUMBNAIL_SIZE, "k": K_DOMINANT, "sample_px": SAMPLE_PX, "seed": RNG_SEED,
              "crop": BORDER_CROP, "cols": CACHE_COLS, "version": 2}
    if os.path.isfile(cache_path) and os.path.isfile(params_path):
        try:
            with open(params_path, encoding="utf-8") as fh:
                old = json.load(fh)
        except Exception:
            old = None
        if old != params:
            print("Cache settings changed; features will be extracted from scratch.")
            os.remove(cache_path)
    elif os.path.isfile(cache_path):
        os.remove(cache_path)
    with open(params_path, "w", encoding="utf-8") as fh:
        json.dump(params, fh)

    work = meta
    if QUICK_TEST_MODE and DATA_SOURCE == "official" and len(meta) > QUICK_N_FILMS:
        work = meta.sample(n=QUICK_N_FILMS, random_state=RNG_SEED)
        print("QUICK_TEST_MODE: " + str(QUICK_N_FILMS) + " randomly selected films are used.")

    done = set()
    if os.path.isfile(cache_path):
        done = set(pd.read_csv(cache_path, usecols=["movie_id"], dtype={"movie_id": str})["movie_id"])
        print("Cache holds " + str(len(done)) + " films.")
    todo = work[~work["movie_id"].isin(done)]
    print("Posters to process: " + str(len(todo)))
    CH = 500
    t0 = time.time()
    for start in range(0, len(todo), CH):
        part = todo.iloc[start:start + CH]
        res = Parallel(n_jobs=N_JOBS)(
            delayed(extract_one)(mid, path, THUMBNAIL_SIZE, RNG_SEED, BORDER_CROP)
            for mid, path in zip(part["movie_id"], part["jpeg_path"]))
        dfc = pd.DataFrame(res).reindex(columns=CACHE_COLS)
        dfc.to_csv(cache_path, mode="a", index=False, header=not os.path.isfile(cache_path))
        n_done = min(start + CH, len(todo))
        el = time.time() - t0
        eta = el / n_done * (len(todo) - n_done)
        print("  " + str(n_done) + " / " + str(len(todo)) + " posters  |  elapsed " + "{:.1f}".format(el / 60)
              + " min, remaining ~" + "{:.1f}".format(eta / 60) + " min")

    cache = pd.read_csv(cache_path, dtype={"movie_id": str}).drop_duplicates("movie_id")
    n_fail = int((cache["ok"] != 1).sum())
    cache = cache[cache["ok"] == 1].drop(columns=["ok"])
    feats = work.merge(cache, on="movie_id", how="inner").reset_index(drop=True)
    feats["mid_int"] = [id_to_int(m) for m in feats["movie_id"]]
    note("FEATURES: " + str(len(feats)) + " posters: 15 HSV + 10 CIELAB + 15 border-cropped features extracted; "
         + "unreadable images: " + str(n_fail) + ".")
    out = feats.drop(columns=["jpeg_path", "mid_int"])
    save_table(out, "R10_features_all_films", combine=False)
    return feats


# =============================================================================
# STAGE 2b: SUBSAMPLE SENSITIVITY (resize, grayscale weights, k)
# =============================================================================
def stage_sensitivity_subset(feats):
    sub = feats.sample(n=min(SENSITIVITY_N, len(feats)), random_state=RNG_SEED)
    res = Parallel(n_jobs=N_JOBS)(delayed(extract_sensitivity)(mid, path, THUMBNAIL_SIZE, RNG_SEED)
                                  for mid, path in zip(sub["movie_id"], sub["jpeg_path"]))
    res = [r for r in res if r is not None]
    s = pd.DataFrame(res).merge(sub, on="movie_id", how="inner")
    rows = []
    for c in FEATURE_COLS + LAB_ONLY_COLS:
        rows.append({"check": "resize 300x300 vs 150x150", "feature": c,
                     "pearson_r": float(np.corrcoef(s[c], s["s300_" + c])[0, 1]),
                     "spearman_rho": safe_spearman(s[c], s["s300_" + c]),
                     "mean_abs_difference": float(np.mean(np.abs(s[c] - s["s300_" + c])))})
    for c in ["entropy", "contrast"]:
        rows.append({"check": "grayscale BT.709 (skimage rgb2gray weights) vs BT.601", "feature": c,
                     "pearson_r": float(np.corrcoef(s[c], s["bt709_" + c])[0, 1]),
                     "spearman_rho": safe_spearman(s[c], s["bt709_" + c]),
                     "mean_abs_difference": float(np.mean(np.abs(s[c] - s["bt709_" + c])))})
    for c in ["dom1_hue_sin", "dom1_hue_cos", "dom1_saturation", "dom1_value", "dom1_proportion"]:
        rows.append({"check": "k-means k=4 vs k=5", "feature": c,
                     "pearson_r": float(np.corrcoef(s[c], s["k4_" + c])[0, 1]),
                     "spearman_rho": safe_spearman(s[c], s["k4_" + c]),
                     "mean_abs_difference": float(np.mean(np.abs(s[c] - s["k4_" + c])))})
    t = pd.DataFrame(rows)
    t.insert(0, "n_posters", len(s))
    save_table(t, "R12_sensitivity_resize_grayscale_k")
    for chk, g in t.groupby("check"):
        note("SENSITIVITY (" + chk + ", n=" + str(len(s)) + "): Pearson r range "
             + fmt(g["pearson_r"].min()) + " - " + fmt(g["pearson_r"].max()) + ".")


# =============================================================================
# STAGE 3: 10% BORDER-CROP CHECK
# =============================================================================
def stage_border_crop(feats):
    rows = []
    for c in FEATURE_COLS:
        a, b = feats[c].values, feats["crop_" + c].values
        rows.append({"feature": c, "n_posters": len(feats),
                     "pearson_r": float(np.corrcoef(a, b)[0, 1]),
                     "spearman_rho": safe_spearman(a, b),
                     "mean_full": float(a.mean()), "mean_cropped": float(b.mean()),
                     "mean_abs_difference": float(np.mean(np.abs(a - b)))})
    t = pd.DataFrame(rows).sort_values("pearson_r")
    save_table(t, "R13_border_crop_feature_agreement")
    note("BORDER CROP (10%): Pearson r between full-poster and cropped features: min " + fmt(t["pearson_r"].min())
         + " (" + str(t.iloc[0]["feature"]) + "), median " + fmt(t["pearson_r"].median()) + ", max " + fmt(t["pearson_r"].max()) + ".")
    plt.figure(figsize=(8, 6))
    plt.barh(t["feature"], t["pearson_r"], color="steelblue")
    plt.xlim(0, 1)
    plt.xlabel("Pearson r between full-poster and 10%-border-cropped features")
    plt.title("Robustness of colour features to cropping poster borders (n = " + str(len(feats)) + ")")
    save_fig("R_fig03_border_crop_agreement")


def stage_border_crop_reanalysis(feats, Y, genres_ml, resB):
    """Repeat the main analysis (B) with border-cropped posters: do the results depend on frames or credit blocks?"""
    fc = feats.copy()
    for c in FEATURE_COLS:
        fc[c] = fc["crop_" + c]
    resK = analyze_multilabel("RB_CROP", "Multi-label (posters cropped by 10% on each edge)", fc, Y, genres_ml, FEATURE_COLS,
                              do_perm=False, do_shap=False)
    if resB is not None:
        rf_, rc_ = resB["effect"].rank(ascending=False, method="min"), resK["effect"].rank(ascending=False, method="min")
        t = pd.DataFrame({"feature": FEATURE_COLS,
                          "R2_full_poster": [float(resB["effect"][f]) for f in FEATURE_COLS],
                          "rank_full_poster": [int(rf_[f]) for f in FEATURE_COLS],
                          "R2_border_cropped": [float(resK["effect"][f]) for f in FEATURE_COLS],
                          "rank_border_cropped": [int(rc_[f]) for f in FEATURE_COLS]}).sort_values("rank_full_poster")
        save_table(t, "R14_border_crop_effect_sizes_full_vs_cropped")
        perf = pd.DataFrame([{"metric": k, "full_poster": resB["perf"][k], "border_cropped": resK["perf"][k]}
                             for k in ["macro_auc", "macro_f1", "micro_f1"]]
                            + [{"metric": "pillai_trace", "full_poster": resB["pillai"], "border_cropped": resK["pillai"]}])
        save_table(perf, "R15_border_crop_classification_full_vs_cropped")
        note("BORDER CROP -- MAIN ANALYSIS REPEATED: RF macro-AUC full poster " + fmt(resB["perf"]["macro_auc"]) + " -> cropped "
             + fmt(resK["perf"]["macro_auc"]) + "; feature-ranking Spearman rho = "
             + fmt(safe_spearman(t["R2_full_poster"], t["R2_border_cropped"])) + ".")
    return resK


# =============================================================================
# STAGE 4: GENRE DEFINITIONS
# =============================================================================
def stratified_cap(df, label_col, cap, rng):
    parts = []
    for g in sorted(df[label_col].unique()):
        sub = df[df[label_col] == g]
        if len(sub) > cap:
            sub = sub.iloc[np.sort(rng.choice(len(sub), size=cap, replace=False))]
        parts.append(sub)
    return pd.concat(parts).reset_index(drop=True)


def load_original_sample():
    """Read the feature file of the original first-listed-label sample, if present."""
    path = next((p for p in ORIGINAL_FEATURES_CANDIDATES if os.path.isfile(p)), None)
    if path is None:
        return None
    try:
        orig = pd.read_excel(path)
    except Exception as e:
        print("  [WARNING] " + path + " could not be read: " + str(e))
        return None
    need = set(FEATURE_COLS + ["movie_id", "primary_genre"])
    if not need.issubset(set(orig.columns)):
        print("  [WARNING] " + path + " does not contain the expected columns.")
        return None
    orig = orig.copy()
    orig["label"] = orig["primary_genre"].astype(str)
    orig["mid_int"] = [id_to_int(m) for m in orig["movie_id"]]
    return orig, path


def stage_original_exact():
    """Exact reproduction of the original first-listed-label analysis: same sample, same stored feature
    values, same row order, same model and fold settings. Yields per-genre precision / recall / F1 /
    support and the class-specific SHAP values of that model. Needs no poster files."""
    loaded = load_original_sample()
    if loaded is None:
        print("Feature file of the original first-listed sample not found; stage skipped.")
        return None
    orig, path = loaded
    alm = None
    if "all_genres" in orig.columns:
        import ast
        alm = {}
        for mid, txt in zip(orig["movie_id"], orig["all_genres"]):
            try:
                alm[mid] = set(ast.literal_eval(str(txt)))
            except Exception:
                alm[mid] = set()
    res = analyze_single_label("RA0", "Exact reproduction of the original first-listed analysis (" + os.path.basename(path) + ")",
                               orig, FEATURE_COLS, all_label_map=alm, do_perm=False, do_shap=True)
    stage_imbalance_checks(orig, FEATURE_COLS, tag="RA0")
    return res


def stage_definitions(feats):
    defs = {}
    rng = np.random.default_rng(RNG_SEED)

    # ---- B: multi-label (all films; genres carried by at least MIN_FILMS_PER_GENRE_ML films) ----
    all_counts = collections.Counter(itertools.chain.from_iterable(feats["genres"]))
    genres_ml = [g for g, c in sorted(all_counts.items(), key=lambda kv: -kv[1]) if c >= MIN_FILMS_PER_GENRE_ML]
    if len(genres_ml) < 3:
        raise RuntimeError("Not enough genres for the multi-label analysis (threshold: " + str(MIN_FILMS_PER_GENRE_ML) + ").")
    Y = np.array([[1 if g in set(gs) else 0 for g in genres_ml] for gs in feats["genres"]], dtype=int)
    defs["genres_ml"] = genres_ml
    defs["Y"] = Y
    note("DEFINITION B (multi-label, MAIN ANALYSIS): " + str(len(feats)) + " films, " + str(len(genres_ml))
         + " genres (each in >= " + str(MIN_FILMS_PER_GENRE_ML) + " films): " + ", ".join(genres_ml))
    save_table(pd.DataFrame({"genre": genres_ml, "n_films": Y.sum(axis=0), "prevalence": Y.mean(axis=0)}),
               "R20_definitionB_multilabel_genres")

    # ---- A: first-listed label ----
    dfA, srcA = None, ""
    loaded = load_original_sample() if DATA_SOURCE == "official" else None
    if loaded is not None:
        orig, orig_path = loaded
        try:
            lab_map = dict(zip(orig["mid_int"], orig["label"]))
            cand = feats[feats["mid_int"].isin(lab_map)].copy()
            cand["label"] = cand["mid_int"].map(lab_map)
            share = len(cand) / max(len(orig), 1)
            enough = share >= 0.95 or (QUICK_TEST_MODE and len(cand) >= 300)
            if enough:
                dfA = cand
                srcA = ("IDENTICAL to the original first-listed sample (" + orig_path + "; " + str(len(dfA)) + " / " + str(len(orig))
                        + " films matched)")
                # agreement of stored and re-extracted feature values (reproducibility check)
                m = dfA.merge(orig[["mid_int"] + FEATURE_COLS], on="mid_int", suffixes=("", "_orig"))
                rr = []
                for c in FEATURE_COLS:
                    rr.append({"feature": c, "n_films": len(m),
                               "pearson_r_new_vs_original": float(np.corrcoef(m[c], m[c + "_orig"])[0, 1]),
                               "mean_abs_difference": float(np.mean(np.abs(m[c] - m[c + "_orig"])))})
                save_table(pd.DataFrame(rr), "R21_definitionA_feature_reproduction_check")
            else:
                print("  [WARNING] Only " + fmt(100 * share, 1)
                      + "% of the films in the original sample were found in the dataset; definition A will be re-sampled.")
        except Exception as e:
            print("  [WARNING] Original sample could not be matched (" + str(e) + "); definition A will be re-sampled.")
            dfA = None
    if dfA is None:
        top = feats["first_label"].value_counts().head(K_GENRES_ORIGINAL).index.tolist()
        dfA = feats[feats["first_label"].isin(top)].copy()
        dfA["label"] = dfA["first_label"]
        dfA = stratified_cap(dfA, "label", TARGET_N_PER_GENRE, rng)
        srcA = "re-sampled with the original first-listed RULE (original sample file not found)"
    defs["A"] = dfA.reset_index(drop=True)
    note("DEFINITION A (first-listed label): " + str(len(dfA)) + " films -- " + srcA)

    # ---- C: single-label films only ----
    single = feats[feats["n_labels"] == 1].copy()
    single["label"] = single["first_label"]
    vc = single["label"].value_counts()
    keep = vc[vc >= MIN_N_SINGLE].index.tolist()
    dfC = stratified_cap(single[single["label"].isin(keep)], "label", TARGET_N_PER_GENRE, rng)
    defs["C"] = dfC
    note("DEFINITION C (single-label films only): " + str(len(dfC)) + " films, " + str(len(keep)) + " genres: "
         + ", ".join(str(g) + "=" + str(int(n)) for g, n in dfC["label"].value_counts().items()))

    # ---- D2: rarest-label rule ----
    setml = set(genres_ml)
    cnt = {g: all_counts[g] for g in genres_ml}
    lists = [[g for g in gs if g in setml] for gs in feats["genres"]]
    has = np.array([len(l) > 0 for l in lists])
    defs["ml_lists"] = lists
    defs["ml_has"] = has
    rare = [min(l, key=lambda g: (cnt[g], g)) if l else None for l in lists]
    dfD2 = feats[has].copy()
    dfD2["label"] = [r for r in rare if r is not None]
    vc = dfD2["label"].value_counts()
    dfD2 = stratified_cap(dfD2[dfD2["label"].isin(vc[vc >= MIN_N_SINGLE].index)], "label", TARGET_N_PER_GENRE, rng)
    defs["D2"] = dfD2
    note("DEFINITION D2 (rarest label of each film): " + str(len(dfD2)) + " films, " + str(dfD2["label"].nunique()) + " genres.")

    inv = []
    for tag in ["A", "C", "D2"]:
        d = defs[tag][["movie_id", "title", "year", "genres", "label"]].copy()
        d.insert(0, "definition", tag)
        inv.append(d)
    save_table(pd.concat(inv), "R22_sample_inventory_single_label_definitions", combine=False)
    comp = pd.DataFrame({tag: defs[tag]["label"].value_counts() for tag in ["A", "C", "D2"]}).fillna(0).astype(int)
    save_table(comp.rename_axis("genre").reset_index(), "R23_sample_sizes_by_definition")
    defs["all_label_map"] = dict(zip(feats["movie_id"], [set(g) for g in feats["genres"]]))
    return defs


# =============================================================================
# SHAP HELPERS
# =============================================================================
def shap_array(model, X):
    """Return SHAP values as an (n, p, c) array; supports old (list) and new (3-D array) shap versions."""
    import shap
    explainer = shap.TreeExplainer(model)
    try:
        sv = explainer.shap_values(X, check_additivity=False)
    except TypeError:
        sv = explainer.shap_values(X)
    if isinstance(sv, list):
        arr = np.stack([np.asarray(a) for a in sv], axis=-1)
    else:
        arr = np.asarray(sv)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        elif arr.ndim == 3 and arr.shape[1] != X.shape[1] and arr.shape[2] == X.shape[1]:
            arr = np.transpose(arr, (0, 2, 1))
    return arr


def summarize_shap(tag, arr, Xs, classes, feature_cols, focus):
    """Class-specific, signed SHAP summaries and figures. arr: (n, p, c)."""
    n, p, c = arr.shape
    mean_abs = np.abs(arr).mean(axis=0)                       # (p, c)
    rows = []
    rho_m = np.zeros((p, c))
    for ci, g in enumerate(classes):
        for j, f in enumerate(feature_cols):
            x, s = Xs[:, j], arr[:, j, ci]
            rho = safe_spearman(x, s)
            rho_m[j, ci] = rho
            q1, q3 = np.quantile(x, [0.25, 0.75])
            lo = float(s[x <= q1].mean()) if np.any(x <= q1) else np.nan
            hi = float(s[x >= q3].mean()) if np.any(x >= q3) else np.nan
            rows.append({"genre": g, "feature": f, "mean_abs_shap": float(mean_abs[j, ci]),
                         "direction_spearman_rho(feature,shap)": rho,
                         "mean_shap_bottom_quartile_of_feature": lo,
                         "mean_shap_top_quartile_of_feature": hi,
                         "direction": "higher feature -> HIGHER probability of genre" if rho > 0.1 else
                                      ("higher feature -> LOWER probability of genre" if rho < -0.1 else "no clear monotone direction"),
                         "rank_within_genre": 0})
    long = pd.DataFrame(rows)
    long["rank_within_genre"] = long.groupby("genre")["mean_abs_shap"].rank(ascending=False, method="min").astype(int)
    save_table(long.sort_values(["genre", "rank_within_genre"]), tag + "_shap_by_genre_signed")

    glob_imp = pd.Series(mean_abs.mean(axis=1), index=feature_cols).sort_values(ascending=False)
    save_table(glob_imp.rename("mean_abs_shap").rename_axis("feature").reset_index(), tag + "_shap_global")

    signed = np.sign(rho_m) * mean_abs
    save_table(pd.DataFrame(signed.T, index=classes, columns=feature_cols).rename_axis("genre").reset_index(),
               tag + "_shap_signed_matrix")

    # family level (for correlated features): mean |sum of SHAP values within a family|
    fams = []
    for f in feature_cols:
        if FEATURE_FAMILY.get(f, f) not in fams:
            fams.append(FEATURE_FAMILY.get(f, f))
    fam_rows = []
    for fam in fams:
        idx = [j for j, f in enumerate(feature_cols) if FEATURE_FAMILY.get(f, f) == fam]
        grouped = np.abs(arr[:, idx, :].sum(axis=1)).mean(axis=0)     # (c,)
        row = {"feature_family": fam, "n_features": len(idx), "all_genres_mean": float(grouped.mean())}
        for ci, g in enumerate(classes):
            row[g] = float(grouped[ci])
        fam_rows.append(row)
    fam_df = pd.DataFrame(fam_rows).sort_values("all_genres_mean", ascending=False)
    save_table(fam_df, tag + "_shap_feature_family")

    # figure 1: global importance
    gi = glob_imp.sort_values()
    plt.figure(figsize=(8, 6))
    plt.barh(gi.index, gi.values, color="steelblue")
    plt.xlabel("Mean |SHAP value| (change in predicted genre probability), averaged over genres")
    plt.title("Global SHAP feature importance (" + tag + ")")
    save_fig(tag + "_fig_shap_global")

    # figure 2: class-specific signed heat map
    vmax = float(np.abs(signed).max()) or 1.0
    plt.figure(figsize=(max(9, 0.7 * p + 3), max(4, 0.42 * c + 2.5)))
    im = plt.imshow(signed.T, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.xticks(range(p), feature_cols, rotation=60, ha="right")
    plt.yticks(range(c), classes)
    plt.colorbar(im, label="Mean |SHAP|, signed by direction\n(red: higher feature value raises genre probability)")
    plt.title("Class-specific, signed SHAP contributions (" + tag + ")")
    save_fig(tag + "_fig_shap_by_genre_signed")

    # figure 3: dependence plots for the focus genres
    for g in [g for g in focus if g in classes]:
        ci = list(classes).index(g)
        top = np.argsort(-mean_abs[:, ci])[:3]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
        for ax, j in zip(axes, top):
            x, s = Xs[:, j], arr[:, j, ci]
            ax.scatter(x, s, s=7, alpha=0.3, color="#1f77b4", edgecolors="none")
            try:
                edges = np.unique(np.quantile(x, np.linspace(0, 1, 11)))
                if len(edges) > 2:
                    mids, meds = [], []
                    for a, b in zip(edges[:-1], edges[1:]):
                        msk = (x >= a) & (x <= b)
                        if msk.sum() > 0:
                            mids.append(np.median(x[msk]))
                            meds.append(np.median(s[msk]))
                    ax.plot(mids, meds, color="#d62728", linewidth=2.2, label="Binned median")
                    ax.legend(fontsize=8)
            except Exception:
                pass
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xlabel(feature_cols[j])
            ax.set_ylabel("SHAP value for " + g)
        fig.suptitle("SHAP dependence: three most influential features for " + g + " (" + tag + ")")
        save_fig(tag + "_fig_shap_dependence_" + g.replace(" ", "_").replace("-", ""))
    return {"global": glob_imp, "long": long, "family": fam_df}


def shap_single_label(tag, X, y, genres, feature_cols, n_explain):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=RNG_SEED)
    rf = RandomForestClassifier(n_estimators=RF_TREES, random_state=RNG_SEED, n_jobs=N_JOBS).fit(Xtr, ytr)
    Xs = Xte
    if len(Xte) > n_explain + len(genres):
        try:
            Xs, _, _, _ = train_test_split(Xte, yte, train_size=n_explain, stratify=yte, random_state=RNG_SEED)
        except Exception:
            Xs = Xte[np.random.default_rng(RNG_SEED).choice(len(Xte), n_explain, replace=False)]
    try:
        t0 = time.time()
        print("  Computing SHAP (" + str(len(Xs)) + " test observations, " + str(len(genres)) + " classes)...")
        arr = shap_array(rf, Xs)
        order = [list(rf.classes_).index(g) for g in genres]
        arr = arr[:, :, order]
        print("  SHAP done (" + "{:.1f}".format((time.time() - t0) / 60) + " min)")
        return summarize_shap(tag, arr, Xs, genres, feature_cols, FOCUS_GENRES)
    except Exception as e:
        print("  [WARNING] SHAP unavailable (" + str(e) + "); falling back to permutation importance (NO class-specific results).")
        pi = permutation_importance(rf, Xte, yte, n_repeats=30, random_state=RNG_SEED, scoring="f1_macro")
        imp = pd.Series(pi.importances_mean, index=feature_cols).sort_values(ascending=False)
        save_table(imp.rename("permutation_importance").rename_axis("feature").reset_index(),
                   tag + "_permutation_importance_FALLBACK")
        return {"global": None, "long": None, "family": None}


# =============================================================================
# SINGLE-LABEL ANALYSIS (definitions A, C, D2; HSV and CIELAB feature sets)
# =============================================================================
def make_models(n_jobs):
    return {
        "RandomForest": RandomForestClassifier(n_estimators=RF_TREES, random_state=RNG_SEED, n_jobs=n_jobs),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=RNG_SEED),
        # RBF kernels are scale-sensitive, so the SVM is fitted to standardized features;
        # the unscaled variant is kept for comparison.
        "SVM_RBF_standardized": make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=RNG_SEED)),
        "SVM_RBF_unscaled_as_submitted": SVC(kernel="rbf", random_state=RNG_SEED),
    }


def plot_confusion(cmn, genres, title, name):
    k = len(genres)
    plt.figure(figsize=(max(7, 0.62 * k + 2), max(6, 0.55 * k + 2)))
    plt.imshow(cmn, cmap="Blues", vmin=0, vmax=max(0.5, float(np.nanmax(cmn))))
    plt.xticks(range(k), genres, rotation=45, ha="right")
    plt.yticks(range(k), genres)
    plt.xlabel("Predicted genre")
    plt.ylabel("True genre")
    plt.title(title)
    for i in range(k):
        for j in range(k):
            plt.text(j, i, "{:.0f}".format(100 * cmn[i, j]), ha="center", va="center", fontsize=8,
                     color="white" if cmn[i, j] > 0.3 else "black")
    plt.colorbar(label="Row-normalised proportion (cell labels: % of true genre)")
    save_fig(name)


def analyze_single_label(tag, title, df, feature_cols, all_label_map=None, do_perm=False,
                         do_shap=True, shap_n=SHAP_N_EXPLAIN):
    genres = df["label"].value_counts().index.tolist()
    X = df[feature_cols].values.astype(float)
    y = labels_of(df)
    k = len(genres)
    chance = 1.0 / k
    res = {"kind": "single", "tag": tag, "genres": genres, "n": len(df), "feature_cols": feature_cols}
    print(title + ": n=" + str(len(df)) + ", k=" + str(k))

    # 1) descriptives
    g = df.groupby("label")[feature_cols]
    desc = g.mean().add_suffix("_mean").join(g.std().add_suffix("_sd"))
    desc.insert(0, "n", df["label"].value_counts())
    save_table(desc.loc[genres].rename_axis("genre").reset_index(), tag + "_descriptives_by_genre")

    # 2) omnibus (permutation Pillai) + eta^2
    D = (y[:, None] == np.array(genres)[None, :]).astype(float)
    pillai, p_pillai, eta2, p_eta = mv_permutation_test(X, D, N_PERM_STATS, RNG_SEED)
    save_table(pd.DataFrame([{"statistic": "Pillai's trace", "observed_value": pillai, "n_permutations": N_PERM_STATS,
                              "p_value": p_pillai, "n_films": len(df), "k_genres": k}]), tag + "_manova_permutation")

    # 3) univariate
    urows = []
    for j, f in enumerate(feature_cols):
        groups = [X[y == gg, j] for gg in genres]
        try:
            Hs, pk = stats.kruskal(*groups)
        except Exception:
            Hs, pk = np.nan, 1.0
        Fs, pf = stats.f_oneway(*groups)
        urows.append({"feature": f, "kruskal_H": Hs, "p_kruskal": pk, "F": Fs, "p_anova": pf,
                      "eta_squared": float(eta2[j]), "p_eta_squared_permutation": float(p_eta[j])})
    uni = pd.DataFrame(urows)
    uni["p_fdr_bh"] = bh_fdr(uni["p_kruskal"].fillna(1.0).values)
    uni["significant_fdr_05"] = uni["p_fdr_bh"] < 0.05
    uni = uni.sort_values("eta_squared", ascending=False).reset_index(drop=True)
    uni["rank"] = np.arange(1, len(uni) + 1)
    save_table(uni, tag + "_univariate_tests_fdr")
    res["effect"] = uni.set_index("feature")["eta_squared"]
    res["pillai"], res["p_pillai"] = pillai, p_pillai

    pu = uni.sort_values("eta_squared")
    plt.figure(figsize=(8, 6))
    plt.barh(pu["feature"], pu["eta_squared"], color=["#d62728" if s else "#c7c7c7" for s in pu["significant_fdr_05"]])
    plt.xlabel("Effect size (eta squared)")
    plt.title("Genre effect size per colour feature (" + tag + ")\n(red = significant after FDR correction)", fontsize=12)
    save_fig(tag + "_fig_effect_sizes")

    # 4) classifiers
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG_SEED)
    models = make_models(N_JOBS)
    crow = []
    for name, model in models.items():
        t0 = time.time()
        sc = cross_validate(model, X, y, cv=cv, scoring=["f1_macro", "balanced_accuracy"])
        crow.append({"model": name, "macro_f1_mean": sc["test_f1_macro"].mean(), "macro_f1_sd": sc["test_f1_macro"].std(),
                     "balanced_accuracy_mean": sc["test_balanced_accuracy"].mean(),
                     "balanced_accuracy_sd": sc["test_balanced_accuracy"].std(), "chance_level": chance})
        print("  " + name + ": macro-F1=" + fmt(crow[-1]["macro_f1_mean"]) + " (" + "{:.0f}".format(time.time() - t0) + " s)")
    clf = pd.DataFrame(crow).sort_values("macro_f1_mean", ascending=False).reset_index(drop=True)
    save_table(clf, tag + "_classification_results")
    # Primary model = Random Forest (per-genre table, confusion matrix, permutation test and SHAP
    # all use this model); the other models serve as comparisons.
    best_name = "RandomForest"
    ci = clf.set_index("model")
    res["perf"] = {"rf_macro_f1": float(ci.loc["RandomForest", "macro_f1_mean"]),
                   "rf_balanced_accuracy": float(ci.loc["RandomForest", "balanced_accuracy_mean"]),
                   "top_model": str(clf.iloc[0]["model"]), "top_model_macro_f1": float(clf.iloc[0]["macro_f1_mean"]),
                   "chance": chance}

    # 5) per-genre table + row-normalised confusion matrix -- Random Forest
    rf = models["RandomForest"]
    y_pred = cross_val_predict(rf, X, y, cv=cv)
    rep = classification_report(y, y_pred, labels=genres, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, y_pred, labels=genres)
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    prow = []
    for i, gg in enumerate(genres):
        r = rep[gg]
        off = cmn[i].copy()
        off[i] = -1
        j = int(np.argmax(off))
        prow.append({"genre": gg, "precision": r["precision"], "recall": r["recall"], "f1": r["f1-score"],
                     "support": int(r["support"]), "n_predicted_as_genre": int(cm[:, i].sum()),
                     "most_frequent_confusion": genres[j], "share_confused_with_it": float(cmn[i, j])})
    per = pd.DataFrame(prow)
    tot = pd.DataFrame([{"genre": "MACRO AVERAGE", "precision": rep["macro avg"]["precision"],
                         "recall": rep["macro avg"]["recall"], "f1": rep["macro avg"]["f1-score"],
                         "support": len(y), "n_predicted_as_genre": len(y)}])
    save_table(pd.concat([per, tot], ignore_index=True), tag + "_per_genre_precision_recall_f1")
    save_table(pd.DataFrame(cm, index=genres, columns=genres).rename_axis("true_genre").reset_index(),
               tag + "_confusion_matrix_counts")
    save_table(pd.DataFrame(cmn, index=genres, columns=genres).rename_axis("true_genre").reset_index(),
               tag + "_confusion_matrix_row_normalised")
    plot_confusion(cmn, genres, "Row-normalised confusion matrix (Random Forest, " + tag + ")", tag + "_fig_confusion_normalised")
    ps = per.sort_values("f1")
    plt.figure(figsize=(8, max(4, 0.4 * k + 1.5)))
    plt.barh(ps["genre"], ps["f1"], color="steelblue", label="F1")
    plt.scatter(ps["recall"], ps["genre"], color="#d62728", zorder=3, label="Recall")
    plt.scatter(ps["precision"], ps["genre"], color="black", marker="x", zorder=3, label="Precision")
    plt.axvline(chance, color="gray", linestyle="--", label="Chance (1/k)")
    plt.xlabel("Score")
    plt.title("Per-genre classification performance (Random Forest, " + tag + ")")
    plt.legend(fontsize=8)
    save_fig(tag + "_fig_per_genre_performance")
    res["per_genre"] = per
    extra = {"strict_accuracy": float(np.mean(y_pred == y))}
    if all_label_map is not None:
        hit = [p in all_label_map.get(m, set()) for p, m in zip(y_pred, df["movie_id"])]
        extra["any_label_accuracy"] = float(np.mean(hit))
        extra["note"] = "any_label_accuracy: prediction counted correct if it matches ANY of the film's IMDb genre labels"
    save_table(pd.DataFrame([extra]), tag + "_strict_vs_any_label_accuracy")
    res["extra"] = extra

    # 6) label-permutation test vs. chance
    if do_perm:
        print("  Label-permutation test (" + str(N_PERM_CLASSIFIER) + " permutations) -- this can take a while...")
        best = clone(models[best_name])
        try:
            best.set_params(n_jobs=1)
        except Exception:
            pass
        obs, perm_scores, pval = permutation_test_score(best, X, y, cv=cv, scoring="f1_macro",
                                                        n_permutations=N_PERM_CLASSIFIER, random_state=RNG_SEED,
                                                        n_jobs=N_JOBS, verbose=5)
        save_table(pd.DataFrame([{"model": best_name, "observed_macro_f1": obs, "n_permutations": N_PERM_CLASSIFIER,
                                  "p_value": pval, "chance_level": chance,
                                  "null_mean": float(np.mean(perm_scores)), "null_max": float(np.max(perm_scores))}]),
                   tag + "_permutation_test_classifier")
        plt.figure(figsize=(7, 4.5))
        plt.hist(perm_scores, bins=20, color="lightgray", edgecolor="black", label="Null distribution (permuted labels)")
        plt.axvline(obs, color="red", linewidth=2, label="Observed macro-F1 = " + fmt(obs))
        plt.axvline(chance, color="blue", linestyle="--", linewidth=1.5, label="Chance level = " + fmt(chance))
        plt.xlabel("Macro-F1 score")
        plt.ylabel("Frequency")
        plt.title("Permutation test vs. chance (" + best_name + ", " + tag + ", p = " + fmt(pval, 4) + ")")
        plt.legend(fontsize=9)
        save_fig(tag + "_fig_permutation_test")
        res["perf"]["p_perm"] = float(pval)

    # 7) SHAP
    res["shap_global"] = None
    if do_shap:
        sh = shap_single_label(tag, X, y, genres, feature_cols, shap_n)
        res["shap_global"] = sh["global"]
        res["shap_long"] = sh["long"]

    top3 = ", ".join(uni["feature"].head(3).tolist())
    best_g = per.sort_values("f1", ascending=False)["genre"].head(2).tolist()
    worst_g = per.sort_values("f1")["genre"].head(2).tolist()
    note("[" + tag + "] " + title + ": n=" + str(len(df)) + ", k=" + str(k) + " | Pillai=" + fmt(pillai) + ", p=" + fmt(p_pillai)
         + " | largest eta^2: " + top3 + " | RF macro-F1=" + fmt(res["perf"]["rf_macro_f1"]) + ", balanced accuracy=" + fmt(res["perf"]["rf_balanced_accuracy"])
         + " (top model: " + res["perf"]["top_model"] + " " + fmt(res["perf"]["top_model_macro_f1"]) + ")"
         + " (chance " + fmt(chance) + ")" + (", p=" + fmt(res["perf"]["p_perm"]) if "p_perm" in res["perf"] else "")
         + " | best genres: " + ", ".join(best_g) + "; weakest: " + ", ".join(worst_g) + ".")
    return res


def stage_imbalance_checks(dfA, feature_cols, tag="RA"):
    """Is the weak performance of the smallest genre a matter of sample size? (class weights; equal n per genre)"""
    genres = dfA["label"].value_counts().index.tolist()
    X, y = dfA[feature_cols].values.astype(float), labels_of(dfA)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG_SEED)

    def per_genre(model, Xa, ya):
        yp = cross_val_predict(model, Xa, ya, cv=cv)
        rep = classification_report(ya, yp, labels=genres, output_dict=True, zero_division=0)
        return {g: (rep[g]["recall"], rep[g]["f1-score"]) for g in genres}, rep["macro avg"]["f1-score"]

    base, mf_base = per_genre(RandomForestClassifier(n_estimators=RF_TREES, random_state=RNG_SEED, n_jobs=N_JOBS), X, y)
    bal, mf_bal = per_genre(RandomForestClassifier(n_estimators=RF_TREES, random_state=RNG_SEED, n_jobs=N_JOBS,
                                                   class_weight="balanced"), X, y)
    n_min = int(dfA["label"].value_counts().min())
    rng = np.random.default_rng(RNG_SEED)
    eq = stratified_cap(dfA, "label", n_min, rng)
    eqr, mf_eq = per_genre(RandomForestClassifier(n_estimators=RF_TREES, random_state=RNG_SEED, n_jobs=N_JOBS),
                           eq[feature_cols].values.astype(float), labels_of(eq))
    rows = []
    for g in genres:
        rows.append({"genre": g, "n_in_sample": int((y == g).sum()),
                     "recall_baseline": base[g][0], "f1_baseline": base[g][1],
                     "recall_class_weight_balanced": bal[g][0], "f1_class_weight_balanced": bal[g][1],
                     "recall_equal_n_per_genre": eqr[g][0], "f1_equal_n_per_genre": eqr[g][1],
                     "equal_n_used": n_min})
    rows.append({"genre": "MACRO-F1", "n_in_sample": len(y), "f1_baseline": mf_base,
                 "f1_class_weight_balanced": mf_bal, "f1_equal_n_per_genre": mf_eq, "equal_n_used": n_min})
    save_table(pd.DataFrame(rows), tag + "_imbalance_sensitivity_per_genre")
    small = dfA["label"].value_counts().idxmin()
    note("[" + tag + "] IMBALANCE CHECK: smallest genre " + str(small) + " (n=" + str(n_min) + ") recall: baseline "
         + fmt(base[small][0]) + ", class-weighted " + fmt(bal[small][0]) + ", equal-n " + fmt(eqr[small][0]) + ".")


def stage_random_label(feats, defs, feature_cols):
    """D1: each film receives one of its labels at random (order-independent), repeated N times."""
    lists, has = defs["ml_lists"], defs["ml_has"]
    base = feats[has].reset_index(drop=True)
    lists = [l for l in lists if l]
    lens = np.array([len(l) for l in lists])
    rng = np.random.default_rng(RNG_SEED + 1)
    eta_all, pil, clf_rows = [], [], []
    for r in range(N_RANDOM_LABEL_DRAWS):
        pick = np.floor(rng.random(len(lists)) * lens).astype(int)
        d = base.copy()
        d["label"] = [l[i] for l, i in zip(lists, pick)]
        d = stratified_cap(d, "label", TARGET_N_PER_GENRE, rng)
        genres = sorted(d["label"].unique())
        X, y = d[feature_cols].values.astype(float), labels_of(d)
        D = (y[:, None] == np.array(genres)[None, :]).astype(float)
        Xc, Q, T, Tinv = mv_prepare(X, D)
        pl, e2 = mv_stats(Xc, Q, T, Tinv)
        eta_all.append(e2)
        pil.append(pl)
        if r < N_RANDOM_LABEL_CLF_DRAWS:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG_SEED)
            sc = cross_validate(RandomForestClassifier(n_estimators=RF_TREES, random_state=RNG_SEED, n_jobs=N_JOBS),
                                X, y, cv=cv, scoring=["f1_macro", "balanced_accuracy"])
            clf_rows.append({"draw": r, "n_films": len(d), "k_genres": len(genres), "chance": 1.0 / len(genres),
                             "macro_f1": sc["test_f1_macro"].mean(), "balanced_accuracy": sc["test_balanced_accuracy"].mean()})
        print("  random-label draw " + str(r + 1) + " / " + str(N_RANDOM_LABEL_DRAWS))
    E = np.array(eta_all)
    ranks = np.array([stats.rankdata(-row, method="min") for row in E])
    t = pd.DataFrame({"feature": feature_cols, "eta_squared_mean": E.mean(axis=0), "eta_squared_sd": E.std(axis=0),
                      "rank_mean": ranks.mean(axis=0), "rank_min": ranks.min(axis=0), "rank_max": ranks.max(axis=0)})
    t = t.sort_values("eta_squared_mean", ascending=False)
    t.insert(0, "n_draws", N_RANDOM_LABEL_DRAWS)
    save_table(t, "RD1_random_label_rule_effect_sizes")
    c = pd.DataFrame(clf_rows)
    save_table(c, "RD1_random_label_rule_classification")
    note("[D1] RANDOM-LABEL RULE (" + str(N_RANDOM_LABEL_DRAWS) + " draws): mean Pillai " + fmt(np.mean(pil))
         + "; largest eta^2: " + ", ".join(t["feature"].head(3).tolist()) + "; mean RF macro-F1 "
         + fmt(c["macro_f1"].mean()) + " (chance " + fmt(c["chance"].mean()) + ").")
    return {"kind": "single", "tag": "D1", "effect": t.set_index("feature")["eta_squared_mean"], "shap_global": None,
            "perf": {"rf_macro_f1": float(c["macro_f1"].mean()), "chance": float(c["chance"].mean())}}


# =============================================================================
# MULTI-LABEL ANALYSIS (definition B -- MAIN ANALYSIS; independent of label order)
# =============================================================================
def ml_rf(n_jobs):
    return RandomForestClassifier(n_estimators=ML_RF_TREES, max_depth=ML_RF_MAX_DEPTH, min_samples_leaf=ML_RF_MIN_LEAF,
                                  random_state=RNG_SEED, n_jobs=n_jobs)


def ovr_oof(model_name, X, Y):
    """Binary relevance (one model per genre), 5-fold; out-of-fold probabilities and thresholded predictions."""
    kf = KFold(n_splits=5, shuffle=True, random_state=RNG_SEED)
    oof = np.full(Y.shape, 0.5)
    pred = np.zeros(Y.shape, dtype=int)
    for tr, te in kf.split(X):
        for j in range(Y.shape[1]):
            ytr = Y[tr, j]
            if ytr.min() == ytr.max():
                continue
            if model_name == "RandomForest":
                m = ml_rf(N_JOBS)
            else:
                m = HistGradientBoostingClassifier(random_state=RNG_SEED)
            m.fit(X[tr], ytr)
            pr = m.predict_proba(X[te])[:, 1]
            oof[te, j] = pr
            pred[te, j] = (pr > ytr.mean()).astype(int)     # threshold = genre prevalence in the training fold
    return oof, pred


def ml_macro_auc_once(X, Y, perm_seed, permute, n_trees, max_depth, min_leaf, seed):
    rng = np.random.default_rng(perm_seed)
    Yp = Y[rng.permutation(len(Y))] if permute else Y
    kf = KFold(n_splits=3, shuffle=True, random_state=seed)
    oof = np.full(Y.shape, 0.5)
    for tr, te in kf.split(X):
        for j in range(Y.shape[1]):
            ytr = Yp[tr, j]
            if ytr.min() == ytr.max():
                continue
            m = RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth, min_samples_leaf=min_leaf,
                                       random_state=seed, n_jobs=1)
            m.fit(X[tr], ytr)
            oof[te, j] = m.predict_proba(X[te])[:, 1]
    aucs = [roc_auc_score(Yp[:, j], oof[:, j]) for j in range(Y.shape[1]) if Yp[:, j].min() != Yp[:, j].max()]
    return float(np.mean(aucs))


def profile_figures(tag, feats, Y, genres):
    """Descriptive profile figures for the multi-label definition (a film counts in each genre it carries)."""
    z = feats[FEATURE_COLS].copy()
    z = (z - z.mean()) / z.std()
    prof = pd.DataFrame([z[Y[:, j] == 1].mean() for j in range(len(genres))], index=genres)
    save_table(prof.rename_axis("genre").reset_index(), tag + "_standardised_genre_profiles")

    key = ["mean_saturation", "mean_value", "colorfulness", "warm_cool_logodds"]
    fig, axes = plt.subplots(2, 2, figsize=(max(11, 0.7 * len(genres) + 4), 9))
    for ax, f in zip(axes.ravel(), key):
        ax.boxplot([feats.loc[Y[:, j] == 1, f].values for j in range(len(genres))], showfliers=False)
        ax.set_xticks(range(1, len(genres) + 1))
        ax.set_xticklabels(genres, rotation=55, ha="right")
        ax.set_title(f.replace("_", " ").title())
    fig.suptitle("Distribution of key colour features by genre (all genre labels; films count in each of their genres)", fontsize=13)
    save_fig(tag + "_fig_boxplots_by_genre")

    fig, axes = plt.subplots(1, len(genres), figsize=(1.6 * len(genres), 2.0))
    for ax, j in zip(np.atleast_1d(axes), range(len(genres))):
        sub = feats[Y[:, j] == 1]
        h = (np.arctan2(sub["dom1_hue_sin"].mean(), sub["dom1_hue_cos"].mean()) / (2 * np.pi)) % 1.0
        sw = mcolors.hsv_to_rgb([h, float(np.clip(sub["dom1_saturation"].mean(), 0, 1)), float(np.clip(sub["dom1_value"].mean(), 0, 1))])
        ax.imshow(np.tile(sw, (10, 10, 1)))
        ax.set_title(genres[j], fontsize=9)
        ax.axis("off")
    fig.suptitle("Average dominant poster colour by genre", fontsize=13)
    save_fig(tag + "_fig_dominant_colour_by_genre")

    rfe = ["mean_saturation", "mean_value", "colorfulness", "warm_cool_logodds", "entropy", "contrast"]
    rlab = ["Saturation", "Value", "Colorfulness", "Warm-Cool", "Entropy", "Contrast"]
    ang = [i / float(len(rfe)) * 2 * np.pi for i in range(len(rfe))]
    ang = ang + ang[:1]
    ncols = 5
    nrows = int(np.ceil(len(genres) / ncols))
    vmax = float(np.nanmax(np.abs(prof[rfe].values))) * 1.15
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 5.9 * nrows), subplot_kw=dict(polar=True))
    axes = np.atleast_1d(axes).ravel()
    for ax, g in zip(axes, genres):
        v = prof.loc[g, rfe].values
        v = np.append(v, v[0])
        ax.plot(ang, v, linewidth=3, color="#1f77b4")
        ax.fill(ang, v, alpha=0.28, color="#1f77b4")
        ax.set_ylim(-vmax, vmax)
        ax.set_xticks(ang[:-1])
        ax.set_xticklabels(rlab, fontsize=14, fontweight="bold")
        ax.tick_params(axis="x", pad=14)
        ax.set_yticklabels([])
        ax.set_title(g, fontsize=20, fontweight="bold", pad=26)
    for ax in axes[len(genres):]:
        ax.axis("off")
    fig.suptitle("Standardized colour profile by genre (all genre labels)", fontsize=26, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.7, wspace=1.1)
    save_fig(tag + "_fig_radar_grid")

    plt.figure(figsize=(10, 9))
    ax = plt.subplot(111, polar=True)
    cmap = plt.get_cmap("tab20")
    for i, g in enumerate(genres):
        v = prof.loc[g, rfe].values
        v = np.append(v, v[0])
        ax.plot(ang, v, linewidth=2, label=g, color=cmap(i % 20))
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(rlab, fontsize=13)
    ax.set_title("Standardized colour profile by genre (overlaid)", y=1.1, fontsize=14)
    ax.legend(loc="upper right", bbox_to_anchor=(1.42, 1.1), fontsize=10)
    save_fig(tag + "_fig_radar_overlay")


def analyze_multilabel(tag, title, feats, Y, genres, feature_cols, do_perm=True, do_shap=True,
                       shap_genres=None, shap_n=SHAP_N_EXPLAIN_ML, do_profiles=False):
    X = feats[feature_cols].values.astype(float)
    n, G = Y.shape
    res = {"kind": "multi", "tag": tag, "genres": genres, "n": n, "feature_cols": feature_cols}
    print(title + ": n=" + str(n) + ", number of genres=" + str(G))

    # 1) descriptives + member vs. non-member standardized mean difference (Cohen's d), Mann-Whitney, FDR
    drow, desc = [], []
    for j, g in enumerate(genres):
        m1, m0 = Y[:, j] == 1, Y[:, j] == 0
        n1, n0 = int(m1.sum()), int(m0.sum())
        drec = {"genre": g, "n_films": n1}
        for fi, f in enumerate(feature_cols):
            a, b = X[m1, fi], X[m0, fi]
            sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n0 - 1) * b.var(ddof=1)) / max(n1 + n0 - 2, 1))
            d = (a.mean() - b.mean()) / sp if sp > 0 else 0.0
            se = np.sqrt((n1 + n0) / (n1 * n0) + d ** 2 / (2.0 * (n1 + n0)))
            try:
                pmw = stats.mannwhitneyu(a, b, alternative="two-sided")[1]
            except Exception:
                pmw = 1.0
            drow.append({"genre": g, "feature": f, "n_members": n1, "n_non_members": n0,
                         "mean_members": float(a.mean()), "mean_non_members": float(b.mean()),
                         "cohen_d": float(d), "d_ci95_low": float(d - 1.96 * se), "d_ci95_high": float(d + 1.96 * se),
                         "p_mann_whitney": float(pmw)})
            drec[f + "_mean"] = float(a.mean())
            drec[f + "_sd"] = float(a.std(ddof=1))
        desc.append(drec)
    dl = pd.DataFrame(drow)
    dl["p_fdr_bh"] = bh_fdr(dl["p_mann_whitney"].values)
    dl["significant_fdr_05"] = dl["p_fdr_bh"] < 0.05
    save_table(dl, tag + "_genre_feature_effects_cohen_d")
    save_table(pd.DataFrame(desc), tag + "_descriptives_by_genre")
    dm = dl.pivot(index="genre", columns="feature", values="cohen_d").loc[genres, feature_cols]
    save_table(dm.rename_axis("genre").reset_index(), tag + "_cohen_d_matrix")
    res["d_long"] = dl
    res["d_matrix"] = dm
    vmax = max(0.3, float(np.abs(dm.values).max()))
    plt.figure(figsize=(max(10, 0.7 * len(feature_cols) + 3), max(5, 0.42 * G + 2.5)))
    im = plt.imshow(dm.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.xticks(range(len(feature_cols)), feature_cols, rotation=60, ha="right")
    plt.yticks(range(G), genres)
    plt.colorbar(im, label="Cohen's d (films carrying the genre vs. all other films)")
    plt.title("Genre colour signatures independent of label order (" + tag + ")")
    save_fig(tag + "_fig_colour_signature_cohen_d")

    # 2) omnibus: multivariate-regression Pillai's trace + per-feature R^2 (permutation)
    pillai, p_pillai, r2, p_r2 = mv_permutation_test(X, Y.astype(float), N_PERM_STATS, RNG_SEED)
    save_table(pd.DataFrame([{"statistic": "Pillai's trace (multivariate regression of features on all genre indicators)",
                              "observed_value": pillai, "n_permutations": N_PERM_STATS, "p_value": p_pillai,
                              "n_films": n, "n_genres": G}]), tag + "_omnibus_permutation")
    uni = pd.DataFrame({"feature": feature_cols, "R2_explained_by_genre_labels": r2, "p_permutation": p_r2})
    uni["p_fdr_bh"] = bh_fdr(uni["p_permutation"].values)
    uni["significant_fdr_05"] = uni["p_fdr_bh"] < 0.05
    uni["max_abs_cohen_d_across_genres"] = [float(np.abs(dm[f]).max()) for f in feature_cols]
    uni["genre_with_max_abs_d"] = [str(dm[f].abs().idxmax()) for f in feature_cols]
    uni = uni.sort_values("R2_explained_by_genre_labels", ascending=False).reset_index(drop=True)
    uni["rank"] = np.arange(1, len(uni) + 1)
    save_table(uni, tag + "_feature_R2_by_genre_labels")
    res["effect"] = uni.set_index("feature")["R2_explained_by_genre_labels"]
    res["pillai"], res["p_pillai"] = pillai, p_pillai
    pu = uni.sort_values("R2_explained_by_genre_labels")
    plt.figure(figsize=(8, 6))
    plt.barh(pu["feature"], pu["R2_explained_by_genre_labels"],
             color=["#d62728" if s else "#c7c7c7" for s in pu["significant_fdr_05"]])
    plt.xlabel("Variance explained by the full set of genre labels (R squared)")
    plt.title("Genre effect size per colour feature (" + tag + ")\n(red = significant after FDR correction)", fontsize=12)
    save_fig(tag + "_fig_effect_sizes")

    if do_profiles:
        profile_figures(tag, feats, Y, genres)
        corr = feats[feature_cols].corr(method="spearman")
        save_table(corr.rename_axis("feature").reset_index(), tag + "_feature_correlation_spearman")

    # 3) classification (binary relevance), per-genre metrics
    prevalence = Y.mean(axis=0)
    summ = []
    best_auc, oof_rf, per_rf = -1, None, None
    for mname in ["RandomForest", "HistGradientBoosting"]:
        t0 = time.time()
        oof, pred = ovr_oof(mname, X, Y)
        rows = []
        for j, g in enumerate(genres):
            yj = Y[:, j]
            auc = roc_auc_score(yj, oof[:, j])
            ap = average_precision_score(yj, oof[:, j])
            pr, rc, f1, _ = precision_recall_fscore_support(yj, pred[:, j], average="binary", zero_division=0)
            try:
                pmw = stats.mannwhitneyu(oof[yj == 1, j], oof[yj == 0, j], alternative="greater")[1]
            except Exception:
                pmw = 1.0
            rows.append({"model": mname, "genre": g, "support": int(yj.sum()), "prevalence": float(prevalence[j]),
                         "roc_auc": float(auc), "average_precision": float(ap),
                         "average_precision_chance(=prevalence)": float(prevalence[j]),
                         "precision": float(pr), "recall": float(rc), "f1": float(f1),
                         "n_predicted_positive": int(pred[:, j].sum()), "p_auc_gt_chance_mann_whitney": float(pmw)})
        per = pd.DataFrame(rows)
        per["p_fdr_bh"] = bh_fdr(per["p_auc_gt_chance_mann_whitney"].values)
        summ.append({"model": mname, "macro_roc_auc": per["roc_auc"].mean(), "macro_average_precision": per["average_precision"].mean(),
                     "mean_prevalence(AP chance)": float(prevalence.mean()), "macro_f1": per["f1"].mean(),
                     "micro_f1": f1_score(Y, pred, average="micro", zero_division=0),
                     "n_genres_auc_significant_fdr05": int((per["p_fdr_bh"] < 0.05).sum()), "n_genres": G})
        print("  " + mname + ": macro-AUC=" + fmt(summ[-1]["macro_roc_auc"]) + " (" + "{:.0f}".format(time.time() - t0) + " s)")
        save_table(per, tag + "_per_genre_metrics_" + mname)
        if mname == "RandomForest":
            oof_rf, per_rf = oof, per
    cs = pd.DataFrame(summ).sort_values("macro_roc_auc", ascending=False)
    save_table(cs, tag + "_classification_summary")
    res["perf"] = {"macro_auc": float(cs.set_index("model").loc["RandomForest", "macro_roc_auc"]),
                   "macro_f1": float(cs.set_index("model").loc["RandomForest", "macro_f1"]),
                   "micro_f1": float(cs.set_index("model").loc["RandomForest", "micro_f1"]),
                   "best_model": str(cs.iloc[0]["model"])}
    res["per_genre"] = per_rf
    ps = per_rf.sort_values("roc_auc")
    plt.figure(figsize=(8, max(4, 0.4 * G + 1.5)))
    plt.barh(ps["genre"], ps["roc_auc"], color="steelblue")
    plt.axvline(0.5, color="gray", linestyle="--", label="Chance (AUC = .50)")
    plt.xlim(0.4, max(0.8, float(ps["roc_auc"].max()) + 0.03))
    plt.xlabel("ROC AUC (out-of-fold, colour features only)")
    plt.title("Per-genre discriminability from colour (Random Forest, " + tag + ")")
    plt.legend(fontsize=9)
    save_fig(tag + "_fig_per_genre_auc")

    # 4) permutation test vs. chance with refitting (macro AUC)
    if do_perm:
        rng = np.random.default_rng(RNG_SEED)
        if PERM_ML_MAX_N is not None and n > PERM_ML_MAX_N:
            idx = np.sort(rng.choice(n, PERM_ML_MAX_N, replace=False))
        else:
            idx = np.arange(n)
        Xp, Yp = X[idx], Y[idx]
        keepg = [j for j in range(G) if 10 <= Yp[:, j].sum() <= len(idx) - 10]
        Yp = Yp[:, keepg]
        t0 = time.time()
        obs = ml_macro_auc_once(Xp, Yp, 0, False, PERM_ML_TREES, ML_RF_MAX_DEPTH, ML_RF_MIN_LEAF, RNG_SEED)
        one = time.time() - t0
        print("  Multi-label permutation test: " + str(N_PERM_MULTILABEL) + " permutations, n=" + str(len(idx))
              + " (one round ~" + "{:.0f}".format(one) + " s)...")
        null = Parallel(n_jobs=N_JOBS)(
            delayed(ml_macro_auc_once)(Xp, Yp, RNG_SEED + 1000 + b, True, PERM_ML_TREES, ML_RF_MAX_DEPTH, ML_RF_MIN_LEAF, RNG_SEED)
            for b in range(N_PERM_MULTILABEL))
        null = np.array(null)
        pval = (1 + int((null >= obs).sum())) / (1 + N_PERM_MULTILABEL)
        save_table(pd.DataFrame([{"statistic": "macro ROC AUC (binary-relevance Random Forest, 3-fold, labels permuted jointly per film)",
                                  "observed": obs, "null_mean": float(null.mean()), "null_sd": float(null.std()),
                                  "null_max": float(null.max()), "n_permutations": N_PERM_MULTILABEL, "p_value": pval,
                                  "n_films_used": len(idx), "n_genres_used": len(keepg), "n_trees": PERM_ML_TREES}]),
                   tag + "_permutation_test_classifier")
        plt.figure(figsize=(7, 4.5))
        plt.hist(null, bins=15, color="lightgray", edgecolor="black", label="Null distribution (permuted labels)")
        plt.axvline(obs, color="red", linewidth=2, label="Observed macro-AUC = " + fmt(obs))
        plt.axvline(0.5, color="blue", linestyle="--", linewidth=1.5, label="Chance level = .500")
        plt.xlabel("Macro ROC AUC")
        plt.ylabel("Frequency")
        plt.title("Permutation test vs. chance (" + tag + ", p = " + fmt(pval, 4) + ")")
        plt.legend(fontsize=9)
        save_fig(tag + "_fig_permutation_test")
        res["perf"]["p_perm"] = float(pval)

    # 5) class-specific SHAP: one binary Random Forest per genre
    res["shap_global"] = None
    if do_shap:
        try:
            sg = [g for g in (shap_genres if shap_genres is not None else genres) if g in genres]
            tr, te = train_test_split(np.arange(n), test_size=0.3, random_state=RNG_SEED)
            rng = np.random.default_rng(RNG_SEED)
            ex = te if len(te) <= shap_n else np.sort(rng.choice(te, shap_n, replace=False))
            Xs = X[ex]
            arrs = []
            t0 = time.time()
            for g in sg:
                j = genres.index(g)
                m = ml_rf(N_JOBS).fit(X[tr], Y[tr, j])
                a = shap_array(m, Xs)
                pos = list(m.classes_).index(1) if a.shape[2] == len(m.classes_) else a.shape[2] - 1
                arrs.append(a[:, :, pos])
                print("  SHAP: " + g + " done (" + "{:.1f}".format((time.time() - t0) / 60) + " min)")
            arr = np.stack(arrs, axis=-1)
            sh = summarize_shap(tag, arr, Xs, sg, feature_cols, FOCUS_GENRES)
            res["shap_global"] = sh["global"]
            res["shap_long"] = sh["long"]
        except ImportError as e:
            print("  [WARNING] shap package missing; multi-label SHAP skipped (" + str(e) + ").")
        except Exception as e:
            print("  [WARNING] Multi-label SHAP failed: " + str(e))
            traceback.print_exc()

    bestg = per_rf.sort_values("roc_auc", ascending=False)["genre"].head(3).tolist()
    worstg = per_rf.sort_values("roc_auc")["genre"].head(3).tolist()
    note("[" + tag + "] " + title + ": n=" + str(n) + ", " + str(G) + " genres | Pillai=" + fmt(pillai) + ", p=" + fmt(p_pillai)
         + " | largest R^2: " + ", ".join(uni["feature"].head(3).tolist()) + " | RF macro-AUC=" + fmt(res["perf"]["macro_auc"])
         + (", permutation p=" + fmt(res["perf"]["p_perm"]) if "p_perm" in res["perf"] else "")
         + " | most discriminable: " + ", ".join(bestg) + "; weakest: " + ", ".join(worstg) + ".")
    return res


# =============================================================================
# HSV vs. CIELAB COMPARISON
# =============================================================================
def stage_hsv_lab(feats, RES):
    pairs = [("mean_value", "mean_L"), ("std_value", "std_L"), ("dom1_value", "dom1_L"),
             ("mean_saturation", "mean_chroma"), ("std_saturation", "std_chroma"), ("dom1_saturation", "dom1_chroma")]
    rows = []
    for a, b in pairs:
        rows.append({"hsv_feature": a, "cielab_feature": b, "n_posters": len(feats),
                     "pearson_r": float(np.corrcoef(feats[a], feats[b])[0, 1]),
                     "spearman_rho": safe_spearman(feats[a], feats[b])})
    t = pd.DataFrame(rows)
    save_table(t, "RL1_hsv_vs_cielab_feature_correlations")
    note("HSV-CIELAB: r(mean_value, mean_L*) = " + fmt(t.iloc[0]["pearson_r"]) + "; r(dom1_value, dom1_L*) = "
         + fmt(t.iloc[2]["pearson_r"]) + "; r(mean_saturation, mean_C*) = " + fmt(t.iloc[3]["pearson_r"]) + ".")

    plt.figure(figsize=(6.5, 5.5))
    plt.hexbin(feats["mean_value"], feats["mean_L"], gridsize=45, cmap="viridis", mincnt=1)
    plt.colorbar(label="Number of posters")
    plt.xlabel("Mean HSV Value")
    plt.ylabel("Mean CIELAB L*")
    plt.title("HSV Value vs. CIELAB lightness per poster (r = " + fmt(t.iloc[0]["pearson_r"]) + ")")
    save_fig("RL_fig_hsv_value_vs_cielab_L")

    comp = []
    for d in ["B", "C", "A"]:
        h, l = RES.get(d + "_hsv"), RES.get(d + "_lab")
        if h is None or l is None:
            continue
        rh = h["effect"].rank(ascending=False, method="min")
        rl = l["effect"].rank(ascending=False, method="min")
        sh, sl = h.get("shap_global"), l.get("shap_global")
        rsh = sh.rank(ascending=False, method="min") if sh is not None else None
        rsl = sl.rank(ascending=False, method="min") if sl is not None else None
        for a in FEATURE_COLS:
            b = HSV_TO_LAB[a]
            comp.append({"definition": d, "hsv_feature": a, "cielab_feature": b,
                         "effect_size_hsv": float(h["effect"][a]), "rank_hsv": int(rh[a]),
                         "effect_size_cielab": float(l["effect"][b]), "rank_cielab": int(rl[b]),
                         "shap_hsv": float(sh[a]) if sh is not None and a in sh.index else np.nan,
                         "shap_rank_hsv": int(rsh[a]) if rsh is not None and a in rsh.index else np.nan,
                         "shap_cielab": float(sl[b]) if sl is not None and b in sl.index else np.nan,
                         "shap_rank_cielab": int(rsl[b]) if rsl is not None and b in rsl.index else np.nan})
    if comp:
        c = pd.DataFrame(comp)
        save_table(c, "RL2_effect_sizes_and_shap_hsv_vs_cielab")
        perf = []
        for d in ["B", "C", "A"]:
            h, l = RES.get(d + "_hsv"), RES.get(d + "_lab")
            if h is None or l is None:
                continue
            for key in sorted(set(h["perf"]) & set(l["perf"])):
                if isinstance(h["perf"][key], (int, float)):
                    perf.append({"definition": d, "metric": key, "hsv_feature_set": h["perf"][key], "cielab_feature_set": l["perf"][key]})
        save_table(pd.DataFrame(perf), "RL3_classification_hsv_vs_cielab")
        for d in c["definition"].unique():
            s = c[(c["definition"] == d) & (c["hsv_feature"].isin(["mean_value", "dom1_value", "colorfulness"]))]
            note("[" + d + "] HSV -> CIELAB effect-size rank: " + "; ".join(
                str(r["hsv_feature"]) + " #" + str(r["rank_hsv"]) + " -> " + str(r["cielab_feature"]) + " #" + str(r["rank_cielab"])
                for _, r in s.iterrows()) + ".")
        d0 = c["definition"].iloc[0]
        s = c[c["definition"] == d0]
        xs = np.arange(len(s))
        plt.figure(figsize=(11, 5))
        plt.bar(xs - 0.2, s["effect_size_hsv"], width=0.4, label="HSV-based feature", color="#1f77b4")
        plt.bar(xs + 0.2, s["effect_size_cielab"], width=0.4, label="CIELAB counterpart", color="#2ca02c")
        plt.xticks(xs, [a + "\n/ " + b if a != b else a for a, b in zip(s["hsv_feature"], s["cielab_feature"])], rotation=60, ha="right", fontsize=8)
        plt.ylabel("Effect size")
        plt.title("Genre effect sizes: HSV features vs. CIELAB counterparts (definition " + d0 + ")")
        plt.legend()
        save_fig("RL_fig_effect_sizes_hsv_vs_cielab")


# =============================================================================
# FEATURE RANKING ACROSS GENRE DEFINITIONS
# =============================================================================
def stage_rank_comparison(RES):
    keys = [k for k in ["A0", "A_hsv", "B_hsv", "C_hsv", "D1", "D2_hsv"] if RES.get(k) is not None]
    labels = {"A0": "A0 submitted analysis", "A_hsv": "A first-listed", "B_hsv": "B multi-label", "C_hsv": "C single-genre films",
              "D1": "D1 random label", "D2_hsv": "D2 rarest label"}
    eff = pd.DataFrame({labels[k]: RES[k]["effect"].reindex(FEATURE_COLS) for k in keys})
    rk = eff.rank(ascending=False, method="min").astype(int)
    out = eff.add_prefix("effect | ").join(rk.add_prefix("rank | "))
    save_table(out.rename_axis("feature").reset_index(), "RX1_effect_size_ranks_across_genre_definitions")
    sk = [k for k in keys if RES[k].get("shap_global") is not None]
    if sk:
        sh = pd.DataFrame({labels[k]: RES[k]["shap_global"].reindex(FEATURE_COLS) for k in sk})
        srk = sh.rank(ascending=False, method="min").astype(int)
        save_table(sh.add_prefix("mean|SHAP| | ").join(srk.add_prefix("rank | ")).rename_axis("feature").reset_index(),
                   "RX2_shap_ranks_across_genre_definitions")
    rows = []
    for a, b in itertools.combinations(eff.columns, 2):
        rows.append({"definition_1": a, "definition_2": b, "quantity": "effect size",
                     "spearman_rho_of_feature_ranking": safe_spearman(eff[a], eff[b])})
    if sk:
        for a, b in itertools.combinations(sh.columns, 2):
            rows.append({"definition_1": a, "definition_2": b, "quantity": "mean |SHAP|",
                         "spearman_rho_of_feature_ranking": safe_spearman(sh[a], sh[b])})
    save_table(pd.DataFrame(rows), "RX3_rank_agreement_between_definitions")
    plt.figure(figsize=(1.6 * len(rk.columns) + 4, 7))
    im = plt.imshow(rk.values, aspect="auto", cmap="viridis_r", vmin=1, vmax=len(FEATURE_COLS))
    plt.xticks(range(len(rk.columns)), rk.columns, rotation=30, ha="right")
    plt.yticks(range(len(FEATURE_COLS)), FEATURE_COLS)
    for i in range(rk.shape[0]):
        for j in range(rk.shape[1]):
            plt.text(j, i, str(rk.values[i, j]), ha="center", va="center", color="black" if rk.values[i, j] <= 8 else "white", fontsize=9)
    plt.colorbar(im, label="Rank of feature by genre effect size (1 = largest)")
    plt.title("Does the feature ranking depend on how genre is defined?")
    save_fig("RX_fig_effect_rank_across_definitions")
    for f in ["mean_value", "dom1_value", "colorfulness"]:
        note("RANK (" + f + "): " + "; ".join(str(c) + " #" + str(int(rk.loc[f, c])) for c in rk.columns) + ".")


# =============================================================================
# H3: AGREEMENT WITH DIRECTIONAL PREDICTIONS
# =============================================================================
def stage_h3(resB, resB_lab):
    if resB is None:
        raise RuntimeError("No results for definition B.")
    dl = resB["d_long"].set_index(["genre", "feature"])
    dll = resB_lab["d_long"].set_index(["genre", "feature"]) if resB_lab is not None else None
    sl = resB.get("shap_long")
    sl = sl.set_index(["genre", "feature"]) if sl is not None else None
    rows = []
    for genre, feat, sign, assoc, src in H3_PREDICTIONS:
        if (genre, feat) not in dl.index:
            rows.append({"genre": genre, "feature": feat, "predicted_direction": "+" if sign > 0 else "-",
                         "association": assoc, "source_note": src, "status": "NOT TESTABLE (genre not in the multi-label set)"})
            continue
        r = dl.loc[(genre, feat)]
        d, lo, hi = float(r["cohen_d"]), float(r["d_ci95_low"]), float(r["d_ci95_high"])
        conf = (np.sign(d) == sign) and (lo > 0 or hi < 0)
        contra = (np.sign(d) == -sign) and (lo > 0 or hi < 0)
        rec = {"genre": genre, "feature": feat, "predicted_direction": "+" if sign > 0 else "-", "association": assoc,
               "source_note": src, "observed_cohen_d": d, "d_ci95_low": lo, "d_ci95_high": hi,
               "status": "CONSISTENT" if conf else ("CONTRADICTED" if contra else "INCONCLUSIVE (CI includes 0)")}
        lf = HSV_TO_LAB.get(feat, feat)
        if dll is not None and (genre, lf) in dll.index:
            rec["cielab_feature"] = lf
            rec["observed_cohen_d_cielab"] = float(dll.loc[(genre, lf)]["cohen_d"])
            rec["same_sign_in_cielab"] = bool(np.sign(rec["observed_cohen_d_cielab"]) == np.sign(d))
        if sl is not None and (genre, feat) in sl.index:
            rho = float(sl.loc[(genre, feat)]["direction_spearman_rho(feature,shap)"])
            rec["shap_direction_rho"] = rho
            rec["shap_direction_matches_prediction"] = bool(np.sign(rho) == sign and abs(rho) > 0.1)
            rec["shap_rank_within_genre"] = int(sl.loc[(genre, feat)]["rank_within_genre"])
        rows.append(rec)
    t = pd.DataFrame(rows)
    save_table(t, "RH1_H3_a_priori_predictions_vs_observed")
    tested = t[~t["status"].str.startswith("NOT TESTABLE")]
    k = int((tested["status"] == "CONSISTENT").sum())
    c = int((tested["status"] == "CONTRADICTED").sum())
    m = len(tested)
    try:
        pbin = stats.binomtest(k, k + c, 0.5, alternative="greater").pvalue if (k + c) > 0 else np.nan
    except Exception:
        pbin = np.nan
    summ = {"n_predictions_listed": len(t), "n_testable": m, "n_consistent": k, "n_contradicted": c,
            "n_inconclusive": m - k - c, "sign_test_p_one_sided(consistent vs contradicted)": pbin,
            "CAUTION": "Predictions combine an assumed affective tone per genre with published colour-affect relations "
                       "and were not preregistered. Agreement shows consistency between "
                       "poster colour regularities / model behaviour and published associations, not viewers' cognitive processes."}
    if "shap_direction_matches_prediction" in tested.columns:
        summ["n_shap_direction_matches"] = int(tested["shap_direction_matches_prediction"].fillna(False).sum())
    save_table(pd.DataFrame([summ]), "RH2_H3_agreement_summary")
    note("H3 (directional predictions): of " + str(m) + " testable predictions, " + str(k) + " consistent, " + str(c)
         + " contradicted, " + str(m - k - c) + " inconclusive (sign test p=" + fmt(pbin) + "). "
         + "Predictions were not preregistered.")


def stage_pad(feats, Y, genres):
    """Pleasure-arousal-dominance indices from the Valdez & Mehrabian (1994) regression weights (on L* and C*)."""
    zb = (feats["mean_L"] - feats["mean_L"].mean()) / feats["mean_L"].std()
    zs = (feats["mean_chroma"] - feats["mean_chroma"].mean()) / feats["mean_chroma"].std()
    P = 0.69 * zb + 0.22 * zs
    A = -0.31 * zb + 0.60 * zs
    Dm = -0.76 * zb + 0.32 * zs
    rows = []
    for j, g in enumerate(genres):
        m = Y[:, j] == 1
        rec = {"genre": g, "n_films": int(m.sum())}
        for nm, v in [("pleasure", P), ("arousal", A), ("dominance", Dm)]:
            x = v[m].values
            se = x.std(ddof=1) / np.sqrt(len(x))
            rec[nm + "_index_mean"] = float(x.mean())
            rec[nm + "_ci95_low"] = float(x.mean() - 1.96 * se)
            rec[nm + "_ci95_high"] = float(x.mean() + 1.96 * se)
        rows.append(rec)
    t = pd.DataFrame(rows)
    t["note"] = ("Indices = Valdez & Mehrabian (1994) equations applied to z-scored mean L* (brightness) and mean C*ab "
                 "(saturation): P=.69B+.22S, A=-.31B+.60S, D=-.76B+.32S. Theory-derived predictions, not viewer ratings.")
    save_table(t, "RH3_valdez_mehrabian_PAD_indices_by_genre")
    plt.figure(figsize=(8, 6.5))
    plt.errorbar(t["pleasure_index_mean"], t["arousal_index_mean"],
                 xerr=[t["pleasure_index_mean"] - t["pleasure_ci95_low"], t["pleasure_ci95_high"] - t["pleasure_index_mean"]],
                 yerr=[t["arousal_index_mean"] - t["arousal_ci95_low"], t["arousal_ci95_high"] - t["arousal_index_mean"]],
                 fmt="o", color="#1f77b4", ecolor="gray", capsize=2)
    for _, r in t.iterrows():
        plt.annotate(r["genre"], (r["pleasure_index_mean"], r["arousal_index_mean"]), xytext=(5, 4), textcoords="offset points", fontsize=9)
    plt.axhline(0, color="black", linewidth=0.6)
    plt.axvline(0, color="black", linewidth=0.6)
    plt.xlabel("Predicted pleasure index (from poster lightness and chroma)")
    plt.ylabel("Predicted arousal index")
    plt.title("Affect predicted from poster colour by the Valdez & Mehrabian (1994) equations")
    save_fig("RH_fig_predicted_pleasure_arousal_by_genre")


# =============================================================================
# EXPORT
# =============================================================================
OUTPUT_GUIDE = """
OUTPUT GUIDE   (files in tables/ and figures/; prefixes: R = raw data, RA0/RA/RB/RC/RD = genre definitions,
                RL = HSV vs. CIELAB, RX = across definitions, RH = hypothesis 3)
----------------------------------------------------------------------------------------------------------------
Order of genre labels in the JSON files           R02_genre_order_audit, R02b_examples_not_alphabetical
All-label vs. first-position distributions        R03_genre_frequency_all_vs_first_position, R_fig01
Dataset provenance                                R01_dataset_provenance, R01b_release_year_distribution
Label overlap / co-occurrence                     R04_labels_per_film, R05-R07_cooccurrence_*, R_fig02
Order-independent genre definitions               RB_* (multi-label, MAIN), RC_* (single-label films)
Alternative single-label rules                    RD1_random_label_rule_*, RD2_* (rarest label)
Feature ranking across definitions                RX1/RX2/RX3_*_across_genre_definitions, RX_fig_effect_rank
Per-genre precision / recall / F1 / support       RA0_/RA_/RC_/RD2_per_genre_precision_recall_f1,
  and row-normalised confusion matrices           *_confusion_matrix_row_normalised, *_fig_confusion_normalised
Class imbalance (smallest genre)                  RA0_/RA_imbalance_sensitivity_per_genre, *_strict_vs_any_label_accuracy
10% border crop                                   R13_border_crop_feature_agreement, R_fig03,
                                                  R14/R15_border_crop_*_full_vs_cropped, RB_CROP_*
Resize, grayscale weights, k                      R12_sensitivity_resize_grayscale_k
HSV value vs. CIELAB L*                           RL1/RL2/RL3_*, RL_fig_*, *_LAB_* tables
Class-specific, signed SHAP and dependence plots  *_shap_by_genre_signed, *_shap_signed_matrix,
                                                  *_fig_shap_by_genre_signed, *_fig_shap_dependence_<genre>
SHAP for correlated features                      *_shap_feature_family, RB_feature_correlation_spearman
Directional predictions (H3)                      RH1_H3_a_priori_predictions_vs_observed, RH2_*, RH3_*
Implementation details (grayscale, entropy, SHAP) implementation_details.txt
"""


def stage_export(t_start):
    with open(os.path.join(OUTPUT_DIR, "SUMMARY_REPORT.txt"), "w", encoding="utf-8") as fh:
        fh.write("SUMMARY REPORT\n")
        fh.write("Created: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M") + " | DATA_SOURCE=" + DATA_SOURCE
                 + " | QUICK_TEST_MODE=" + str(QUICK_TEST_MODE) + "\n")
        fh.write("Total run time: " + str(datetime.timedelta(seconds=int(time.time() - t_start))) + "\n\n")
        if SMALL:
            fh.write("!!! NOTE: this is a TEST run (demo / quick). The numbers are not research results. !!!\n\n")
        fh.write("KEY RESULTS\n" + "-" * 14 + "\n")
        for line in SUMMARY:
            fh.write(line + "\n")
        if FAILED:
            fh.write("\nFAILED STAGES: " + "; ".join(FAILED) + "\n(see run_log.txt for details)\n")
        fh.write("\n" + OUTPUT_GUIDE)
    try:
        used = set()
        with pd.ExcelWriter(os.path.join(OUTPUT_DIR, "ALL_RESULTS_combined.xlsx"), engine="openpyxl") as writer:
            for name, d, index in TABLES:
                sheet = name[:28]
                i = 1
                while sheet in used:
                    sheet = name[:26] + "_" + str(i)
                    i += 1
                used.add(sheet)
                d.to_excel(writer, sheet_name=sheet, index=index)
        print("Combined workbook written (" + str(len(TABLES)) + " sheets).")
    except Exception as e:
        print("  [WARNING] Combined workbook could not be written: " + str(e))
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for root_, _, files_ in os.walk(OUTPUT_DIR):
            for fn in files_:
                full = os.path.join(root_, fn)
                zf.write(full, arcname=os.path.relpath(full, OUTPUT_DIR))
    print("Zip written: " + ZIP_PATH)


# =============================================================================
# MAIN
# =============================================================================
def main():
    t_start = time.time()
    for d in [BASE_DIR, OUTPUT_DIR, FIG_DIR, TAB_DIR, CACHE_DIR]:
        os.makedirs(d, exist_ok=True)
    print("Working directory: " + BASE_DIR)
    print("Output directory : " + OUTPUT_DIR)
    print("DATA_SOURCE=" + DATA_SOURCE + " | QUICK_TEST_MODE=" + str(QUICK_TEST_MODE))

    run_stage("0. Environment and implementation details", stage_environment)
    RES = {}
    if DATA_SOURCE == "official":
        # Needs no poster files; produces results even when the dataset is not available.
        RES["A0"] = run_stage("0b. Exact reproduction of the original first-listed analysis (per-genre table + class-specific SHAP)",
                              stage_original_exact)
    try:
        root = run_stage("Dataset preparation", prepare_dataset, critical=True)
    except Exception:
        print(MANUAL_DOWNLOAD_HELP.format(url=MMIMDB_URL, dest=MMIMDB_OFFICIAL_DIR))
        if RES.get("A0") is not None:
            print("NOTE: the reproduction of the first-listed analysis (RA0_* tables and figures) finished and was saved to:")
            print("     " + OUTPUT_DIR)
        run_stage("Export (partial results)", stage_export, t_start)
        raise
    meta = run_stage("1. Raw-data audit (label order, frequencies, co-occurrence)", stage_audit, root, critical=True)
    feats = run_stage("2. Feature extraction (all films)", stage_features, meta, critical=True)
    if RUN_SENSITIVITY_SUBSET:
        run_stage("2b. Resize / grayscale / k sensitivity (subsample)", stage_sensitivity_subset, feats)
    run_stage("3. 10% border-crop check", stage_border_crop, feats)
    defs = run_stage("4. Genre definitions and samples", stage_definitions, feats, critical=True)
    Y, genres_ml, alm = defs["Y"], defs["genres_ml"], defs["all_label_map"]

    RES["B_hsv"] = run_stage("5. DEFINITION B -- multi-label (MAIN ANALYSIS)", analyze_multilabel, "RB",
                             "Multi-label, all films", feats, Y, genres_ml, FEATURE_COLS,
                             do_perm=True, do_shap=True, do_profiles=True)
    run_stage("5b. Main analysis repeated with border-cropped posters", stage_border_crop_reanalysis,
              feats, Y, genres_ml, RES.get("B_hsv"))
    RES["C_hsv"] = run_stage("6. DEFINITION C -- single-label films only", analyze_single_label, "RC",
                             "Single-label films", defs["C"], FEATURE_COLS, all_label_map=alm, do_perm=True, do_shap=True)
    if RUN_DEF_A:
        RES["A_hsv"] = run_stage("7. DEFINITION A -- first-listed label", analyze_single_label, "RA",
                                 "First-listed label", defs["A"], FEATURE_COLS, all_label_map=alm,
                                 do_perm=RUN_PERM_FOR_A, do_shap=True)
        run_stage("7b. DEFINITION A -- imbalance sensitivity (class weights / equal n)", stage_imbalance_checks, defs["A"], FEATURE_COLS)
    RES["D2_hsv"] = run_stage("8. DEFINITION D2 -- rarest-label rule", analyze_single_label, "RD2",
                              "Rarest-label rule", defs["D2"], FEATURE_COLS, all_label_map=alm, do_perm=False, do_shap=False)
    RES["D1"] = run_stage("9. DEFINITION D1 -- random-label rule", stage_random_label, feats, defs, FEATURE_COLS)

    if RUN_LAB:
        RES["B_lab"] = run_stage("10. CIELAB -- multi-label", analyze_multilabel, "RB_LAB", "Multi-label (CIELAB features)",
                                 feats, Y, genres_ml, FEATURE_COLS_LAB, do_perm=False, do_shap=True,
                                 shap_genres=FOCUS_GENRES, shap_n=SHAP_N_EXPLAIN_LAB)
        RES["C_lab"] = run_stage("11. CIELAB -- single-label films", analyze_single_label, "RC_LAB",
                                 "Single-label films (CIELAB features)", defs["C"], FEATURE_COLS_LAB,
                                 do_perm=False, do_shap=True, shap_n=SHAP_N_EXPLAIN_LAB)
        if RUN_DEF_A:
            RES["A_lab"] = run_stage("12. CIELAB -- first-listed label", analyze_single_label, "RA_LAB",
                                     "First-listed label (CIELAB features)", defs["A"], FEATURE_COLS_LAB,
                                     do_perm=False, do_shap=False)
        run_stage("13. HSV vs. CIELAB comparison", stage_hsv_lab, feats, RES)

    run_stage("14. Feature ranking across genre definitions", stage_rank_comparison, RES)
    run_stage("15. H3 -- agreement with directional predictions", stage_h3, RES.get("B_hsv"), RES.get("B_lab"))
    run_stage("16. Valdez & Mehrabian pleasure-arousal-dominance indices", stage_pad, feats, Y, genres_ml)
    run_stage("17. Export", stage_export, t_start)

    print("\nAll outputs: " + OUTPUT_DIR)
    if FAILED:
        print("WARNING -- failed stages: " + "; ".join(FAILED))


if __name__ == "__main__":
    os.makedirs(BASE_DIR, exist_ok=True)
    log_file = open(LOG_PATH, "a", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)
    start = datetime.datetime.now()
    print("\n\n########## RUN STARTED: " + start.strftime("%Y-%m-%d %H:%M:%S") + " ##########")
    try:
        main()
        end = datetime.datetime.now()
        print("\n########## FINISHED: " + end.strftime("%Y-%m-%d %H:%M:%S") + " (total run time: " + str(end - start) + ") ##########")
    except Exception:
        print("\n########## FATAL ERROR -- RUN STOPPED ##########")
        traceback.print_exc()
        print("\nDetails: " + LOG_PATH)
    finally:
        log_file.close()
