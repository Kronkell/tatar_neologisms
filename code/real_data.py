"""E4: реальные попарные оценки сложности слов (BenchLS + Word Complexity Lexicon)."""
from __future__ import annotations

import hashlib
import math
import urllib.request
from collections import Counter
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

import config


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_raw_data(raw_dir=None) -> dict:
    """Скачать BenchLS и WCL в data/raw/ (если нет) и проверить sha256."""
    raw_dir = config.RAW_DATA_DIR if raw_dir is None else raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, (url, sha) in config.RAW_FILES.items():
        path = raw_dir / name
        if not path.exists():
            tmp = path.with_suffix(path.suffix + ".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(path)
        got = _sha256(path)
        if got != sha:
            raise RuntimeError(f"sha256 не совпадает для {name}: {got} != {sha}")
        paths[name] = path
    return paths


def load_wcl(path):
    """Словарь слово → массив 11 оценок (NaN — не оценивал); число дублей после lower()."""
    df = pd.read_csv(path, sep="\t", header=None, keep_default_na=False, na_filter=False,
                     quoting=3, dtype=str)
    ratings = {}
    n_dup = 0
    for row in df.itertuples(index=False):
        w = str(row[0]).lower()
        r = np.array([float(x) for x in row[1:1 + config.N_ANNOTATORS]])
        r[r < 0] = np.nan
        if w in ratings:
            n_dup += 1
        ratings[w] = r
    return ratings, n_dup


def load_benchls(path, vocab):
    """Группы: {целевое} ∪ {замены}, нижний регистр, без пробелов, только слова из vocab."""
    groups = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").rstrip("\r").split("\t")
            if len(parts) < 3:
                continue
            words = [parts[1]]
            for item in parts[3:]:
                if ":" not in item:
                    continue
                words.append(item.split(":", 1)[1])
            seen = []
            for w in words:
                w = w.strip().lower()
                if not w or " " in w or w not in vocab or w in seen:
                    continue
                seen.append(w)
            if len(seen) >= config.MIN_GROUP_SIZE:
                groups.append(seen)
    return groups


@dataclass
class RealData:
    words: list            # объекты (слова), индекс = номер объекта
    groups: list           # список массивов индексов объектов
    ratings: np.ndarray    # (N, 11) оценки сложности, NaN — нет оценки
    n_wcl_dups: int


def load_real(raw_dir=None) -> RealData:
    paths = ensure_raw_data(raw_dir)
    wcl, n_dup = load_wcl(paths["lexicon_annotations.tsv"])
    groups_w = load_benchls(paths["BenchLS.txt"], wcl)
    words = sorted({w for g in groups_w for w in g})
    idx = {w: i for i, w in enumerate(words)}
    groups = [np.array([idx[w] for w in g], dtype=int) for g in groups_w]
    ratings = np.vstack([wcl[w] for w in words])
    return RealData(words, groups, ratings, n_dup)


def all_comparisons(data: RealData, annotators=None):
    """Все строгие сравнения внутри групп: массивы winner, loser, annotator; и число ничьих.

    Победитель — более простое слово (меньшая оценка сложности).
    """
    ann = range(config.N_ANNOTATORS) if annotators is None else annotators
    win, lose, who = [], [], []
    ties = 0
    R = data.ratings
    for S in data.groups:
        for a, b in combinations(S, 2):
            for k in ann:
                ra, rb = R[a, k], R[b, k]
                if math.isnan(ra) or math.isnan(rb):
                    continue
                if ra == rb:
                    ties += 1
                elif ra < rb:
                    win.append(a); lose.append(b); who.append(k)
                else:
                    win.append(b); lose.append(a); who.append(k)
    return (np.array(win, dtype=int), np.array(lose, dtype=int), np.array(who, dtype=int)), ties


def control_counts(data: RealData) -> dict:
    (w, _, _), ties = all_comparisons(data)
    return {"groups": len(data.groups), "words": len(data.words), "strict": int(len(w)),
            "ties": int(ties)}


# ---------------------------------------------------------------- признаки
def _charlm_train_words(exclude):
    import wordfreq
    return [w for w in wordfreq.top_n_list("en", config.CHARLM_TOPN)
            if w.isalpha() and w not in exclude]


def charlm_scores(words, train_words, k=config.CHARLM_K):
    """Средний log p на символ (включая символ конца) по символьной триграммной модели, add-k."""
    BOS, EOS = "\x02", "\x03"
    tri, bi = Counter(), Counter()
    alphabet = {EOS}
    for w in train_words:
        s = BOS + BOS + w + EOS
        alphabet.update(w)
        for t in range(2, len(s)):
            tri[s[t - 2:t + 1]] += 1
            bi[s[t - 2:t]] += 1
    for w in words:
        alphabet.update(w)
    A = len(alphabet)
    out = np.empty(len(words))
    for n, w in enumerate(words):
        s = BOS + BOS + w + EOS
        lp = 0.0
        for t in range(2, len(s)):
            lp += math.log((tri[s[t - 2:t + 1]] + k) / (bi[s[t - 2:t]] + k * A))
        out[n] = lp / (len(s) - 2)
    return out


def raw_features(words) -> dict:
    import pyphen
    import wordfreq
    dic = pyphen.Pyphen(lang="en_US")
    train = _charlm_train_words(set(words))
    return {
        "zipf": np.array([wordfreq.zipf_frequency(w, "en") for w in words]),
        "length": np.array([len(w) for w in words], dtype=float),
        "syllables": np.array([dic.inserted(w).count("-") + 1 for w in words], dtype=float),
        "charlm": charlm_scores(words, train),
    }


def zscore(x):
    x = np.asarray(x, dtype=float)
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else x - x.mean()


def feature_matrix(words, names=config.REAL_FEATURES) -> np.ndarray:
    f = raw_features(words)
    return np.column_stack([zscore(f[n]) for n in names])
