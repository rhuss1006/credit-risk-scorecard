"""
Credit risk scorecard library.

Implements the standard credit-scoring toolkit from scratch so the mechanics
are visible rather than hidden behind a package:

  - monotonic coarse binning for numeric characteristics
  - rare-level collapsing for categorical characteristics
  - Weight of Evidence transform and Information Value screening
  - KS, Gini, gains table
  - PDO score scaling and per-attribute point allocation
  - Population Stability Index at score and characteristic level

Convention used throughout: target is 1 = BAD (defaulted), 0 = GOOD.
WOE is defined as ln(share of goods / share of bads), so a POSITIVE WOE
means the bin is safer than the portfolio average.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

MISSING = "MISSING"
OTHER = "OTHER"


# ---------------------------------------------------------------------------
# WOE / IV core
# ---------------------------------------------------------------------------

def woe_table(bins: pd.Series, target: pd.Series) -> pd.DataFrame:
    """
    Per-bin good/bad counts, WOE and IV contribution.

    The +0.5 adjustment is the standard Haldane correction. Without it a bin
    containing zero bads produces an infinite WOE, which then propagates into
    the model and makes the scorecard unshippable.
    """
    df = pd.DataFrame({"bin": bins.astype(str), "target": target.values})
    grp = df.groupby("bin", observed=True)["target"].agg(["count", "sum"])
    grp.columns = ["total", "bad"]
    grp["good"] = grp["total"] - grp["bad"]

    tot_bad = grp["bad"].sum()
    tot_good = grp["good"].sum()

    grp["pct_bad"] = (grp["bad"] + 0.5) / (tot_bad + 0.5 * len(grp))
    grp["pct_good"] = (grp["good"] + 0.5) / (tot_good + 0.5 * len(grp))

    grp["woe"] = np.log(grp["pct_good"] / grp["pct_bad"])
    grp["iv_contrib"] = (grp["pct_good"] - grp["pct_bad"]) * grp["woe"]
    grp["bad_rate"] = grp["bad"] / grp["total"]
    grp["pop_pct"] = grp["total"] / grp["total"].sum()
    return grp


def information_value(tbl: pd.DataFrame) -> float:
    return float(tbl["iv_contrib"].sum())


def iv_verdict(iv: float) -> str:
    if iv < 0.02:
        return "unpredictive"
    if iv < 0.10:
        return "weak"
    if iv < 0.30:
        return "medium"
    if iv < 0.50:
        return "strong"
    return "suspicious, check leakage"


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------

@dataclass
class NumericBinner:
    """
    Coarse-classes a numeric characteristic into monotonic-WOE bins.

    Strategy: start from quantile cuts, then repeatedly merge the adjacent pair
    that most violates monotonicity until the WOE trend is strictly one
    direction. Regulators and business reviewers reject a characteristic where
    risk zig-zags, because it cannot be explained to a declined applicant.
    """
    name: str
    max_bins: int = 10
    min_bin_frac: float = 0.05
    edges: list = field(default_factory=list)

    def fit(self, x: pd.Series, y: pd.Series) -> "NumericBinner":
        x = pd.to_numeric(x, errors="coerce")
        mask = x.notna()
        xv, yv = x[mask], y[mask]

        qs = np.linspace(0, 1, self.max_bins + 1)
        edges = np.unique(np.quantile(xv, qs))
        edges[0], edges[-1] = -np.inf, np.inf
        edges = list(edges)

        min_n = max(int(self.min_bin_frac * len(xv)), 20)

        # enforce minimum bin size
        while len(edges) > 3:
            counts = pd.cut(xv, bins=edges).value_counts(sort=False).values
            small = int(np.argmin(counts))
            if counts[small] >= min_n:
                break
            drop = small + 1 if small == 0 else small
            edges.pop(drop)

        # enforce monotonic WOE
        while len(edges) > 3:
            woes = self._bin_woes(xv, yv, edges)
            diffs = np.diff(woes)
            up, down = np.sum(diffs > 0), np.sum(diffs < 0)
            direction = 1 if up >= down else -1
            violations = np.where(np.sign(diffs) != direction)[0]
            if len(violations) == 0:
                break
            worst = violations[np.argmax(np.abs(diffs[violations]))]
            edges.pop(worst + 1)

        self.edges = edges
        return self

    def _bin_woes(self, x, y, edges):
        b = pd.cut(x, bins=edges)
        tbl = woe_table(b, y)
        order = pd.cut(x, bins=edges).cat.categories.astype(str)
        return tbl.reindex(order)["woe"].values

    def transform(self, x: pd.Series) -> pd.Series:
        x = pd.to_numeric(x, errors="coerce")
        out = pd.cut(x, bins=self.edges).astype(str)
        return pd.Series(np.where(x.isna(), MISSING, out), index=x.index)


@dataclass
class CategoricalBinner:
    """
    Collapses levels that are too thin to estimate a stable WOE into OTHER.
    A bin holding 1 percent of the population produces a WOE driven by a
    handful of accounts and will not replicate next quarter.
    """
    name: str
    min_bin_frac: float = 0.05
    keep: list = field(default_factory=list)

    def fit(self, x: pd.Series, y: pd.Series) -> "CategoricalBinner":
        x = x.astype(str).fillna(MISSING)
        freq = x.value_counts(normalize=True)
        self.keep = sorted(freq[freq >= self.min_bin_frac].index.tolist())
        return self

    def transform(self, x: pd.Series) -> pd.Series:
        x = x.astype(str).fillna(MISSING)
        return x.where(x.isin(self.keep), OTHER)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def ks_statistic(y_true, y_score) -> float:
    """
    Max vertical distance between the cumulative bad and cumulative good
    distributions across the score range. The credit industry's default
    discrimination measure. y_score = probability of bad.
    """
    df = pd.DataFrame({"y": np.asarray(y_true), "s": np.asarray(y_score)})
    df = df.sort_values("s")
    cum_bad = (df["y"] == 1).cumsum() / max((df["y"] == 1).sum(), 1)
    cum_good = (df["y"] == 0).cumsum() / max((df["y"] == 0).sum(), 1)
    return float(np.max(np.abs(cum_bad - cum_good)))


def gini(y_true, y_score) -> float:
    return 2 * roc_auc_score(y_true, y_score) - 1


def gains_table(y_true, score, n_bands: int = 10) -> pd.DataFrame:
    """
    Score bands from worst to best with actual bad rate and lift.
    This is the artefact a business stakeholder actually reads.
    """
    df = pd.DataFrame({"y": np.asarray(y_true), "score": np.asarray(score)})
    df["band"] = pd.qcut(df["score"].rank(method="first"), n_bands,
                         labels=[f"{i+1}" for i in range(n_bands)])
    tbl = df.groupby("band", observed=True).agg(
        accounts=("y", "size"),
        bads=("y", "sum"),
        min_score=("score", "min"),
        max_score=("score", "max"),
    )
    tbl["bad_rate"] = tbl["bads"] / tbl["accounts"]
    base = tbl["bads"].sum() / tbl["accounts"].sum()
    tbl["lift"] = tbl["bad_rate"] / base
    tbl["cum_bad_capture"] = tbl["bads"].cumsum() / tbl["bads"].sum()
    tbl["cum_pop"] = tbl["accounts"].cumsum() / tbl["accounts"].sum()
    return tbl.round(4)


def psi(expected, actual, n_bands: int = 10) -> float:
    """
    Population Stability Index. Bands are fixed on the expected (development)
    distribution, then the actual distribution is poured into those same bands.
    < 0.10 stable | 0.10-0.25 moderate shift | > 0.25 investigate
    """
    expected, actual = np.asarray(expected), np.asarray(actual)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bands + 1)))
    edges[0], edges[-1] = -np.inf, np.inf

    e = np.histogram(expected, bins=edges)[0] / len(expected)
    a = np.histogram(actual, bins=edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def categorical_psi(expected: pd.Series, actual: pd.Series) -> float:
    """PSI for a binned characteristic rather than a continuous score."""
    e = expected.astype(str).value_counts(normalize=True)
    a = actual.astype(str).value_counts(normalize=True)
    levels = sorted(set(e.index) | set(a.index))
    ev = np.clip(e.reindex(levels).fillna(0).values, 1e-6, None)
    av = np.clip(a.reindex(levels).fillna(0).values, 1e-6, None)
    return float(np.sum((av - ev) * np.log(av / ev)))


def psi_verdict(v: float) -> str:
    if v < 0.10:
        return "stable"
    if v < 0.25:
        return "moderate shift"
    return "significant shift, investigate"


# ---------------------------------------------------------------------------
# Score scaling
# ---------------------------------------------------------------------------

@dataclass
class ScoreScaler:
    """
    Score = Offset + Factor * ln(odds_good_to_bad)
    Factor = PDO / ln(2)
    Offset = base_score - Factor * ln(base_odds)

    Higher score = lower risk. With PDO = 20, every 20 points up doubles the
    good:bad odds. This is the entire mechanism behind a consumer credit score.
    """
    base_score: float = 600.0
    base_odds: float = 50.0
    pdo: float = 20.0

    @property
    def factor(self) -> float:
        return self.pdo / np.log(2)

    @property
    def offset(self) -> float:
        return self.base_score - self.factor * np.log(self.base_odds)

    def from_prob_bad(self, p_bad) -> np.ndarray:
        p_bad = np.clip(np.asarray(p_bad, dtype=float), 1e-9, 1 - 1e-9)
        odds_good = (1 - p_bad) / p_bad
        return self.offset + self.factor * np.log(odds_good)

    def points_table(self, intercept: float, coefs: pd.Series,
                     woe_tables: dict) -> pd.DataFrame:
        """
        Allocate the score into per-attribute points.

        ln(odds_good) = -(b0 + sum_i b_i * woe_i), so

            score = offset - factor*b0 - factor * sum_i b_i * woe_i

        Spreading the constant evenly across the n characteristics in the model:

            points_i = (offset - factor*b0)/n  -  factor * b_i * woe_i

        Points then sum exactly to the applicant's total score, which is what
        makes adverse-action reason codes possible: you report the
        characteristics where the applicant lost the most points.
        """
        n = len(coefs)
        const_share = (self.offset - self.factor * intercept) / n

        rows = []
        for char, beta in coefs.items():
            tbl = woe_tables[char]
            for attr, r in tbl.iterrows():
                rows.append({
                    "characteristic": char,
                    "attribute": attr,
                    "pop_pct": round(r["pop_pct"], 4),
                    "bad_rate": round(r["bad_rate"], 4),
                    "woe": round(r["woe"], 4),
                    "coefficient": round(beta, 4),
                    "points": round(const_share - self.factor * beta * r["woe"], 1),
                })
        out = pd.DataFrame(rows)
        return out.sort_values(["characteristic", "points"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cutoff selection
# ---------------------------------------------------------------------------

def cutoff_analysis(y_true, score, cost_bad_approved: float = 5.0,
                    cost_good_declined: float = 1.0,
                    n_points: int = 40) -> pd.DataFrame:
    """
    Sweep the approve/decline threshold and cost it out.

    The German Credit documentation supplies an explicit cost matrix: approving
    a bad applicant is 5x as expensive as declining a good one. Cutoff choice
    is therefore an economics question, not a statistics question, which is the
    correct answer when an interviewer asks how you pick a threshold.
    """
    y = np.asarray(y_true)
    s = np.asarray(score)
    rows = []
    for c in np.quantile(s, np.linspace(0.02, 0.98, n_points)):
        approve = s >= c
        bad_approved = int(np.sum(approve & (y == 1)))
        good_declined = int(np.sum(~approve & (y == 0)))
        rows.append({
            "cutoff": round(float(c), 1),
            "approval_rate": round(float(approve.mean()), 4),
            "bad_rate_of_approved": round(
                float(y[approve].mean()) if approve.any() else 0.0, 4),
            "bads_approved": bad_approved,
            "goods_declined": good_declined,
            "expected_cost": round(
                cost_bad_approved * bad_approved
                + cost_good_declined * good_declined, 1),
        })
    return pd.DataFrame(rows)
