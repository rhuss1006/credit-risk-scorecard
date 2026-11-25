"""
Credit Risk Scorecard, end-to-end build on the German Credit dataset.

Run:  python src/build_scorecard.py

Pipeline
  1. load and label
  2. out-of-sample split (fit everything on train only)
  3. coarse-class every characteristic, monotonic WOE for numerics
  4. Information Value screen
  5. WOE transform
  6. correlation screen
  7. logistic regression PD model
  8. coefficient sign review
  9. KS / Gini / AUC / gains table
 10. PDO score scaling and per-attribute points table
 11. cost-based cutoff selection using the dataset's own cost matrix
 12. simulated population shift and PSI monitoring
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scorecard import (  # noqa: E402
    CategoricalBinner, NumericBinner, ScoreScaler, categorical_psi,
    cutoff_analysis, gains_table, gini, information_value, iv_verdict,
    ks_statistic, psi, psi_verdict, woe_table,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "german.data")
OUT = os.path.join(ROOT, "outputs")
SEED = 42

COLUMNS = [
    "checking_status", "duration_months", "credit_history", "purpose",
    "credit_amount", "savings_status", "employment_since",
    "installment_rate_pct", "personal_status_sex", "other_debtors",
    "residence_since", "property", "age_years", "other_installment_plans",
    "housing", "existing_credits", "job", "dependents", "telephone",
    "foreign_worker", "target",
]

NUMERIC = [
    "duration_months", "credit_amount", "installment_rate_pct",
    "residence_since", "age_years", "existing_credits", "dependents",
]

# Excluded before modelling, not on statistical grounds.
# Under ECOA / Regulation B a credit model may not use sex or marital status,
# and immigration or nationality proxies carry disparate-impact risk. Dropping
# them up front is what a real scorecard developer does, and it is a better
# interview answer than discovering a high IV on a protected characteristic.
PROTECTED = ["personal_status_sex", "foreign_worker"]

IV_FLOOR = 0.02
CORR_CEILING = 0.70


def banner(txt):
    print("\n" + "=" * 74)
    print(txt)
    print("=" * 74)


# ---------------------------------------------------------------------------

def load():
    df = pd.read_csv(DATA, sep=r"\s+", header=None, names=COLUMNS)
    # source encoding: 1 = good, 2 = bad -> model target 1 = BAD
    df["target"] = (df["target"] == 2).astype(int)
    return df


def main():
    os.makedirs(OUT, exist_ok=True)

    banner("1. LOAD")
    df = load()
    print(f"records          {len(df)}")
    print(f"characteristics  {df.shape[1] - 1}")
    print(f"bad rate         {df['target'].mean():.2%}")
    print(f"\ndropped as protected / disparate-impact risk: {PROTECTED}")
    df = df.drop(columns=PROTECTED)

    banner("2. SPLIT")
    y = df["target"]
    X = df.drop(columns=["target"])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED)
    print(f"train {len(X_tr)}  bad rate {y_tr.mean():.2%}")
    print(f"test  {len(X_te)}  bad rate {y_te.mean():.2%}")
    print("bins and WOE are fit on TRAIN only, then applied to test. "
          "Fitting them on all data leaks the target.")

    banner("3-4. COARSE CLASSING AND IV SCREEN")
    binners, woe_tables, ivs = {}, {}, {}
    tr_bins = pd.DataFrame(index=X_tr.index)
    te_bins = pd.DataFrame(index=X_te.index)

    for col in X_tr.columns:
        if col in NUMERIC:
            b = NumericBinner(col).fit(X_tr[col], y_tr)
        else:
            b = CategoricalBinner(col).fit(X_tr[col], y_tr)
        binners[col] = b
        tr_bins[col] = b.transform(X_tr[col])
        te_bins[col] = b.transform(X_te[col])

        tbl = woe_table(tr_bins[col], y_tr)
        woe_tables[col] = tbl
        ivs[col] = information_value(tbl)

    iv_report = (pd.Series(ivs).sort_values(ascending=False)
                 .rename("iv").to_frame())
    iv_report["bins"] = [len(woe_tables[c]) for c in iv_report.index]
    iv_report["verdict"] = iv_report["iv"].map(iv_verdict)
    iv_report["retained"] = iv_report["iv"] >= IV_FLOOR
    print(iv_report.round(4).to_string())
    iv_report.round(4).to_csv(os.path.join(OUT, "iv_report.csv"))

    kept = iv_report.index[iv_report["retained"]].tolist()
    print(f"\nretained {len(kept)} of {len(iv_report)} at IV >= {IV_FLOOR}")

    banner("5-6. WOE TRANSFORM AND CORRELATION SCREEN")
    tr_woe = pd.DataFrame(
        {c: tr_bins[c].map(woe_tables[c]["woe"]).astype(float) for c in kept},
        index=X_tr.index)
    te_woe = pd.DataFrame(
        {c: te_bins[c].map(woe_tables[c]["woe"]).astype(float) for c in kept},
        index=X_te.index)
    # unseen test level -> WOE 0, i.e. portfolio-average risk, the neutral choice
    unseen = int(te_woe.isna().sum().sum())
    te_woe = te_woe.fillna(0.0)
    print(f"test cells hitting an unseen level, set to neutral WOE 0: {unseen}")

    corr = tr_woe.corr().abs()
    drop = set()
    for i, a in enumerate(corr.columns):
        for b_ in corr.columns[i + 1:]:
            if corr.loc[a, b_] > CORR_CEILING and a not in drop and b_ not in drop:
                weaker = a if ivs[a] < ivs[b_] else b_
                drop.add(weaker)
                print(f"  |r|={corr.loc[a, b_]:.2f} between {a} and {b_}, "
                      f"dropping lower-IV {weaker}")
    if not drop:
        print(f"  no pair above |r| = {CORR_CEILING}, nothing dropped")
    feats = [c for c in kept if c not in drop]
    tr_woe, te_woe = tr_woe[feats], te_woe[feats]

    banner("7-8. MODEL FIT AND COEFFICIENT SIGN REVIEW")
    # Target is BAD and WOE rises with safety, so every coefficient must be
    # negative. A positive one says the characteristic makes risk worse as it
    # gets safer, which is nonsense and almost always collinearity or a thin
    # bin. A scorecard cannot ship with one: you could not defend the points
    # to a regulator or explain them to a declined applicant.
    #
    # Standard remediation is to remove the offending characteristic and refit,
    # one at a time, because dropping one often repairs the others.
    def fit(cols):
        m = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
        m.fit(tr_woe[cols], y_tr)
        return m, pd.Series(m.coef_[0], index=cols)

    model, coefs = fit(feats)
    first_pass = coefs.copy()

    round_no = 0
    while (coefs > 0).any():
        round_no += 1
        offender = coefs.idxmax()
        print(f"  round {round_no}: {offender} has coefficient "
              f"{coefs[offender]:+.4f}, wrong sign, dropping and refitting")
        feats = [c for c in feats if c != offender]
        model, coefs = fit(feats)

    if round_no == 0:
        print("  all coefficients negative on the first fit, no remediation")
    tr_woe, te_woe = tr_woe[feats], te_woe[feats]
    intercept = float(model.intercept_[0])

    review = coefs.rename("coefficient").to_frame()
    review["iv"] = [ivs[c] for c in review.index]
    review["first_pass_coef"] = [round(first_pass[c], 4) for c in review.index]
    review["sign_ok"] = review["coefficient"] < 0
    review = review.sort_values("coefficient")
    print("\nFinal model:")
    print(review.round(4).to_string())
    print(f"\nintercept {intercept:.4f}")
    print(f"characteristics in final scorecard: {len(feats)}")
    assert review["sign_ok"].all(), "wrong-sign coefficient survived"
    print("sign review: all coefficients negative as expected")
    review.round(4).to_csv(os.path.join(OUT, "coefficients.csv"))

    banner("9. PERFORMANCE")
    p_tr = model.predict_proba(tr_woe)[:, 1]
    p_te = model.predict_proba(te_woe)[:, 1]

    perf = pd.DataFrame({
        "train": [roc_auc_score(y_tr, p_tr), gini(y_tr, p_tr),
                  ks_statistic(y_tr, p_tr)],
        "test": [roc_auc_score(y_te, p_te), gini(y_te, p_te),
                 ks_statistic(y_te, p_te)],
    }, index=["AUC", "Gini", "KS"])
    print(perf.round(4).to_string())
    print("\nGini = 2*AUC - 1. KS is the max gap between the cumulative "
          "good and bad distributions.")
    perf.round(4).to_csv(os.path.join(OUT, "performance.csv"))

    banner("10. SCORE SCALING")
    scaler = ScoreScaler(base_score=600, base_odds=50, pdo=20)
    print(f"base score {scaler.base_score:.0f} at odds "
          f"{scaler.base_odds:.0f}:1, PDO {scaler.pdo:.0f}")
    print(f"factor {scaler.factor:.2f}   offset {scaler.offset:.2f}")

    s_tr = scaler.from_prob_bad(p_tr)
    s_te = scaler.from_prob_bad(p_te)
    print(f"train scores {s_tr.min():.0f} to {s_tr.max():.0f}, "
          f"mean {s_tr.mean():.0f}")
    print(f"test  scores {s_te.min():.0f} to {s_te.max():.0f}, "
          f"mean {s_te.mean():.0f}")

    points = scaler.points_table(intercept, coefs, woe_tables)
    points.to_csv(os.path.join(OUT, "scorecard_points.csv"), index=False)
    print(f"\npoints table written, {len(points)} attribute rows")

    # additivity check: per-attribute points must sum to the scaled score
    chk = X_te.iloc[[0]]
    total = 0.0
    for c in feats:
        attr = binners[c].transform(chk[c]).iloc[0]
        row = points[(points["characteristic"] == c) &
                     (points["attribute"] == attr)]
        total += float(row["points"].iloc[0])
    print(f"additivity check: summed attribute points {total:.1f} vs "
          f"scaled score {s_te[0]:.1f}")

    gt = gains_table(y_te, s_te)
    print("\nGains table, band 1 = worst scores:")
    print(gt.to_string())
    gt.to_csv(os.path.join(OUT, "gains_table.csv"))

    banner("11. CUTOFF SELECTION USING THE DATASET COST MATRIX")
    print("german.doc: approving a bad costs 5, declining a good costs 1")
    ca = cutoff_analysis(y_te, s_te, cost_bad_approved=5.0,
                         cost_good_declined=1.0)
    best = ca.loc[ca["expected_cost"].idxmin()]
    print(ca.iloc[::4].to_string(index=False))
    print(f"\nminimum expected cost at score {best['cutoff']:.0f}: "
          f"approve {best['approval_rate']:.1%} of applicants, "
          f"bad rate of approved book {best['bad_rate_of_approved']:.2%}")
    print("Cutoff is an economics decision, not a statistics one.")
    ca.to_csv(os.path.join(OUT, "cutoff_analysis.csv"), index=False)

    banner("12. SIMULATED POPULATION SHIFT AND PSI MONITORING")
    print("Scenario: a new acquisition channel brings in more short-tenure, "
          "thin-file applicants. The model is unchanged, the population is not.")

    rng = np.random.default_rng(SEED)
    # weight selection toward the riskier half of the test book
    w = np.where(s_te < np.median(s_te), 3.0, 1.0)
    w = w / w.sum()
    idx = rng.choice(len(X_te), size=len(X_te), replace=True, p=w)
    shift_bins = te_bins.iloc[idx]
    s_shift = s_te[idx]

    base_psi = psi(s_tr, s_te)
    shift_psi = psi(s_tr, s_shift)
    print(f"\nscore PSI, train vs test      {base_psi:.4f}  "
          f"({psi_verdict(base_psi)})")
    print(f"score PSI, train vs shifted   {shift_psi:.4f}  "
          f"({psi_verdict(shift_psi)})")

    char_psi = pd.Series(
        {c: categorical_psi(tr_bins[c], shift_bins[c]) for c in feats}
    ).sort_values(ascending=False).rename("psi").to_frame()
    char_psi["verdict"] = char_psi["psi"].map(psi_verdict)
    print("\nCharacteristic-level PSI, train vs shifted population:")
    print(char_psi.round(4).to_string())
    char_psi.round(4).to_csv(os.path.join(OUT, "psi_report.csv"))
    print("\nScore PSI tells you THAT the population moved. "
          "Characteristic PSI tells you WHICH input moved.")

    banner("13. ADVERSE ACTION REASON CODES")
    print("ECOA / Regulation B: a declined applicant must be told the specific")
    print("principal reasons. Because scorecard points are additive, the reason")
    print("codes fall straight out: compare the points the applicant earned on")
    print("each characteristic against the maximum available, and report the")
    print("characteristics with the largest shortfall.\n")

    max_points = points.groupby("characteristic")["points"].max()
    lookup = points.set_index(["characteristic", "attribute"])["points"]

    declined = np.argsort(s_te)[:2]  # two lowest-scoring test applicants
    aa_rows = []
    for rank, i in enumerate(declined, start=1):
        rec = X_te.iloc[[i]]
        gaps = []
        for c in feats:
            attr = binners[c].transform(rec[c]).iloc[0]
            earned = float(lookup.loc[(c, attr)])
            gaps.append({
                "applicant": rank,
                "score": round(float(s_te[i]), 1),
                "characteristic": c,
                "attribute": attr,
                "points_earned": earned,
                "points_available": float(max_points[c]),
                "shortfall": round(float(max_points[c]) - earned, 1),
            })
        g = pd.DataFrame(gaps).sort_values("shortfall", ascending=False)
        aa_rows.append(g)
        print(f"Applicant {rank}, score {s_te[i]:.0f}, "
              f"declined at cutoff {best['cutoff']:.0f}")
        print("  principal reasons for adverse action:")
        for _, r in g.head(4).iterrows():
            print(f"    {r['shortfall']:5.1f} pts below best  "
                  f"{r['characteristic']} = {r['attribute']}")
        print()

    pd.concat(aa_rows).to_csv(
        os.path.join(OUT, "adverse_action_reasons.csv"), index=False)
    print("This is the reason a regulated lender uses an additive scorecard")
    print("instead of a gradient-boosted ensemble.")

    # ---------------- plots ----------------
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    fpr, tpr, _ = roc_curve(y_te, p_te)
    ax[0, 0].plot(fpr, tpr, lw=2)
    ax[0, 0].plot([0, 1], [0, 1], "k--", lw=1)
    ax[0, 0].set_title(f"ROC, test AUC = {roc_auc_score(y_te, p_te):.3f}")
    ax[0, 0].set_xlabel("false positive rate")
    ax[0, 0].set_ylabel("true positive rate")

    d = pd.DataFrame({"y": y_te.values, "s": s_te}).sort_values("s")
    cb = (d["y"] == 1).cumsum() / (d["y"] == 1).sum()
    cg = (d["y"] == 0).cumsum() / (d["y"] == 0).sum()
    ax[0, 1].plot(d["s"], cb, label="cumulative bad")
    ax[0, 1].plot(d["s"], cg, label="cumulative good")
    kidx = int(np.argmax(np.abs(cb.values - cg.values)))
    ax[0, 1].vlines(d["s"].iloc[kidx], cg.iloc[kidx], cb.iloc[kidx],
                    color="red", lw=2)
    ax[0, 1].set_title(f"KS = {ks_statistic(y_te, p_te):.3f} "
                       f"at score {d['s'].iloc[kidx]:.0f}")
    ax[0, 1].set_xlabel("score")
    ax[0, 1].legend()

    ax[1, 0].hist(s_te[y_te.values == 0], bins=30, alpha=0.6, label="good",
                  density=True)
    ax[1, 0].hist(s_te[y_te.values == 1], bins=30, alpha=0.6, label="bad",
                  density=True)
    ax[1, 0].axvline(best["cutoff"], color="k", ls="--",
                     label=f"cutoff {best['cutoff']:.0f}")
    ax[1, 0].set_title("Score distribution by outcome")
    ax[1, 0].set_xlabel("score")
    ax[1, 0].legend()

    ax[1, 1].hist(s_te, bins=30, alpha=0.6, label="development", density=True)
    ax[1, 1].hist(s_shift, bins=30, alpha=0.6, label="shifted", density=True)
    ax[1, 1].set_title(f"Population shift, score PSI = {shift_psi:.3f}")
    ax[1, 1].set_xlabel("score")
    ax[1, 1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "diagnostics.png"), dpi=130)

    top = iv_report.index[:6]
    fig2, axes = plt.subplots(2, 3, figsize=(15, 7))
    for a, c in zip(axes.ravel(), top):
        t = woe_tables[c].sort_values("woe")
        a.barh(range(len(t)), t["woe"].values)
        a.set_yticks(range(len(t)))
        a.set_yticklabels([str(i)[:22] for i in t.index], fontsize=7)
        a.axvline(0, color="k", lw=0.8)
        a.set_title(f"{c}\nIV = {ivs[c]:.3f}", fontsize=9)
    fig2.suptitle("Weight of Evidence by attribute, top 6 characteristics "
                  "(positive = safer than average)")
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUT, "woe_patterns.png"), dpi=130)

    banner("DONE")
    for f in sorted(os.listdir(OUT)):
        print(" ", f)


if __name__ == "__main__":
    main()
