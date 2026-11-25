# Credit Risk Scorecard

A production-style credit scorecard built from scratch on the German Credit dataset. Weight-of-evidence binning, information value screening, a logistic regression PD model, PDO score scaling, cost-based cutoff selection, PSI drift monitoring, and ECOA adverse action reason codes.

No scorecard package. The binning, WOE, KS, PSI and point allocation are all implemented directly so the mechanics are visible.

```
python src/build_scorecard.py
```

Requires pandas, numpy, scikit-learn, matplotlib.

---

## Data

1,000 German consumer loan records, 20 characteristics, 30 percent bad rate. Target recoded so 1 = bad (defaulted).

The dataset documentation supplies an explicit cost matrix: approving a bad applicant costs 5, declining a good applicant costs 1. That asymmetry is used to select the cutoff rather than defaulting to 0.5.

**Two characteristics were dropped before modelling on compliance grounds, not statistical ones.** `personal_status_sex` encodes sex and marital status, which ECOA and Regulation B prohibit in a credit decision. `foreign_worker` is a nationality proxy carrying disparate-impact risk. Both were removed up front.

---

## Results

Out-of-sample, 300 held-out records.

| Metric | Train | Test |
|---|---|---|
| AUC | 0.816 | 0.788 |
| Gini | 0.631 | 0.576 |
| KS | 0.508 | 0.479 |

A KS of 48 is in the range of a deployed retail scorecard. The train-to-test gap is modest, which is what WOE binning plus regularization is supposed to deliver.

### Gains table, test set

Band 1 is the worst-scoring decile.

| Band | Accounts | Score range | Bad rate | Lift |
|---|---|---|---|---|
| 1 | 30 | 427 to 470 | 73.3% | 2.44 |
| 2 | 30 | 470 to 486 | 63.3% | 2.11 |
| 5 | 30 | 508 to 518 | 36.7% | 1.22 |
| 6 | 30 | 518 to 531 | 10.0% | 0.33 |
| 10 | 30 | 570 to 626 | 0.0% | 0.00 |

The worst decile defaults at 73 percent, the best at zero. That table is what you put in front of a business stakeholder.

### Information value screen

| Characteristic | IV | Verdict |
|---|---|---|
| checking_status | 0.630 | very strong |
| credit_history | 0.252 | medium |
| savings_status | 0.221 | medium |
| duration_months | 0.206 | medium |
| property | 0.139 | medium |
| employment_since | 0.139 | medium |
| ... | | |
| telephone | 0.003 | unpredictive |
| dependents | 0.000 | unpredictive |

Five characteristics were dropped below IV 0.02. `checking_status` at 0.63 trips the usual leakage flag, but it is a known property of this dataset rather than a data error: having no checking account at all is the single strongest signal in the file.

### Wrong-sign remediation

The first fit produced a positive coefficient on `job`. Since the target is bad and WOE increases with safety, every coefficient must be negative. A positive one says risk rises as the characteristic gets safer, which is indefensible to a regulator and unexplainable to a declined applicant.

`job` was dropped and the model refit. Cost: AUC moved from 0.7889 to 0.7881. **Removing an indefensible characteristic cost 0.0008 of AUC.** That tradeoff is the entire argument for how this industry builds models.

Final scorecard: 12 characteristics, all coefficients negative.

---

## Score scaling

```
Score  = Offset + Factor * ln(odds_good_to_bad)
Factor = PDO / ln(2)
Offset = base_score - Factor * ln(base_odds)
```

Calibrated to 600 points at 50:1 odds with 20 points to double the odds. Factor 28.85, offset 487.12. Scores range 427 to 626.

Points are allocated per attribute so they sum exactly to the applicant's score. Verified in the pipeline with an additivity check.

Sample of the `checking_status` rows from `outputs/scorecard_points.csv`:

| Attribute | Pop % | Bad rate | WOE | Points |
|---|---|---|---|---|
| A11 (negative balance) | 27.4% | 47.9% | -0.759 | 26.0 |
| A12 (0 to 200 DM) | 26.9% | 39.4% | -0.412 | 33.6 |
| A13 (200+ DM) | 6.3% | 27.3% | +0.114 | 45.1 |
| A14 (no account) | 39.4% | 11.6% | +1.176 | 68.5 |

That is a scorecard. Look up each of an applicant's attributes, add the points, get the score.

---

## Cutoff selection

Swept the threshold and costed each one using the dataset's 5:1 cost matrix.

| Cutoff | Approval rate | Bad rate of approved | Expected cost |
|---|---|---|---|
| 474 | 88.0% | 25.0% | 342 |
| 500 | 68.3% | 18.1% | 227 |
| 510 | 58.7% | 14.8% | 190 |
| **517** | **51.3%** | **10.4%** | **160** |
| 544 | 29.0% | 8.1% | 165 |
| 570 | 9.7% | 0.0% | 181 |

Minimum expected cost at 517: approve half the book, carry a 10.4 percent bad rate. Note the cost curve is U-shaped. Tightening past the optimum costs money because you start declining profitable good customers.

Cutoff choice is an economics question, not a statistics question.

---

## Drift monitoring

Simulated a new acquisition channel bringing in riskier applicants. The model is unchanged, the population is not.

| Comparison | Score PSI | Verdict |
|---|---|---|
| Train vs test | 0.022 | stable |
| Train vs shifted | 0.297 | significant, investigate |

Score-level PSI tells you *that* the population moved. Characteristic-level PSI tells you *which* input moved:

| Characteristic | PSI |
|---|---|
| age_years | 0.152 |
| savings_status | 0.142 |
| checking_status | 0.102 |
| housing | 0.059 |

The decomposition is the useful part. It turns "the score dropped" into "the score dropped because the incoming population is younger with thinner savings," which is a population change to accept rather than a model failure to rebuild.

---

## Adverse action reason codes

ECOA and Regulation B require a declined applicant to be told the specific principal reasons. Because the points are additive, the reasons fall straight out: compare points earned on each characteristic against the maximum available and report the largest shortfalls.

Lowest-scoring test applicant, score 427, declined at cutoff 517:

| Shortfall | Reason |
|---|---|
| 42.5 pts | Checking account balance negative |
| 29.6 pts | Savings under 100 DM |
| 25.6 pts | Loan term 15 to 30 months |
| 22.8 pts | Employed under 1 year |

This is the answer to "why not just use XGBoost." A gradient-boosted ensemble can produce a better AUC, but it cannot hand you that table.

---

## Outputs

| File | Contents |
|---|---|
| `run_log.txt` | Full annotated pipeline run |
| `iv_report.csv` | IV, bin count, verdict per characteristic |
| `coefficients.csv` | Final coefficients with first-pass comparison and sign review |
| `performance.csv` | AUC, Gini, KS train and test |
| `scorecard_points.csv` | The scorecard, 53 attribute rows with points |
| `gains_table.csv` | Score bands with bad rate, lift, capture |
| `cutoff_analysis.csv` | Threshold sweep with expected cost |
| `psi_report.csv` | Characteristic-level drift |
| `adverse_action_reasons.csv` | Reason codes for declined applicants |
| `diagnostics.png` | ROC, KS curve, score distributions, drift |
| `woe_patterns.png` | WOE by attribute for the top 6 characteristics |

---

## Notes on method

**Everything is fit on training data only.** Bin edges, WOE values, and the correlation screen are all derived from the 700 training records, then applied unchanged to the test set. Fitting bins on the full dataset is target leakage and is the most common error in scorecard tutorials.

**Monotonic coarse classing.** Numeric characteristics start at 10 quantile bins, then adjacent bins are merged until WOE moves in one consistent direction. Zig-zagging risk is usually noise and is not defensible in review.

**Haldane correction.** WOE uses a +0.5 adjustment on bin counts. Without it a bin with zero bads yields infinite WOE, which propagates into the coefficients.

**Rare level collapsing.** Categorical levels under 5 percent of the population are merged into OTHER. A WOE estimated on eight accounts will not replicate next quarter.

**Unseen test levels** map to WOE 0, portfolio-average risk, which is the neutral choice rather than a guess.

---

## Talking points

Three things here that most candidate projects do not have:

1. **The wrong-sign remediation.** You found an indefensible coefficient, removed the characteristic, and quantified what compliance cost you in AUC. That shows you understand model risk, not just model fitting.
2. **PSI decomposed to characteristic level.** Half the FICO intern job description is production model validation. Most applicants prepare only for model building.
3. **Adverse action reason codes.** It demonstrates you understand why this industry uses interpretable models, which is the thing that actually distinguishes credit scoring from generic ML.

Also worth saying out loud: dropping sex and nationality before modelling, rather than discovering a high IV on a protected characteristic and having to walk it back.
