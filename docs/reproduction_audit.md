# Formula and preprocessing audit

## Equation (11): structural probability

The manuscript defines `p_struct = S_positive / (S_negative + S_positive + 1e-12)`. Earlier repository revisions normalized the two log supports with softmax, omitting the additive stabilizer. These formulas differ when both supports are close to or smaller than `1e-12`. The implementation now evaluates the paper's expression in log space using `logaddexp`, which preserves the stabilizer without prematurely underflowing the two supports.

Before retraining, the old seed-0 checkpoints were evaluated both ways on their original held-out test splits. The largest structural-probability differences were approximately 0.000081 for GMSC, 0.00000054 for Bank and 0.02778 for Default; 2, 0 and 223 samples respectively differed by more than 1e-6. This is an impact diagnostic on *old* weights, not a comparison of retrained model performance. Results produced with the omission remain in Git history at commit `603d00f77de145c5862963054d027c8cd370d064`; the current README reports retrained results.

Independent NumPy reference checks cover Gaussian density, Bhattacharyya overlap, reliability, structurally stabilized probability, gate and final probability. A separate extreme-small-support test checks the denominator explicitly.

## Default feature count

The paper reports 23 original and 29 encoded input features. On the UCI source dataset, `SEX` has 2 values, `EDUCATION` has 7 and `MARRIAGE` has 4. One-hot encoding these three columns with all observed categories produces `20 + 2 + 7 + 4 = 33` features. An alternative that maps `EDUCATION` 0, 5, 6 into 4 and `MARRIAGE` 0 into 3 gives `20 + 2 + 4 + 3 = 29` features. The label split remains fixed and preprocessing is fitted on the training portion only. This mapping is a documented hypothesis, not evidence of the authors' actual preprocessing. Both encodings are retained and evaluated; do not select an encoding by looking at test performance.
