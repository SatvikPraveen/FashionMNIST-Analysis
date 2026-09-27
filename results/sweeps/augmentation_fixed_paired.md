Paired by seed against **tinyvgg|full** (test_acc; positive = better than baseline)

| group | n | Δ mean | 95% CI of Δ | p (paired t) | wins / losses |
|---|---|---|---|---|---|
| tinyvgg|no_crop | 5 | +0.0100 | [-0.0004, +0.0204] | 0.055 | 4 / 1 |
| tinyvgg|no_rotation | 5 | +0.0062 | [-0.0012, +0.0135] | 0.080 | 5 / 0 |
| tinyvgg|no_mixup_cutmix | 5 | -0.0006 | [-0.0119, +0.0108] | 0.898 | 2 / 3 |
| tinyvgg|no_flip | 5 | -0.0029 | [-0.0148, +0.0089] | 0.528 | 2 / 3 |
| tinyvgg|legacy | 5 | -0.0036 | [-0.0105, +0.0034] | 0.228 | 1 / 4 |
| tinyvgg|no_aug | 5 | -0.0094 | [-0.0204, +0.0016] | 0.077 | 1 / 4 |
