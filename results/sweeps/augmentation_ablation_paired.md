Paired by seed against **tinyvgg|full** (test_acc; positive = better than baseline)

| group | n | Δ mean | 95% CI of Δ | p (paired t) | wins / losses |
|---|---|---|---|---|---|
| tinyvgg|no_crop | 5 | +0.0095 | [+0.0027, +0.0162] | 0.018 | 5 / 0 |
| tinyvgg|no_rotation | 5 | +0.0055 | [-0.0047, +0.0157] | 0.211 | 4 / 1 |
| tinyvgg|no_flip | 5 | +0.0020 | [-0.0088, +0.0129] | 0.629 | 3 / 2 |
| tinyvgg|no_mixup_cutmix | 5 | -0.0009 | [-0.0144, +0.0126] | 0.863 | 3 / 2 |
| tinyvgg|no_aug | 5 | -0.0065 | [-0.0160, +0.0029] | 0.127 | 1 / 4 |
