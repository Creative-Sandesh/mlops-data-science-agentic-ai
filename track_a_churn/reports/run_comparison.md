# Track A - MLflow run comparison (held-out test split, threshold 0.5)

Sorted by F1 then ROC-AUC.

| run         |   accuracy |   precision |   recall |     f1 |   roc_auc |   fit_seconds |
|:------------|-----------:|------------:|---------:|-------:|----------:|--------------:|
| logreg_C1.0 |     0.7627 |      0.5379 |   0.8534 | 0.6599 |    0.8614 |        0.025  |
| logreg_C0.1 |     0.7617 |      0.5366 |   0.8534 | 0.6589 |    0.8608 |        0.05   |
| rf_depth12  |     0.7708 |      0.5505 |   0.8195 | 0.6586 |    0.8598 |        0.8195 |
| rf_depth5   |     0.7505 |      0.5227 |   0.8647 | 0.6516 |    0.8589 |        0.2796 |
| xgb_depth3  |     0.7606 |      0.5366 |   0.8271 | 0.6509 |    0.8615 |        0.4375 |
| xgb_depth6  |     0.7677 |      0.5546 |   0.7068 | 0.6215 |    0.8362 |        0.3382 |

## Hyperparameters

| run         | model_family        | C   | penalty   | n_estimators   | max_depth   | learning_rate   | min_samples_leaf   | class_weight   | scale_pos_weight   |
|:------------|:--------------------|:----|:----------|:---------------|:------------|:----------------|:-------------------|:---------------|:-------------------|
| logreg_C1.0 | logistic_regression | 1.0 | l2        | -              | -           | -               | -                  | balanced       | -                  |
| logreg_C0.1 | logistic_regression | 0.1 | l2        | -              | -           | -               | -                  | balanced       | -                  |
| rf_depth12  | random_forest       | -   | -         | 400            | 12          | -               | 5                  | balanced       | -                  |
| rf_depth5   | random_forest       | -   | -         | 200            | 5           | -               | 5                  | balanced       | -                  |
| xgb_depth3  | xgboost             | -   | -         | 200            | 3           | 0.05            | -                  | -              | 2.77               |
| xgb_depth6  | xgboost             | -   | -         | 300            | 6           | 0.1             | -                  | -              | 2.77               |
