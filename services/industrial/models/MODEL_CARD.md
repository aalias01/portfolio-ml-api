# Model card

## Question

This model predicts near-term machine failure from five operating inputs. It is a binary classifier, separate from the CMAPSS turbofan project that estimates remaining useful life.

## Data

The data is AI4I 2020 Predictive Maintenance: 10,000 machine records with a 3.4% failure rate. Inputs are air temperature, process temperature, rotational speed, torque, and tool wear. The API derives five more features from those inputs: process minus air temperature, mechanical power, wear per unit speed, torque relative to wear, and a high-load flag.

## Model

The deployed artifact is XGBoost trained with SMOTE and class weighting. The committed threshold is 0.775, selected by a modeled cost sweep.

## Economics

The sweep prices a missed failure at $50,000 and a false alarm at $2,000. At threshold 0.775, the held-out test set has 62 of 68 failures caught, 6 missed, and 83 false alarms, for $466,000 modeled total cost.

## Metrics

| Metric | Value |
|---|---:|
| PR-AUC | 0.841 |
| ROC-AUC | 0.979 |
| Recall at 0.775 | 91.2% |
| Random forest PR-AUC | 0.820 |
| Logistic regression PR-AUC | 0.455 |

## Limits

The costs are modeled assumptions, not customer measurements. AI4I is a single synthetic dataset. Random-failure rows add irreducible noise, so misses are expected and should be shown plainly.
