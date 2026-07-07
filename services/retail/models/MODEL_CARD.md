# Retail Returns Intelligence model card

## Scope

This repository ships four models trained on UCI Online Retail II: a LightGBM return-risk classifier, an Isolation Forest excessive-returner detector, a KMeans customer segmentation model, and a substitute-product recommender.

## Data

UCI Online Retail II contains 1,067,371 transactions from December 2009 to December 2011. Returns are labeled by `C`-prefixed invoices. About 22.8% of rows lack `CustomerID`; the pipeline handles those rows rather than dropping them.

## Classifier

The return-risk classifier is LightGBM trained on point-in-time transaction and customer-history features. The split is temporal: train on 2009 through mid 2011, test on the last half of 2011. Results: ROC-AUC 0.992, PR-AUC 0.852, and top-decile precision 15.9% against a 1.83% base rate, an 8.7x lift. XGBoost is kept as a comparison model with PR-AUC 0.849.

The API returns a probability, a display tier, and the top SHAP factors for the scored transaction. The display tiers are High at 0.6 and Medium at 0.3. The balanced-precision operating threshold in `classifier_meta.json` is a separate notebook 04 result. The top-decile lift is a ranking metric, not a threshold.

## Anomaly detector

The excessive-returner detector is an Isolation Forest on customer-level behavior. It flags 294 customers at 5% contamination. This is the operational serial-returner signal used by the API profile response.

## Segmentation

The segmentation model is KMeans with `k=4`, trained on RFM and return-behavior features. The silhouette score is 0.238. Segment labels are assigned with a deterministic centroid rule in `scripts/build_api_artifacts.py`: Returner, Premium Loyal, At-Risk, and Healthy Browser.

One caveat matters: the Returner segment is roughly 18 wholesale-like accounts where return value concentrates. It is a policy bucket, not the fraud signal. The Isolation Forest flag is the broader excessive-returner signal.

## Recommender

The substitute recommender uses product-description embeddings plus ALS artifacts built offline. The published evaluation is Recall@10 0.093 and NDCG@10 0.046. For API serving, `scripts/build_api_artifacts.py` precomputes `models/invoice_substitutes.joblib` so Render can serve invoice lookups without loading recommender training libraries.

The current lookup rationale is content-only when served from the precomputed artifact. It reports content similarity, catalogue return rate, and whether the candidate appears in that customer's return history.

## Demo cases

`scripts/build_api_artifacts.py` also writes `models/demo_cases.joblib`, a curated pool of real purchase invoice lines for the public frontend. The cases are varied across Low, Medium, and High display tiers, all four customer segments, and behavior-anomaly examples. They are not a separate model and they do not change training metrics. They keep the live demo realistic by sending real historical invoice context through `/score` instead of letting visitors invent impossible customer, invoice, price, and stock-code combinations.

## Limitations

The data is from one UK online retailer and covers 2009 to 2011. Return labels are cancellations in the source data; the dataset does not separate refunds, store credit, and partial-line cancellations. The policy test in notebook 07 is an effect-size simulation, not a live randomized experiment.

## Files

- `api/predictor.py` loads the runtime artifacts and serves predictions.
- `scripts/build_api_artifacts.py` builds the API lookup tables.
- `models/classifier_meta.json` stores the classifier operating-point metadata.
- `notebooks/04_classification_model.ipynb`, `05_anomaly_detection.ipynb`, `06_segmentation.ipynb`, `08_backtesting.ipynb`, and `10_substitute_recommender.ipynb` hold the training and evaluation trail.
