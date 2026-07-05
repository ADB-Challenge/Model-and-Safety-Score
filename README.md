# Speed Safety Score v1
--------------------------------------------------------------------------
For SpeedSafetyScorev1, the author built a road safety causal analysis pipeline utilizing the DoWhy causal inference framework (employing Directed Acyclic Graphs and Backdoor Adjustment via Linear Regression) to isolate and estimate the causal impact of the 85th percentile speed on fatal accident rates using the ADB Thailand dataset. As a result, the model calculated normalized impact coefficients across different road classes and land uses—identifying primary/secondary urban and secondary rural segments as high-priority areas—and successfully categorized 11,544 road segments into a four-tier safety prioritization matrix exported to safety_score_1.geojson


### 1. Model Results & Interpretation


| Road Group | Model Coeff | Impact per 1km/h F85 reduction (%) | Priority |
| :--- | :--- | :--- | :--- |
| primary_URBAN | 0.11288 | 0.5484 | High |
| primary_RURAL | -0.19929 | -0.7057 | Low |
| secondary_URBAN | 0.06769 | 0.3206 | High |
| secondary_RURAL | 0.15104 | 0.8031 | High |
| motorway_URBAN | -0.13833 | -0.5874 | Low |
| motorway_RURAL | 0.00000 | 0.0000 | Low |
| trunk_URBAN | 0.05453 | 0.2722 | Medium |
| trunk_RURAL | 0.02331 | 0.0931 | Medium |



# Speed Safety Score v2
--------------------------------------------------------------------------
## 1. Non-linear model to score safety risk of the roads in Thailand
shem_nonlinear_branch folder contains a Jupyter notebook that performs non-linear model training.

### Model Performance & Evaluation

Cross-validation (CV) results for predicting road safety risk on two target metrics:

*   **Total Accidents** (Over-dispersed, 0% zeros):
    *   **Best Model**: **Model D+W** (Gradient-Boosted Trees + spatial lag) yields the lowest CV MAE (**2.124**) and Poisson Deviance (**2.710**), and the highest Top-5% precision (**0.413**).
    *   **Interpretable Model**: **Model B_INT** (Negative Binomial + interactions) performs competitively (MAE: 2.234) and is retained for its coefficient interpretability.
*   **Deaths** (Moderately over-dispersed, 74.8% zeros):
    *   **Best Model**: **Model D+W** achieves the lowest CV MAE (**0.552**) and Poisson Deviance (**1.207**).

#### Key Pipeline Takeaways
1. **Traffic Exposure Matters**: Model A (no traffic volume) is dominated on all metrics; `log_volume` is kept as a covariate in all production models.
2. **Interpretable vs. ML Tradeoff**: Model B_INT's interaction terms improve accident prediction and offer named coefficients/p-values, making it a valuable secondary signal alongside GBT.
3. **Spatial Lag Boosts Performance**: Incorporating spatial correlation (**Model D+W**) consistently reduces MAE/deviance. It is the primary score used to rank and prioritize roads.

### Spatial Autocorrelation & The "+W" Step

Road segments are physically connected, meaning risk clusters along corridors rather than being independent. Moran’s I tests on non-spatial residuals confirm significant positive spatial autocorrelation:
*   **Total Accidents residuals**: $I = 0.126$ to $0.180$ ($p \le 0.0002$)
*   **Death residuals**: $I = 0.070$ ($p < 0.04$)

To capture this corridor-level risk, **Model D+W** introduces a spatially lagged neighbour signal (`W_log_target`), representing the row-standardized mean of a segment's neighbours' log-transformed outcomes. This allows:
1. **Prioritization**: High-risk corridor identification for known road networks.
2. **Cold-Start Scoring**: Borrowing risk signal from connected, known roads to score new ("unmerged") road segments in Part 3.


# Street View Analysis
----------------------------------------------------------

For street_view_analysis, the author developed a computer vision and safety scoring pipeline that matched 2024 accident points to road segments in the ADB Thailand dataset, sampled 1,000 segments with multiple accidents to fetch 2,000 Google Street View images, and analyzed them using YOLO-World object detection and Mask2Former scene segmentation. As a result, the pipeline computed per-segment Speed Safety Scores and generated an observational correlation bubble chart visualizing the relationships between operating speed, detected road objects, and accident rates.

