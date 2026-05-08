# Insurance-Charge-Prediction---Linear-Regression

This was one of my early machine learning projects where I worked on predicting medical insurance charges using Linear Regression. The dataset has information on 1,338 patients and the goal was to figure out what factors drive up insurance costs and build a model to predict them.

## What's the problem?
A fictional insurance company wants to predict how much they should charge a customer based on their personal details like age, BMI whether they smoke and the region they live in. Instead of using flat rates, the idea is to use data to come up with more accurate and fair pricing.


## About the Dataset
1,338 records with 7 columns:
- `age` — age of the patient
- `sex` — male or female
- `bmi` — body mass index
- `children` — number of dependents
- `smoker` — yes or no
- `region` — northeast, northwest, southeast, southwest
- `charges` — the actual insurance cost (this is what we're predicting)

## What I did
**EDA first** — I explored each feature individually and then looked at how they relate to charges. Plotted box plots, scatter plots and a correlation heatmap. Smoking stood out immediately as the biggest factor.

**Log transformation** — The charges column was heavily right-skewed (skewness of 1.52), so I applied a log transformation which brought it down to -0.09 and made the distribution much more normal.

**Encoding** — Converted categorical variables (sex, smoker, region) into dummy variables using one-hot encoding with drop_first=True to avoid multicollinearity.

**Model** — Built an OLS Linear Regression model using statsmodels on a 70/30 train-test split.

## Results

| Metric              | Value |
| R-Squared           | 0.759 |
| Adjusted R-Squared  | 0.757 |
| RMSE                | ~$9,409 |

The model explains about 76% of the variance in charges which is decent for a simple linear regression on this kind of data.

The smoker coefficient came out to 1.557 on log-transformed charges by far the largest of any feature, confirming that smoking is the dominant cost driver.


## Files
```
├── Insurance_Charges_Prediction.ipynb   # full notebook with code and outputs
├── Insurance_Charges_Prediction_data.csv             # dataset
└── README.md
```

## Libraries used

Python · Pandas · NumPy · Statsmodels · Scikit-learn · Matplotlib · Seaborn
