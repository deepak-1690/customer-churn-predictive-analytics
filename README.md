Customer Churn Prediction (Python)

A complete, reproducible predictive-analytics project: it builds, validates and compares several models that estimate which subscription customers are likely to cancel in the next 90 days, then turns the best model into a decision rule a retention team can use.

The full strategy write-up (use-case, model selection logic, validation plan, bias discussion, six-week timeline) is in docs/Predictive_Analytics_Strategy.docx.

What this project does
Frames churn as a binary classification problem
Compares a baseline, logistic regression, a decision tree, a random forest and gradient boosting
Uses stratified 5-fold cross-validation and GridSearchCV for tuning
Keeps a 20% hold-out test set that is only touched once, at the end
Picks the decision threshold from business costs (a missed churner costs more than a wasted offer)
Runs a subgroup check and permutation importance to look at fairness and drivers
Ships with unit tests and a scoring script for new customers

About the data: the repository generates a synthetic dataset (7,000 customers) so anyone can run it without private data. The numbers below show that the pipeline works, not real business performance. To use real data, replace data/customer_churn.csv and adjust the column lists in src/config.py.

Project structure
churn-predictive-analytics/
├── README.md
├── requirements.txt
├── LICENSE
├── docs/
│   └── Predictive_Analytics_Strategy.docx   # strategy document
├── src/
│   ├── config.py          # paths, seeds, feature lists, error costs
│   ├── generate_data.py   # creates the synthetic dataset
│   ├── preprocess.py      # imputation, scaling, one-hot encoding
│   ├── models.py          # candidate models + tuning grids
│   ├── train.py           # split, tune, compare, select, evaluate, save
│   ├── evaluate.py        # metrics, plots, threshold search, subgroup report
│   └── predict.py         # score new customers with the saved model
├── tests/
│   └── test_pipeline.py
├── reports/
│   ├── metrics.csv
│   ├── feature_importance.csv
│   └── figures/           # model comparison, ROC / PR / confusion matrix, calibration
├── data/                  # generated CSV lands here (git-ignored)
└── models/                # saved model lands here (git-ignored)
Getting started

Requires Python 3.10 or newer.

bash
# 1. clone and enter the folder
git clone https://github.com/<your-username>/churn-predictive-analytics.git
cd churn-predictive-analytics

# 2. (optional) virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. install dependencies
pip install -r requirements.txt

# 4. create the dataset, run the tests, train
python -m src.generate_data
python -m pytest -q
python -m src.train

Training takes roughly one to two minutes on a laptop. Results are printed to the console and saved in reports/.

Score new customers
bash
python -m src.predict path/to/new_customers.csv

The CSV needs the same feature columns as the training data (see src/config.py). The output adds churn_probability and flagged_for_retention.

How the workflow is set up
Step	What happens	Where
1	Load data, stratified 80/20 split	src/train.py
2	Preprocessing inside a scikit-learn Pipeline (so it is fit on training folds only, no leakage)	src/preprocess.py
3	Tune each model with 5-fold stratified CV, scoring on ROC-AUC	src/models.py, src/train.py
4	Select the simplest model within one standard deviation of the best CV score	src/train.py
5	Choose a cost-based threshold from out-of-fold predictions	src/evaluate.py
6	One final evaluation on the hold-out set	src/train.py
7	Permutation importance, subgroup check, save model	src/train.py

The threshold costs live in src/config.py (COST_FALSE_NEGATIVE = 150, COST_FALSE_POSITIVE = 50). Change them to match your own economics and re-run.

Results (synthetic data)

Cross-validation uses the 5,600-row training set. Test numbers use the 1,400-row hold-out set. The selected model (logistic regression) uses the cost-optimal threshold of 0.38; the others are shown at 0.50.

Model	CV ROC-AUC	Test ROC-AUC	Recall	Precision	F1
Baseline (majority)	0.500	0.500	0.000	0.000	0.000
Logistic regression	0.763 ± 0.012	0.735	0.819	0.502	0.622
Decision tree	0.737 ± 0.012	0.711	0.609	0.563	0.585
Random forest	0.757 ± 0.010	0.722	0.638	0.565	0.600
Gradient boosting	0.758 ± 0.009	0.725	0.683	0.571	0.622

Show Image Show Image

Takeaways

The three real models are statistically close, so the selection rule picks the simplest one.
Contract type is by far the strongest driver, followed by monthly charges, support calls and tenure. These effects were built into the synthetic data, so re-check them on real data.
Performance is uneven across contract types: recall is about 0.96 for month-to-month customers but only 0.19 for two-year customers. This is the kind of gap the subgroup report is meant to catch, and the strategy document lists mitigation options.
Limitations
Results come from synthetic data and must not be quoted as business results.
Tuning grids are small to keep runtime short; a real run should search wider and use repeated CV.
The model predicts risk, not who will respond to a retention offer (that would need uplift modelling).
Sensitive or proxy features (age, senior status) should be reviewed before any real deployment.
Ideas for next steps
Calibrate probabilities (CalibratedClassifierCV) and weight customers by lifetime value
Add a temporal train/test split once real timestamps are available
Try survival analysis to predict when a customer will leave
Add monthly monitoring for drift in AUC and input distributions
