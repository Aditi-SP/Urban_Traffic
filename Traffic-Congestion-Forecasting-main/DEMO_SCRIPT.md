# 2–3 minute demo script (for judges)

Start the app first: `streamlit run app.py` (see README). Keep the browser on the **Overview** tab.

**0:00 – Problem and data (Overview, Data & Wrangling)**
"We forecast next-day congestion for 16 Bangalore roads using a public Kaggle dataset of 8,936 daily records.
The data has gaps, so we use each road's last 7 observations. We checked it with PySpark: no missing values or
duplicates." *(Data & Wrangling tab: coverage per road, split table.)*

**0:30 – Fair evaluation (Model Results)**
"We split by time – train to Oct 2023, test from March to August 2024 – and scored both models and two simple
baselines on exactly the same 856 test rows. An earlier version showed R² = 0.9995; that was target leakage, which we
found and removed."
Point at the metrics table: "LSTM and XGBoost reach R² of about 0.2 – but so does simply predicting each road's
average. **So neither model beats the baseline.** That is the honest result: next-day congestion barely depends on
yesterday in this dataset." *(Exploratory Analysis tab: congestion is explained by same-day volume, which is unknown
the day before.)*

**1:15 – Live Prediction Demo (🎯 tab, Mode 1)**
Pick a road and a date → **Predict**. "These are real historical windows from the test period, not live traffic.
Both trained models and both baselines forecast tomorrow; the hollow circle is what actually happened."
Switch to *What-if scenario* → *Heavy congestion* → **Predict** → "I edit today's conditions; the forecast moves only
slightly and not always in the expected direction – evidence the models learned little day-to-day signal."
Show the prediction log, then **Reset demo**.

**2:15 – Real-time mode and honesty (Mode 2, Limitations)**
"We investigated TomTom's traffic API. It gives only a current speed snapshot – no volume, capacity, incidents,
weather or history – so it cannot feed these models, and we don't pretend otherwise." *(Show the coverage table.)*
"Limitations: daily data (no rush-hour analysis), one split, no uncertainty intervals, PySpark runs locally."

**Likely questions**
* *Why not 'LSTM is better'?* Test RMSE 19.93 vs 20.07 is a negligible difference, and both ≈ road-mean baseline (19.85).
* *Is the traffic live?* No. Mode 1 uses historical data; Mode 2 is an unverified interface that cannot predict.
* *Is Spark big data here?* No: local mode on a small file, to demonstrate the workflow.
