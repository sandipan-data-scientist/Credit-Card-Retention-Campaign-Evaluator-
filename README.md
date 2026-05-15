# Credit Card Retention Campaign: A/B Testing and Predictive Analytics

---

## Project Overview

This project delivers a complete, end-to-end data science solution for evaluating the effectiveness of a financial institution's multi-arm credit card retention campaign. Using a dataset of 100,000 customer records, the project combines rigorous statistical hypothesis testing with machine learning to answer two fundamental business questions:

- Did the retention campaign actually work?
- Which customers should receive the next offer?

The solution covers the full data science lifecycle — exploratory analysis, feature engineering, statistical testing, predictive modeling, model deployment via a REST API, and a Streamlit-based analytics frontend.

---

# Problem Statement

A financial institution runs a portfolio of credit card customers across five segments:

- SME
- HNI
- Affluent
- Mass Market
- Student

Acquiring a new credit card customer costs an estimated five to seven times more than retaining an existing one. To combat churn, the marketing team launched a retention campaign that assigned customers to one of six offer types:

- Cashback
- Travel Rewards
- Low Interest
- Premium Lifestyle
- Dining Rewards
- Balance Transfer

Offers were delivered across five channels:

- Email
- SMS
- Branch
- Phone Call
- Social Media

The business needed evidence-based answers:

- Were customers who received high-engagement offers like Cashback genuinely more likely to remain active after six months?
- Or did they stay regardless of what they were offered?

This project answers those questions with statistical rigor and then builds a predictive scoring engine to make the next campaign smarter.

---

# Dataset

| Field | Type | Description |
|---|---|---|
| Account_ID | String | Unique customer identifier |
| Customer_Age | Integer | Customer age in years (range: 21–69) |
| Annual_Income_USD | Float | Declared annual income |
| Credit_Score | Integer | Credit bureau score (300–900) |
| Offer_Type | Categorical | Treatment arm — one of 6 offer types |
| Discount_Percent | Float | Incentive magnitude attached to the offer |
| Credit_Limit_USD | Float | Approved credit limit |
| Avg_Monthly_Spend_USD | Float | Average monthly card spend |
| Campaign_Cost_USD | Float | Bank's cost to run this customer's campaign |
| Profit_Margin_Percent | Float | Net margin from this customer |
| Campaign_Channel | Categorical | Delivery channel for the offer |
| Region | Categorical | Geographic region |
| Customer_Segment | Categorical | SME, HNI, Affluent, Mass Market, Student |
| Accepted_Offer | Binary (0/1) | Whether the customer accepted the offer |
| Retained_6_Months | Binary (0/1) | Whether the customer was still active at 6 months |
| Campaign_Date | Date | Date the campaign was executed |

The dataset contains:

- 100,000 rows
- No missing values
- No duplicate account IDs

Offer types are distributed near-equally at roughly 16.5–16.8% each, making this a well-balanced multi-arm experiment.

---

# Project Architecture

```text
Raw CSV Dataset
      |
      v
Data Inspection & Cleaning
      |
      v
Exploratory Data Analysis
      |
      v
Feature Engineering
      |
      +----------------------------+
      |                            |
      v                            v
Hypothesis Testing          Predictive Modeling
(Z-test, Chi-Square)        (Logistic Regression)
      |                            |
      v                            v
Statistical Conclusions     Serialized ML Pipeline
                                   |
                                   v
                           FastAPI REST Endpoint
                                   |
                                   v
                         Streamlit Frontend Dashboard
```

---

# Folder Structure

```text
project_testing_campaign_retention_evaluator_ab_testing_run/
|
|-- app/
|   |-- main.py
|       FastAPI application with /predict and /batch_predict endpoints
|
|-- models/
|   |-- offer_acceptance_pipeline.joblib
|       Serialized sklearn Pipeline (preprocessor + classifier)
|
|-- notebook/
|   |-- retention_campaign_analysis.ipynb
|       Full Jupyter Notebook — sequential, storytelling style
|
|-- streamlit_app.py
|       Multi-tab Streamlit frontend dashboard
|
|-- Dockerfile
|       Docker container definition
|
|-- requirements.txt
|       Python dependency list
|
|-- README.md
|       This file
```

---

# Installation

## Clone and Set Up the Environment

```bash
git clone https://github.com/your-org/project_testing_campaign_retention_evaluator_ab_testing_run.git

cd project_testing_campaign_retention_evaluator_ab_testing_run

python -m venv venv

source venv/bin/activate
# On Windows:
# venv\Scripts\activate

pip install -r requirements.txt
```

## Run the Jupyter Notebook

```bash
jupyter notebook notebook/retention_campaign_analysis.ipynb
```

---

# API Usage

## Start the FastAPI Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## Single Customer Prediction — POST `/predict`

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Customer_Age": 42,
    "Annual_Income_USD": 85000,
    "Credit_Score": 710,
    "Discount_Percent": 15.0,
    "Credit_Limit_USD": 25000,
    "Avg_Monthly_Spend_USD": 3200,
    "Profit_Margin_Percent": 18.5,
    "Spend_to_Income_Ratio": 0.4518,
    "Credit_Utilization": 0.128,
    "Is_High_Value": 1,
    "Month_Sin": 0.866,
    "Month_Cos": 0.5,
    "Offer_Type": "Cashback",
    "Campaign_Channel": "Email",
    "Region": "Asia Pacific",
    "Customer_Segment": "SME",
    "Income_Band": "60K-100K",
    "Offer_Discount_Group": "Medium",
    "Age_Group": "Gen X"
  }'
```

## Expected Response

```json
{
  "acceptance_probability": 0.7934,
  "propensity_tier": "Warm Lead",
  "recommended_action": "Email Campaign: Dining Rewards or Low Interest Offer"
}
```

---

## Batch Prediction — POST `/batch_predict`

Send a JSON array of customer objects to the same endpoint structure.

The response returns a predictions array with one entry per customer.

---

## Interactive API Documentation

FastAPI auto-generates Swagger UI documentation at:

```text
http://localhost:8000/docs
```

---

# Streamlit Dashboard

## Run the Dashboard

```bash
streamlit run streamlit_app.py --server.port 8501
```

---

## Dashboard Features

### Campaign Overview

Upload the campaign CSV and instantly see portfolio-level KPIs:

- Total customers
- Overall acceptance rate
- Overall retention rate
- Retention rate by offer type

### A/B Testing Results

Select any two offer types as treatment and control.

The tab runs the Z-test for proportions in real time and displays:

- Z-statistic
- p-value
- Retention lift
- SIGNIFICANT / NOT SIGNIFICANT verdict

### Predict Acceptance

Enter a single customer's profile through an input form.

The app calls the FastAPI backend and returns:

- Acceptance probability
- Propensity tier
- Recommended next action

### Segment Intelligence

Displays a heatmap of acceptance rates across:

- Customer segment
- Offer type combinations

This helps identify which offer works best for which segment.

### Batch Scoring

Upload a CSV of prospective campaign customers.

The app:

- Sends records to the batch prediction endpoint
- Returns a scored output file for the Sales team

---

# Docker Execution

## Build the Container

```bash
docker build -t retention-campaign-api .
```

## Run the Container

```bash
docker run -p 8000:8000 retention-campaign-api
```

The API will be accessible at:

```text
http://localhost:8000
```

The Streamlit app runs separately and connects to the API via the configurable endpoint field in the sidebar.

---

# Statistical Testing Explanation

## Primary Test: Two-Sample Z-Test for Proportions

The core business question:

> Did a specific offer type improve retention compared to a baseline?

This is answered using a Z-test for proportions.

The test is appropriate because:

- The outcome variable is binary (`Retained_6_Months`)
- Group sizes exceed 30
- The groups are independent

### Null Hypothesis (H0)

The retention rate of the treatment group equals the retention rate of the control group.

### Alternative Hypothesis (H1)

The treatment group has a statistically higher retention rate than the control group.

---

## Test Statistic

```text
         p_hat_treatment - p_hat_control
Z = -----------------------------------------------
    sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
```

Where:

- `p_pool` is the combined retention rate across both groups
- The null hypothesis assumes both groups come from the same population

A p-value below `0.05` implies rejection of H0.

---

## Confidence Intervals

The project also reports a 95% confidence interval for the retention rate difference.

If the interval excludes zero:

- The effect is statistically meaningful
- The direction and magnitude of the effect are likely real

---

## Secondary Test: Chi-Square Test of Independence

The Chi-Square test evaluates whether:

- Offer type
- Retention outcome

are statistically independent across all six offer arms simultaneously.

A significant result (`p < 0.05`) implies that offer type matters to retention overall.

---

## Why Statistical Testing Before Modeling?

Without hypothesis testing, observed campaign effects may simply reflect natural customer behavior.

Statistical testing:

- Establishes an explicit baseline
- Distinguishes real treatment effects from random variation
- Prevents misleading conclusions

---

# ML Workflow Explanation

## Target Variable

The model predicts:

```text
Accepted_Offer
```

This is operationally valuable because it is known before campaign execution.

Benefits:

- Prioritize outreach
- Allocate campaign budget efficiently
- Improve targeting precision

---

# Feature Engineering

## Spend_to_Income_Ratio

Annual card spend divided by annual income.

Measures how central the card is to the customer's financial life.

---

## Credit_Utilization

Average monthly spend divided by credit limit.

Acts as a proxy for customer engagement and available credit headroom.

---

## Is_High_Value

Binary flag for customers above the 75th percentile in:

- Income
- Monthly spend

These customers generate disproportionate revenue.

---

## Income_Band

Bucketed income tiers:

- Below 30K
- 30K–60K
- 60K–100K
- 100K–200K
- Above 200K

---

## Offer_Discount_Group

Categorized incentive size:

- Low (0–10%)
- Medium (10–20%)
- High (20%+)

---

## Age_Group

Generational cohorts:

- Gen Z
- Millennial
- Gen X
- Boomer

---

## Month_Sin / Month_Cos

Cyclic encoding of campaign month.

This preserves calendar continuity:

- December is adjacent to January
- Avoids linear month assumptions

---

# Preprocessing

## Numeric Features

Standardized using `StandardScaler`:

- Zero mean
- Unit variance

This prevents high-magnitude features from dominating.

---

## Categorical Features

Encoded using One-Hot Encoding with the first category dropped to avoid multicollinearity.

---

## Leakage Prevention

All preprocessing is:

- Fit only on the training set
- Applied to the test set afterward

This prevents data leakage.

---

# Model Choice: Logistic Regression

Logistic Regression was chosen intentionally due to:

- Explainability
- Probability calibration
- Regulatory interpretability requirements

Benefits:

- Produces calibrated probabilities
- Coefficients directly explain feature impact
- Transparent decision logic

A positive coefficient increases offer acceptance probability, while a negative coefficient decreases it.

---

# Model Evaluation

The model is evaluated using:

- Stratified 20% holdout validation
- 5-fold stratified cross-validation

This ensures stable and reliable performance estimates.

---

## Business Metric Prioritization

Recall is weighted more heavily than Precision because:

- Missing a customer who would accept = lost retention opportunity
- Sending an offer to a customer who declines = relatively low campaign cost

---

# Recommendation Engine

The scoring output is translated into actionable customer tiers.

| Tier | Probability Range | Recommended Action |
|---|---|---|
| Hot Lead | >= 85% | Priority RM-led outreach with premium offer |
| Warm Lead | 70–85% | Targeted email with segment-relevant offer |
| Cold Lead | 50–70% | Discount-heavy SMS nudge |
| Not Likely | < 50% | Exclude from current cycle; revisit next quarter |

High-value customers in the Hot Lead tier receive escalated Relationship Manager outreach.

---

# Business Interpretation of Results

The campaign demonstrates:

- 76.8% overall offer acceptance rate
- 79.9% six-month retention rate

The predictive model enables a transition from:

```text
Mass Broadcast Campaigns
            ->
Precision Targeting
```

Benefits include:

- Reduced campaign cost
- Better retention efficiency
- Personalized customer treatment

The segment-offer heatmap shows that no single offer dominates across all customer groups.

Different segments respond differently:

- SME customers
- HNI customers
- Mass Market customers

This validates the need for model-driven personalization.

---

# Future Improvements

## Upgrade the Model

Replace Logistic Regression with:

- XGBoost
- LightGBM

These models better capture nonlinear interactions.

---

## Add SHAP Explainability

Use SHAP values to explain:

- Why each prediction occurred
- Which features influenced decisions

Useful for:

- Auditability
- Sales transparency
- Regulatory trust

---

## Customer Lifetime Value Weighting

Incorporate CLV into training weights so the model prioritizes high-revenue customers more aggressively.

---

## Real-Time Feature Store Integration

Replace static CSV ingestion with real-time feature serving using:

- Feast
- Tecton
- Databricks Feature Store

---

## Experiment Monitoring

Build live A/B test monitoring with:

- Real-time p-values
- Confidence intervals
- Sample tracking
- Early stopping alerts

---

## Multi-Touch Attribution

Customers interact through multiple channels before making decisions.

Future versions can model attribution across:

- Email
- SMS
- Branch
- Phone Call

to identify the highest ROI combinations.

---

# Requirements

```text
pandas==2.2.2
numpy==1.26.4
scipy==1.13.0
scikit-learn==1.5.0
joblib==1.4.2
matplotlib==3.9.0
seaborn==0.13.2
fastapi==0.111.0
uvicorn[standard]==0.30.1
pydantic==2.7.3
streamlit==1.35.0
requests==2.32.3
python-dateutil==2.9.0
```

---
