
import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns
import io

st.set_page_config(
    page_title="Retention Campaign Analytics",
    layout="wide",
    page_icon="card"
)

st.title("Credit Card Retention Campaign Evaluator")
st.markdown(
    "A/B Testing Analysis | Predictive Offer Acceptance | Segment Intelligence"
)

API_URL = st.sidebar.text_input("API Endpoint", value="http://localhost:8000")

tabs = st.tabs([
    "Campaign Overview",
    "A/B Testing Results",
    "Predict Acceptance",
    "Segment Intelligence",
    "Batch Scoring"
])

with tabs[0]:
    st.header("Campaign Performance Overview")
    uploaded_file = st.file_uploader("Upload Campaign CSV Dataset", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Customers", f"{len(df):,}")
        col2.metric("Acceptance Rate", f"{df['Accepted_Offer'].mean()*100:.1f}%")
        col3.metric("Retention Rate", f"{df['Retained_6_Months'].mean()*100:.1f}%")
        col4.metric("Offer Types", df["Offer_Type"].nunique())

        st.subheader("Retention Rate by Offer Type")
        offer_ret = (
            df.groupby("Offer_Type")["Retained_6_Months"].mean().mul(100)
            .sort_values(ascending=False).reset_index()
        )
        offer_ret.columns = ["Offer_Type", "Retention_Rate"]
        st.bar_chart(offer_ret.set_index("Offer_Type"))

with tabs[1]:
    st.header("A/B Testing Results")
    st.markdown(
        """
        This tab shows statistical significance testing between any two offer groups.
        Select your treatment and control arm to run the Z-test for proportions.
        """
    )
    if uploaded_file is not None:
        offer_types = df["Offer_Type"].unique().tolist()
        col_a, col_b = st.columns(2)
        group_a = col_a.selectbox("Treatment Group", offer_types, index=0)
        group_b = col_b.selectbox("Control Group", offer_types, index=len(offer_types)-1)

        if st.button("Run Hypothesis Test"):
            from scipy.stats import norm as scipy_norm

            data_a = df[df["Offer_Type"] == group_a]["Retained_6_Months"]
            data_b = df[df["Offer_Type"] == group_b]["Retained_6_Months"]
            n_a, n_b = len(data_a), len(data_b)
            p_a, p_b = data_a.mean(), data_b.mean()
            p_pool = (data_a.sum() + data_b.sum()) / (n_a + n_b)
            se = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
            z_stat = (p_a - p_b) / se
            p_value = 1 - scipy_norm.cdf(z_stat)
            diff = p_a - p_b

            res_col1, res_col2, res_col3 = st.columns(3)
            res_col1.metric(f"{group_a} Retention", f"{p_a*100:.2f}%")
            res_col2.metric(f"{group_b} Retention", f"{p_b*100:.2f}%")
            res_col3.metric("Retention Lift", f"{diff*100:.2f} pp")

            st.metric("Z-Statistic", f"{z_stat:.4f}")
            st.metric("P-Value", f"{p_value:.6f}")

            if p_value < 0.05:
                st.success(
                    f"STATISTICALLY SIGNIFICANT: {group_a} outperforms {group_b} "
                    f"with p = {p_value:.4f} (< 0.05)"
                )
            else:
                st.warning(
                    f"NOT STATISTICALLY SIGNIFICANT: Observed difference likely due to "
                    f"random variation (p = {p_value:.4f})"
                )

with tabs[2]:
    st.header("Predict Offer Acceptance Probability")
    st.markdown("Enter customer profile to get real-time acceptance probability from the ML API.")

    c1, c2, c3 = st.columns(3)
    age = c1.number_input("Customer Age", 21, 69, 40)
    income = c2.number_input("Annual Income (USD)", 15000, 200000, 75000)
    credit_score = c3.number_input("Credit Score", 300, 900, 650)

    c4, c5, c6 = st.columns(3)
    credit_limit = c4.number_input("Credit Limit (USD)", 1000, 100000, 20000)
    avg_spend = c5.number_input("Avg Monthly Spend (USD)", 100, 50000, 2000)
    discount = c6.number_input("Discount Percent", 0.0, 50.0, 10.0)

    c7, c8, c9 = st.columns(3)
    offer_type = c7.selectbox("Offer Type", ["Cashback", "Travel Rewards", "Low Interest",
                                              "Premium Lifestyle", "Dining Rewards", "Balance Transfer"])
    channel = c8.selectbox("Campaign Channel", ["Email", "SMS", "Branch", "Phone Call", "Social Media"])
    segment = c9.selectbox("Customer Segment", ["SME", "HNI", "Mass Market", "Student", "Affluent"])

    region = st.selectbox("Region", ["Asia Pacific", "North America", "Europe",
                                      "Middle East", "Latin America"])
    profit_margin = st.slider("Profit Margin %", 0.0, 50.0, 15.0)

    if st.button("Predict Acceptance Probability"):
        spend_income_ratio = (avg_spend * 12) / income
        credit_utilization = avg_spend / credit_limit
        is_high_value = 1 if (income >= 95000 and avg_spend >= 3000) else 0
        discount_group = "Low" if discount <= 10 else "Medium" if discount <= 20 else "High"
        age_group = (
            "Gen Z" if age < 28 else
            "Millennial" if age < 44 else
            "Gen X" if age < 60 else "Boomer"
        )
        income_band = (
            "Below 30K" if income < 30000 else
            "30K-60K" if income < 60000 else
            "60K-100K" if income < 100000 else
            "100K-200K" if income < 200000 else "Above 200K"
        )
        import datetime
        month = datetime.datetime.now().month
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)

        payload = {
            "Customer_Age": age, "Annual_Income_USD": income, "Credit_Score": credit_score,
            "Discount_Percent": discount, "Credit_Limit_USD": credit_limit,
            "Avg_Monthly_Spend_USD": avg_spend, "Profit_Margin_Percent": profit_margin,
            "Spend_to_Income_Ratio": spend_income_ratio, "Credit_Utilization": credit_utilization,
            "Is_High_Value": is_high_value, "Month_Sin": month_sin, "Month_Cos": month_cos,
            "Offer_Type": offer_type, "Campaign_Channel": channel, "Region": region,
            "Customer_Segment": segment, "Income_Band": income_band,
            "Offer_Discount_Group": discount_group, "Age_Group": age_group
        }
        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
            result = response.json()
            st.metric("Acceptance Probability", f"{result['acceptance_probability']*100:.1f}%")
            st.info(f"Propensity Tier: {result['propensity_tier']}")
            st.success(f"Recommended Action: {result['recommended_action']}")
        except Exception as e:
            st.error(f"Could not reach API: {e}. Make sure the FastAPI server is running.")

with tabs[3]:
    st.header("Segment Intelligence")
    if uploaded_file is not None:
        pivot = (
            df.pivot_table(
                values="Accepted_Offer",
                index="Customer_Segment",
                columns="Offer_Type",
                aggfunc="mean"
            ) * 100
        )
        st.subheader("Acceptance Rate (%) by Segment and Offer Type")
        st.dataframe(pivot.style.background_gradient(cmap="YlOrRd", axis=None).format("{:.1f}"))

with tabs[4]:
    st.header("Batch Customer Scoring")
    batch_file = st.file_uploader("Upload Batch Customer File (CSV)", type="csv",
                                   key="batch_uploader")
    if batch_file is not None:
        st.info("Batch scoring requires the API to be running. Ensure models are loaded.")
        st.markdown("Batch file would be sent to POST /batch_predict endpoint.")
