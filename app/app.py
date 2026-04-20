"""
Author: Denz'l Chapman  
Dataset: Industry Market Cap Dataset
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Market Cap Prediction Tool", 
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

@st.cache_resource  
def load_models():
    """Load all saved models and artifacts."""
   
    base_path = Path(__file__).parent.parent / "models"

    models = {}

    try:
        
        models['regression_model'] = joblib.load(base_path / "regression_model.pkl")
        models['regression_scaler'] = joblib.load(base_path / "regression_scaler.pkl")
        models['regression_features'] = joblib.load(base_path / "regression_features.pkl")

        
        models['classification_model'] = joblib.load(base_path / "classification_model.pkl")
        models['classification_scaler'] = joblib.load(base_path / "classification_scaler.pkl")
        models['label_encoder'] = joblib.load(base_path / "label_encoder.pkl")
        models['classification_features'] = joblib.load(base_path / "classification_features.pkl")

        
        try:
            models['binning_info'] = joblib.load(base_path / "binning_info.pkl")
        except:
            models['binning_info'] = None

    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.info("Make sure you've trained and saved your models in the notebooks first!")
        return None

    return models


def make_regression_prediction(models, input_data):
    """Make a regression prediction."""
    
    input_scaled = models['regression_scaler'].transform(input_data)
    
    prediction = models['regression_model'].predict(input_scaled)
    return prediction[0]


def make_classification_prediction(models, input_data):
    """Make a classification prediction."""
    
    input_scaled = models['classification_scaler'].transform(input_data)
    
    prediction = models['classification_model'].predict(input_scaled)
    
    label = models['label_encoder'].inverse_transform(prediction)
    return label[0], prediction[0]

def marketcap_prediction_output(value_billions):
    if value_billions >= 1000:
        trillions = value_billions / 1000
        return f"${trillions:,.2f} Trillion"
    
    elif value_billions >= 1:
        return f"${value_billions:,.2f} Billion"
    
    else:
        millions = value_billions * 1000
        return f"${value_billions:,.2f} Million"

import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

def make_classification_predictions(models, input_df: pd.DataFrame):
    
    feature_names = models["classification_features"]
    X = input_df.reindex(columns=feature_names, fill_value=0)

    
    log_cols = ["revenue", "total-assets", "net-assets", "earnings", "total-debt"]
    for c in log_cols:
        if c in X.columns:
            X[c] = np.log1p(X[c])

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

   
    if "classification_scaler" in models and models["classification_scaler"] is not None:
        X_scaled = models["classification_scaler"].transform(X)
        X = pd.DataFrame(X_scaled, columns=feature_names)

    clf = models["classification_model"]

    
    pred_class = clf.predict(X)[0]          
    proba = clf.predict_proba(X)[0]

    
    le = models.get("label_encoder", None)
    if le is not None:
        pred_label = le.inverse_transform([int(pred_class)])[0]
        class_labels = list(le.inverse_transform(np.arange(len(clf.classes_))))
    else:
        pred_label = pred_class
        class_labels = list(clf.classes_)

    
    proba_dict = {str(lbl): float(p) for lbl, p in zip(class_labels, proba)}

    return pred_label, proba_dict

# =============================================================================
# SIDEBAR - Navigation
# =============================================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a model:",
    ["🏠 Home", "📈 Regression Model", "🏷️ Classification Model"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    """
    This app deploys machine learning models trained on industry market cap data.

    - **Regression**: Predicts market cap value
    - **Classification**: Predicts small/mid/large cap
    """
)

st.sidebar.markdown("**Built by:** Denz'l Chapman")
st.sidebar.markdown("[GitHub Repo]https://github.com/denzlchapman/individual-capstone-denzlchapman.git")


# =============================================================================
# HOME PAGE
# =============================================================================
if page == "🏠 Home":
    st.title("🤖 Machine Learning Market Cap Prediction App")
    st.markdown("### Welcome!")

    st.write(
        """
        This application allows you to make predictions using trained machine learning models.

        **What you can do:**
        - 📈 **Regression Model**: Predicts a companies market cap value 
        - 🏷️ **Classification Model**: Predicts which category (small/mid/large)

        Use the sidebar to navigate between different models.
        """
    )

    
    st.markdown("---")
    st.markdown("### About This Project")
    st.write(
        """
        **Dataset:** This dataset comes from Kaggle and contains financial data on 1500 companies across 8 different industries in the stock market.

        **Problem Statement:** The goal of this project was to build a regression model to predict the market cap of a company and a classification model to categorize companies into small, mid, or large cap based on their financial features.

        **Models Used:**
        - Regression: Gradient Boosting Regression
        - Classification: Logistic Regression
        """
    )
    st.markdown("---")
    st.image("data/raw/image/original-dataset.png", caption="Original Dataset Sample")
    


# =============================================================================
# REGRESSION PAGE
# =============================================================================
elif page == "📈 Regression Model":
    st.title("📊Marketcap Prediction")
    st.write("Enter financial metrics to get market cap prediction.")

    
    models = load_models()

    if models is None:
        st.stop()

    
    features = models['regression_features']

    st.markdown("---")
    st.markdown("### 📊 Financial Metrics Input")

    
    col1, col2 = st.columns(2)

    with col1:
        total_assets = st.number_input(
            "Total Assets (Billions $)",
            min_value = 0.0,
            value = 50.0,
            step = 1.0,
            help="The total value of everything the company owns."
            )

        total_debt = st.number_input(
            "Total Debt (Billions $)",
            min_value = 0.0,
            value = 20.0,
            step = 1.0,
            help="The total amount of money the company owes."
            )

        revenue = st.number_input(
            "Revenue (Billions $)",
            min_value = 0.0,
            value = 30.0,
            step = 1.0,
            help="The 'Top Line' - total income from sales before expenses."
            )

        earnings = st.number_input(
            "Earnings (Billions $)",
            value = 5.0,
            step = 1.0,
            help="The 'Bottom Line' - profit after all expenses."
            )

        net_assets = st.number_input(
            "Net Assets (Billions $)",
            min_value = 0.0,
            value = 25.0,
            step = 1.0,
            help="Total assets minus total liabilities."
            )

    with col2:
        st.markdown("### 📈 Ratios")

    
        return_on_assets = st.number_input(
            "Return on Assets (%)",
            min_value = -100.0,
            max_value = 100.0,
            value = 10.0,
            step = 0.5,
            help="Net Income / Total assets, measures how efficiently a company uses its total assets."
            )

        return_on_equity = st.number_input(
            "Return On Equity (%)",
            min_value = -200.0,
            max_value = 200.0,
            value = 15.0,
            step = 0.5,
            help="Net income / (total assets - total liabilities), measures a company's profitability."
            )

        debt_to_equity = st.number_input(
            "Debt to Equity",
            min_value = 0.0,
            value = 1.0,
            step = 0.5,
            help="Total liabilities / (total assets - total liabilities), higher ratio indicates higher risk & reliance on debt"
            )

    st.markdown("---")
    st.markdown("## 🏭 Industry")

    industry = st.selectbox(
        "Select Industry",
        ["Pharmaceuticals",
        "Technology",
        "Real Estate",
        "Food",
        "Oil & Gas",
        "Insurance",
        "Retail"]
        )

    industry_columns = [
        "Industry_Pharmaceuticals",
        "Industry_Technology",
        "Industry_Real Estate",
        "Industry_Food",
        "Industry_Oil&Gas",
        "Industry_Insurance",
        "Industry_Retail"
        ]

    industry_map = {
        "Pharmaceuticals" : "Industry_Pharmaceuticals",
        "Technology" : "Industry_technology",
        "Real Estate" : "Industry_Real Estate",
        "Food" : "Industry_Food",
        "Oil & Gas" : "Industry_Oil&Gas",
        "Insurance" : "Industry_Insurance",
        "Retail" : "Industry_Retail"
    }  

    input_values = {col: 0 for col in industry_columns}
    input_values[industry_map[industry]] = 1

    input_values.update({
        "total-assets" : total_assets,
        "total-debt" : total_debt,
        "revenue" : revenue,
        "net-assets" : net_assets,
        "earnings" : earnings,
        "return-on-assets" : return_on_assets,
        "return-on-equity" : return_on_equity,
        "debt-to-equity" : debt_to_equity,
    })    



    st.markdown("---")

    
    if st.button("📈 Predict Market Cap", use_container_width=True):
        
        input_df = pd.DataFrame([input_values])
        
        log_cols = ["revenue", "total-assets", "net-assets", "earnings", "total-debt"]
        input_df[log_cols] = np.log1p(input_df[log_cols])
        input_df = input_df.reindex(columns=models["regression_features"], fill_value=0)
        
        prediction_log = make_regression_prediction(models, input_df)
        prediction_billions = np.expm1(prediction_log)
        prediction_output = marketcap_prediction_output(prediction_billions)

        
        st.success(f"### Predicted Marketcap Value: {prediction_output}")

        
        with st.expander("View Input Summary"):
            st.dataframe(input_df)


# =============================================================================
# CLASSIFICATION PAGE
# =============================================================================
elif page == "🏷️ Classification Model":
    st.title("🏷️ Company Size Prediction")
    st.write("Enter financial features to get market cap size prediction.")

    
    models = load_models()

    if models is None:
        st.stop()

   
    features = models['classification_features']
    class_labels = models['label_encoder'].classes_

   
    st.info(f"**Possible Categories:** {', '.join(class_labels)}")

    
    if models['binning_info']:
        with st.expander("How were categories created?"):
            binning = models['binning_info']
            st.write(f"Original target: **{binning['original_target']}**")
            st.write("Categories were created by binning the numerical values:")
            for i, label in enumerate(binning['labels']):
                if i == 0:
                    st.write(f"- **{label}**: < ${binning['bins'][i+1]} Billion")
                elif i == len(binning['labels']) - 1:
                    st.write(f"- **{label}**: >= ${binning['bins'][i]} Billion")
                else:
                    st.write(f"- **{label}**: {binning['bins'][i]} Billion to ${binning['bins'][i+1]} Billion")

    st.markdown("---")
    st.markdown("### Enter Financial Metrics")

    

    col1, col2 = st.columns(2)

    with col1:
        total_assets = st.number_input(
            "Total Assets (Billions $)",
            min_value = 0.0,
            value = 50.0,
            step = 1.0
            )

        total_debt = st.number_input(
            "Total Debt (Billions $)",
            min_value = 0.0,
            value = 20.0,
            step = 1.0
            )

        revenue = st.number_input(
            "Revenue (Billions $)",
            min_value = 0.0,
            value = 30.0,
            step = 1.0
            )

        earnings = st.number_input(
            "Earnings (Billions $)",
            value = 5.0,
            step = 1.0
            )

    with col2:
        net_assets = st.number_input(
            "Net Assets (Billions $)",
            min_value = 0.0,
            value = 25.0,
            step = 1.0
            )

        st.markdown("### 📈 Ratios")

        return_on_assets = st.number_input(
            "Return on Assets (%)",
            min_value = -100.0,
            max_value = 100.0,
            value = 10.0,
            step = 0.5
            )

        return_on_equity = st.number_input(
            "Return On Equity (%)",
            min_value = -200.0,
            max_value = 200.0,
            value = 15.0,
            step = 0.5
            )

        debt_to_equity = st.number_input(
            "Debt to Equity",
            min_value = 0.0,
            value = 1.0,
            step = 0.5
            )

    st.markdown("---")
    st.markdown("## 🏭 Industry")

    industry = st.selectbox(
        "Select Industry",
        ["Pharmaceuticals",
        "Technology",
        "Real Estate",
        "Food",
        "Oil & Gas",
        "Insurance",
        "Retail"]
        )

    industry_columns = [
        "Industry_Pharmaceuticals",
        "Industry_Technology",
        "Industry_Real Estate",
        "Industry_Food",
        "Industry_Oil&Gas",
        "Industry_Insurance",
        "Industry_Retail"
        ]

    industry_map = {
        "Pharmaceuticals" : "Industry_Pharmaceuticals",
        "Technology" : "Industry_technology",
        "Real Estate" : "Industry_Real Estate",
        "Food" : "Industry_Food",
        "Oil & Gas" : "Industry_Oil&Gas",
        "Insurance" : "Industry_Insurance",
        "Retail" : "Industry_Retail"
    }  

    input_values = {col: 0 for col in industry_columns}
    input_values[industry_map[industry]] = 1

    input_values.update({
        "total-assets" : total_assets,
        "total-debt" : total_debt,
        "revenue" : revenue,
        "net-assets" : net_assets,
        "earnings" : earnings,
        "return-on-assets" : return_on_assets,
        "return-on-equity" : return_on_equity,
        "debt-to-equity" : debt_to_equity,
    })    




    st.markdown("---")

   
    if st.button("📊 Predict Market Cap Classification", type="primary"):
        input_df = pd.DataFrame([input_values])
        input_df = input_df.reindex(columns=models["classification_features"], fill_value=0)

        
        predicted_label, predicted_index = make_classification_prediction(models, input_df)

        
        color_map = {
            'Small Cap': '🔴',
            'Mid Cap': '🟡',
            'Large Cap': '🟢'
        }
        emoji = color_map.get(predicted_label, '🔵')

        st.success(f"### Predicted Category: {emoji} {predicted_label}")

        
        with st.expander("View Input Summary"):
            st.dataframe(input_df)
        

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        Built by Denz'l Chapman | Full Stack Academy AI & ML Bootcamp
    </div>
    """,
    unsafe_allow_html=True
)
