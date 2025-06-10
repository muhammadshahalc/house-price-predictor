import streamlit as st 
import joblib 
import numpy as np
import pandas as pd
from PIL import Image

# Configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# Load models and encoders (with caching)
@st.cache_resource
def load_artifacts():
    return {
        "model": joblib.load("models/random_forest_model.pkl"),
        "scaler": joblib.load("models/standard_scaler.pkl"),
        "ohe_transaction": joblib.load("models/transaction_ohe.pkl"),
        "ohe_location": joblib.load("models/location_ohe.pkl"),
        "lbl_furnishing": joblib.load("models/furnishing_label_encoder.pkl"),
        "lbl_ownership": joblib.load("models/ownership_label_encoder.pkl")
    }

artifacts = load_artifacts()

def add_bg_with_blur():
    st.markdown("""
        <style>
        .stApp {
            background-image: url("https://images.unsplash.com/photo-1523821741446-edb2b68bb7a0?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
        }
        </style>
    """, unsafe_allow_html=True)



# UI Components
def main():
    add_bg_with_blur()
    # Header Section
    st.title("🏠 Smart Home Price Predictor")
    st.markdown("""
    <style>
    .big-font {
        font-size:18px !important;
        color: #4f8bf9;
    }
    .result-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("Predict accurate housing prices across major Indian cities")
    
    # Sidebar for additional info
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This app predicts house prices using machine learning. 
        The model considers:
        - Property details
        - Location
        - Transaction type
        - Furnishing status
        """)
        st.markdown("---")
        st.markdown("🔍 *Try different combinations to see how prices change!*")
    
    # Main Content
    col1, col2 = st.columns([1, 2])
    
    with col1:
        image = Image.open("images/house.jpg")
        st.image(image, caption="Find your dream home", use_container_width=True)
    
    with col2:
        # Input Form
        with st.form("prediction_form"):
            st.subheader("Property Details")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                bhk = st.selectbox("Bedrooms (BHK)", [1, 2, 3, 4, 5], index=2)
            with col2:
                bathroom = st.selectbox("Bathrooms", [1, 2, 3, 4, 5, 6], index=1)
            with col3:
                balcony = st.selectbox("Balconies", [0, 1, 2, 3, 4, 5 , 6, 7], index=1)
            
            total_sqft = st.slider("Total Area (sqft)", 500, 10000, 1200, step=50)
            
            col1, col2 = st.columns(2)
            with col1:
                transaction = st.selectbox("Transaction Type", ["resale", "new property"])
                furnishing = st.selectbox("Furnishing", ["semi-furnished", "unfurnished", "furnished"])
            with col2:
                location = st.selectbox("Location", [
                    'mumbai', 'bangalore', 'chennai', 'gurgaon', 'hyderabad', 
                    'kolkata', 'new delhi', 'pune', 'thane', 'ahmedabad',
                    'jaipur', 'noida', 'other', 'chandigarh', 'faridabad',
                    'greater noida', 'mohali', 'surat', 'vadodara', 
                    'visakhapatnam', 'zirakpur'
                ])
                ownership = st.selectbox("Ownership", ["freehold", "leasehold"])
            
            submitted = st.form_submit_button("Predict Price", type="primary")
    
    # Prediction Logic
    if submitted:
        with st.spinner("Calculating price..."):
            try:
                # Prepare input data
                input_data = pd.DataFrame({
                    "bhk": [bhk],
                    "bathroom": [bathroom],
                    "balcony": [balcony],
                    "total_sqft": [total_sqft]
                })
                
                # Encode categorical features
                input_data["furnishing"] = artifacts["lbl_furnishing"].transform([furnishing])[0]
                input_data["ownership"] = artifacts["lbl_ownership"].transform([ownership])[0]
                
                # One-hot encoding
                df_tr = pd.DataFrame(
                    artifacts["ohe_transaction"].transform([[transaction]]),
                    columns=artifacts["ohe_transaction"].get_feature_names_out(["transaction"])
                )
                
                df_loc = pd.DataFrame(
                    artifacts["ohe_location"].transform([[location]]),
                    columns=artifacts["ohe_location"].get_feature_names_out(["location"])
                )
                
                # Combine all features
                input_data = pd.concat([input_data, df_tr, df_loc], axis=1)
                
                # Ensure correct feature order and fill missing with 0
                expected_columns = artifacts["scaler"].feature_names_in_
                for col in expected_columns:
                    if col not in input_data.columns:
                        input_data[col] = 0
                input_data = input_data[expected_columns]
                
                # Scale and predict
                input_scaled = artifacts["scaler"].transform(input_data)
                log_prediction = artifacts["model"].predict(input_scaled)
                final_price = np.expm1(log_prediction)[0]
                
                # Display results
                st.success("Prediction Complete!")
                
                with st.container():
                    st.markdown("### 🏡 Estimated Property Value")
                    st.markdown(f"""
                    <div class="result-box">
                        <h2 style="color:#2e86de; text-align:center;">₹{final_price:,.2f}</h2>
                        <p style="text-align:center;">For a {bhk} BHK in {location.title()}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Optional debug view
                with st.expander("Show technical details"):
                    st.markdown("**Processed Input Features:**")
                    st.dataframe(input_data)
                    st.markdown(f"Log Prediction: {log_prediction[0]:.2f}")
                    
            except Exception as e:
                st.error(f"⚠️ Prediction failed: {str(e)}")
                st.info("Please check your inputs and try again.")

if __name__ == "__main__":
    main()



