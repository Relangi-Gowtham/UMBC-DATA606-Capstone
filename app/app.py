import streamlit as st
import pickle
import pandas as pd
import numpy as np
import sklearn  # Required for pickle to load scikit-learn objects

# --- Page Configuration ---
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📡",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- File Paths ---
MODEL_FILE = 'best_model.pkl'
ENCODER_FILE = 'encoders.pkl'

# --- Load Assets ---
@st.cache_resource
def load_assets(model_path, encoder_path):
    """
    Loads the pickled model, feature names, and encoders.
    """
    try:
        # Load model and feature names
        with open(model_path, 'rb') as f_model:
            model_data = pickle.load(f_model)
        
        # As per your project code, model_data is a dict
        model = model_data["model"]
        feature_names = model_data["features_names"]
        
        # Load encoders
        with open(encoder_path, 'rb') as f_encoder:
            encoders = pickle.load(f_encoder)
            
        return model, feature_names, encoders
        
    except FileNotFoundError:
        st.error(f"Error: Could not find files.")
        st.error(f"Please check paths: '{model_path}' and '{encoder_path}'")
        return None, None, None
    except Exception as e:
        st.error(f"An error occurred while loading files: {e}")
        return None, None, None

# Load the model, feature names, and encoders
model, feature_names, encoders = load_assets(MODEL_FILE, ENCODER_FILE)

# --- Main Application ---
st.title('📡 Telco Customer Churn Predictor')
st.write("Enter the customer's details in the sidebar to predict churn.")

# --- Sidebar for User Inputs ---
st.sidebar.header('Customer Details')

if model and feature_names and encoders:
    
    # This dictionary will hold all our user inputs
    input_features = {}

    # --- Create input fields based on your project's features ---
    
    st.sidebar.subheader("Demographics")
    input_features['gender'] = st.sidebar.selectbox(
        "Gender", 
        options=["Female", "Male"]
    )
    input_features['SeniorCitizen'] = st.sidebar.selectbox(
        "Senior Citizen", 
        options=[0, 1], 
        format_func=lambda x: "Yes" if x == 1 else "No"
    )
    input_features['Partner'] = st.sidebar.radio(
        "Has Partner?", 
        options=["Yes", "No"], 
        horizontal=True
    )
    input_features['Dependents'] = st.sidebar.radio(
        "Has Dependents?", 
        options=["Yes", "No"], 
        horizontal=True
    )

    st.sidebar.subheader("Account Information")
    input_features['tenure'] = st.sidebar.slider(
        "Tenure (Months)", 
        min_value=0, 
        max_value=72, 
        value=1
    )
    input_features['Contract'] = st.sidebar.selectbox(
        "Contract Type",
        options=["Month-to-month", "One year", "Two year"]
    )
    input_features['PaperlessBilling'] = st.sidebar.radio(
        "Paperless Billing?",
        options=["Yes", "No"],
        horizontal=True
    )
    input_features['PaymentMethod'] = st.sidebar.selectbox(
        "Payment Method",
        options=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    input_features['MonthlyCharges'] = st.sidebar.number_input(
        "Monthly Charges ($)", 
        min_value=0.0, 
        max_value=120.0, 
        value=29.85, 
        step=0.05
    )
    # --- TotalCharges input is REMOVED ---
    # It will be calculated automatically

    st.sidebar.subheader("Subscribed Services")
    input_features['PhoneService'] = st.sidebar.radio(
        "Phone Service?",
        options=["Yes", "No"],
        horizontal=True
    )
    input_features['MultipleLines'] = st.sidebar.selectbox(
        "Multiple Lines",
        options=["No phone service", "No", "Yes"]
    )
    input_features['InternetService'] = st.sidebar.selectbox(
        "Internet Service",
        options=["DSL", "Fiber optic", "No"]
    )
    input_features['OnlineSecurity'] = st.sidebar.selectbox(
        "Online Security",
        options=["No", "Yes", "No internet service"]
    )
    input_features['OnlineBackup'] = st.sidebar.selectbox(
        "Online Backup",
        options=["No", "Yes", "No internet service"]
    )
    input_features['DeviceProtection'] = st.sidebar.selectbox(
        "Device Protection",
        options=["No", "Yes", "No internet service"]
    )
    input_features['TechSupport'] = st.sidebar.selectbox(
        "Tech Support",
        options=["No", "Yes", "No internet service"]
    )
    input_features['StreamingTV'] = st.sidebar.selectbox(
        "Streaming TV",
        options=["No", "Yes", "No internet service"]
    )
    input_features['StreamingMovies'] = st.sidebar.selectbox(
        "Streaming Movies",
        options=["No", "Yes", "No internet service"]
    )


    # --- Prediction Logic ---
    if st.sidebar.button('🔮 Predict Churn', use_container_width=True, type="primary"):
        
        try:
            # --- NEW: Automatically calculate TotalCharges ---
            input_features['TotalCharges'] = input_features['tenure'] * input_features['MonthlyCharges']

            # 1. Create a DataFrame from the user's inputs
            # We use a list [input_features] to create a single-row DataFrame
            input_df = pd.DataFrame([input_features])
            
            # --- CHANGED: Renamed header ---
            st.subheader("1. Input Data")
            st.dataframe(input_df)

            # 2. Apply the saved encoders
            # Create a copy to keep the raw inputs for display
            input_df_encoded = input_df.copy()
            
            # As per your project, 'encoders' is a dict {col_name: encoder_obj}
            for column, encoder in encoders.items():
                if column in input_df_encoded.columns:
                    input_df_encoded[column] = encoder.transform(input_df_encoded[column])
                else:
                    st.warning(f"Warning: Column '{column}' from encoder not found in input.")

            # --- REMOVED: Display of encoded data ---
            
            # 3. Ensure column order matches training
            # We loaded 'feature_names' from the model pickle file
            input_df_final = input_df_encoded[feature_names]

            # 4. Make predictions
            prediction = model.predict(input_df_final)
            pred_prob = model.predict_proba(input_df_final)
            
            # Get the single prediction value
            prediction_value = prediction[0]
            
            # Get the probabilities
            prob_no_churn = pred_prob[0][0]
            prob_churn = pred_prob[0][1]
            
            # 5. Display the result
            st.subheader("🎉 Prediction Result")
            
            if prediction_value == 1:
                st.error(f"**Prediction: CHURN** (Probability: {prob_churn:.1%})", icon="🔥")
                st.write("This customer is at high risk of churning.")
            else:
                st.success(f"**Prediction: NO CHURN** (Probability: {prob_no_churn:.1%})", icon="✅")
                st.write("This customer is likely to stay.")
            
            # Show probabilities in columns
            col1, col2 = st.columns(2)
            col1.metric("Probability of 'No Churn'", f"{prob_no_churn:.1%}")
            col2.metric("Probability of 'Churn'", f"{prob_churn:.1%}")

        except Exception as e:
            st.error(f"An error occurred during prediction:")
            st.error(e)
else:
    st.error("Model assets could not be loaded. Please check the file paths and ensure the files are in the same folder as `app.py`.")