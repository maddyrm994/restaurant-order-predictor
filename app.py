import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb

# --- Configuration ---
st.set_page_config(
    page_title="Restaurant Order Prediction",
    page_icon="🍲",
    layout="wide"
)

# --- Caching Functions for Performance ---
@st.cache_resource
def load_model():
    """Load the pre-trained XGBoost model."""
    try:
        model = joblib.load('xgboost_model.joblib')
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'xgboost_model.joblib' is in the same directory.")
        return None

@st.cache_resource
def load_model_columns():
    """Load the column layout the model was trained on."""
    try:
        columns = joblib.load('model_columns.joblib')
        return columns
    except FileNotFoundError:
        st.error("Model columns file not found. Please ensure 'model_columns.joblib' is in the same directory.")
        return None

@st.cache_data
def load_data():
    """Load the base data for creating scenarios."""
    try:
        df = pd.read_csv('full_details_for_app.csv')
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please ensure 'full_details_for_app.csv' is in the same directory.")
        return None

# --- Load Artifacts ---
model = load_model()
model_columns = load_model_columns()
df = load_data()

# --- Main App ---
if model is None or model_columns is None or df is None:
    st.stop() # Stop execution if files are not loaded

st.title("Future Order Prediction Dashboard 🍲")
st.markdown("Select a day and an hour to predict the probability of each menu item being ordered.")

# --- User Inputs in Sidebar ---
st.sidebar.header("Prediction Inputs")

# Create a list of days in the correct order
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
selected_day = st.sidebar.selectbox("Select Day of the Week", day_order)

# Create a list of hours
hours = sorted(df['hour'].unique())
selected_hour = st.sidebar.selectbox("Select Hour of the Day (24-hour format)", hours)

predict_button = st.sidebar.button("Predict Order Pattern", type="primary")

# --- Prediction Logic ---
if predict_button:
    with st.spinner('Calculating probabilities...'):
        # 1. Get unique items and their mappings from the base data
        unique_items = df[['food_item_name', 'food_item_category']].drop_duplicates()
        day_to_type_map = {'Weekend' if d in ['Saturday', 'Sunday'] else 'Weekday' for d in day_order}
        
        # 2. Create a scenario DataFrame for every item for the selected time
        # We will predict for both Dine-In and Take-Away and average the probability
        scenarios = []
        for _, row in unique_items.iterrows():
            item_name = row['food_item_name']
            category = row['food_item_category']
            day_type = 'Weekend' if selected_day in ['Saturday', 'Sunday'] else 'Weekday'
            
            # Scenario for Dine In
            scenarios.append({
                'food_item_name': item_name, 'hour': selected_hour, 'day_of_the_week': selected_day,
                'day_type': day_type, 'order_type': 'Dine In', 'food_item_category': category
            })
            # Scenario for Take Away
            scenarios.append({
                'food_item_name': item_name, 'hour': selected_hour, 'day_of_the_week': selected_day,
                'day_type': day_type, 'order_type': 'Take Away', 'food_item_category': category
            })

        future_df = pd.DataFrame(scenarios)

        # 3. Preprocess the scenario DataFrame exactly as the training data
        future_encoded = pd.get_dummies(future_df)
        future_aligned = future_encoded.reindex(columns=model_columns, fill_value=0)

        # 4. Make predictions (we need probabilities, not just 0/1)
        # .predict_proba gives probabilities for [class 0, class 1]
        predictions_proba = model.predict_proba(future_aligned)[:, 1] # Get probability of class 1 ("ordered")
        future_df['probability'] = predictions_proba

        # 5. Aggregate results to get a single probability per item
        final_predictions = future_df.groupby('food_item_name')['probability'].mean().reset_index()
        final_predictions.rename(columns={'probability': 'Predicted Probability'}, inplace=True)
        
        # 6. Sort and display the results
        sorted_predictions = final_predictions.sort_values('Predicted Probability', ascending=False)
        
        # Format the probability as a percentage for better readability
        sorted_predictions['Predicted Probability'] = sorted_predictions['Predicted Probability'].map('{:.2%}'.format)

        st.subheader(f"Predicted Order Probabilities for {selected_day} at {selected_hour}:00")
        st.dataframe(sorted_predictions, use_container_width=True, height=600)

else:
    st.info("Please select a day and hour in the sidebar and click 'Predict'.")
