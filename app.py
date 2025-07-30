# --- START OF FILE app.py ---

import streamlit as st
import pandas as pd
import joblib
import requests
import holidays
from datetime import date, timedelta

# --- Configuration ---
st.set_page_config(
    page_title="Restaurant Order Prediction",
    page_icon="🍲",
    layout="wide"
)

# --- App Constants (CONFIGURATION) ---
RESTAURANT_LOCATION = "Tamil Nadu" # <-- IMPORTANT: Change to your restaurant's city
HOLIDAY_COUNTRY = 'IN'           # <-- Change to your country code
HOLIDAY_PROVINCE = 'TN'          # <-- Change to your state/province, or None

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
    """Load the base data for getting unique item names."""
    try:
        df = pd.read_csv('full_details_for_app.csv')
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please ensure 'full_details_for_app.csv' is in the same directory.")
        return None

# --- NEW: Function to load API key from file ---
@st.cache_data
def load_api_key():
    """Reads the API key from a local file 'api_key.txt'."""
    try:
        with open('api_key.txt', 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        st.error("API key file not found. Please create 'api_key.txt' and paste your key in it.")
        return None

# --- NEW: Function to get weather forecast for a specific hour ---
def get_hourly_weather_forecast(api_key, location, target_date, target_hour):
    """Fetches weather forecast for a specific date and hour from WeatherAPI."""
    base_url = "http://api.weatherapi.com/v1/forecast.json"
    days = (target_date - date.today()).days + 1
    
    if not (0 < days <= 14):
        st.error(f"Cannot fetch forecast. Date must be within the next 14 days.")
        return None

    params = {'key': api_key, 'q': location, 'days': days}
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        # Get the hourly data for the specific day we're interested in
        hourly_data = response.json()['forecast']['forecastday'][-1]['hour']
        # Return the specific hour's data
        return hourly_data[target_hour]
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return None
    except (KeyError, IndexError):
        st.error("Could not parse the weather data from the API response.")
        return None

# --- Load Artifacts ---
model = load_model()
model_columns = load_model_columns()
df_base = load_data()
api_key = load_api_key()
holiday_calendar = holidays.CountryHoliday(HOLIDAY_COUNTRY, prov=None, state=HOLIDAY_PROVINCE)

# --- Main App ---
if model is None or model_columns is None or df_base is None or api_key is None:
    st.warning("Application cannot start. Please resolve the errors above.")
    st.stop()

st.title("Future Order Prediction Dashboard 🍲")
st.markdown("Select a future date, time, and event status to predict order patterns.")

# --- User Inputs in Sidebar ---
st.sidebar.header("Prediction Inputs")

selected_date = st.sidebar.date_input("Select Date for Prediction", min_value=date.today(), max_value=date.today() + timedelta(days=13))
hours = sorted(df_base['hour'].unique())
selected_hour = st.sidebar.selectbox("Select Hour of the Day (24-hour format)", hours)

st.sidebar.subheader("Event Status")
is_holiday = st.sidebar.checkbox("Is it a public holiday?", value=(selected_date in holiday_calendar))
is_special_event = st.sidebar.checkbox("Is there a special local event? (e.g., concert, festival)")

predict_button = st.sidebar.button("Predict Order Pattern", type="primary", use_container_width=True)

# --- Prediction Logic ---
if predict_button:
    with st.spinner('Fetching weather and calculating probabilities...'):
        
        # 1. Fetch weather data for the selected date and hour
        weather_data = get_hourly_weather_forecast(api_key, RESTAURANT_LOCATION, selected_date, selected_hour)
        
        if weather_data is None:
            st.error("Prediction failed because weather data could not be retrieved.")
            st.stop()

        # 2. Display the fetched weather conditions
        st.subheader(f"Weather Conditions for {selected_date.strftime('%A, %Y-%m-%d')} at {selected_hour}:00")
        with st.container(border=True):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Temperature", f"{weather_data['temp_c']} °C")
            col2.metric("Precipitation", f"{weather_data['precip_mm']} mm")
            col3.metric("Wind Speed", f"{weather_data['wind_kph']} kph")
            col4.metric("Cloud Cover", f"{weather_data['cloud']}%")
        
        # 3. Create scenario DataFrame for prediction
        unique_items = df_base[['food_item_name', 'food_item_category']].drop_duplicates()
        day_of_week = selected_date.strftime('%A')
        day_type = 'Weekend' if day_of_week in ['Saturday', 'Sunday'] else 'Weekday'

        scenarios = []
        for _, row in unique_items.iterrows():
            base_scenario = {
                'hour': selected_hour,
                'food_item_name': row['food_item_name'],
                'food_item_category': row['food_item_category'],
                'day_of_the_week': day_of_week,
                'day_type': day_type,
                'temperature_c': weather_data['temp_c'],
                'wind_kph': weather_data['wind_kph'],
                'precipitation_mm': weather_data['precip_mm'],
                'cloud': weather_data['cloud'],
                'humidity': weather_data['humidity'],
                'pressure_mb': weather_data['pressure_mb'],
                'is_holiday': is_holiday,
                'is_special_event': is_special_event,
            }
            scenarios.append({**base_scenario, 'order_type': 'Dine In'})
            scenarios.append({**base_scenario, 'order_type': 'Take Away'})

        future_df = pd.DataFrame(scenarios)

        # 4. Preprocess, predict, and format results
        future_encoded = pd.get_dummies(future_df, columns=['food_item_name', 'food_item_category', 'day_of_the_week', 'day_type', 'order_type'])
        future_aligned = future_encoded.reindex(columns=model_columns, fill_value=0)
        
        predictions_proba = model.predict_proba(future_aligned)[:, 1]
        future_df['probability'] = predictions_proba

        # 5. Aggregate and display results in tabs
        overall_predictions = future_df.groupby('food_item_name')['probability'].mean().reset_index()
        sorted_overall = overall_predictions.sort_values('probability', ascending=False).reset_index(drop=True)
        sorted_overall.rename(columns={'probability': 'Predicted Probability'}, inplace=True)
        sorted_overall['Predicted Probability'] = sorted_overall['Predicted Probability'].map('{:.2%}'.format)

        detailed_predictions = future_df.pivot(index='food_item_name', columns='order_type', values='probability').reset_index()
        detailed_predictions = detailed_predictions.sort_values(by=['Dine In', 'Take Away'], ascending=False).fillna(0)
        detailed_predictions['Dine In'] = detailed_predictions['Dine In'].map('{:.2%}'.format)
        detailed_predictions['Take Away'] = detailed_predictions['Take Away'].map('{:.2%}'.format)

        st.subheader("Predicted Item Probabilities")
        tab1, tab2 = st.tabs(["📈 Overall Prediction", "📊 Dine-In vs. Take-Away"])
        with tab1:
            st.dataframe(sorted_overall, use_container_width=True, height=600)
        with tab2:
            st.info("This view helps analyze the impact of weather/events on customer behavior.")
            st.dataframe(detailed_predictions, use_container_width=True, height=600,
                         column_order=['food_item_name', 'Dine In', 'Take Away'])

else:
    st.info("⬅️ Please select your prediction criteria in the sidebar and click 'Predict'.")

# --- END OF FILE ---
