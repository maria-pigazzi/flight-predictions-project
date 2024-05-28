import streamlit as st
import pandas as pd
from pickle import load
import xgboost as xgb
# Load your model
xgb_model = load(open("xgb_model.sav", "rb"))
# Define airport code to numerical mapping
airport_mapping = {
    'ATL': 1, 'BOS': 2, 'CLT': 3, 'DEN': 4, 'DFW': 5,
    'DTW': 6, 'EWR': 7, 'IAD': 8, 'JFK': 9, 'LAX': 10,
    'LGA': 11, 'MIA': 12, 'OAK': 13, 'ORD': 14, 'PHL': 15, 'SFO': 16
}
# Assuming these are the feature names in the same order as the training set
feature_names = [
    'daysBetweenDates', 'isBasicEconomy','1depTimeOnly',
    'numberOfStopovers', 'waitingTime', 'numericalStartingAirport', 'numericalDestinationAirport', 'numerical1cabin'
]
# Streamlit app title
st.title("Total Fare Prediction")
# User input features
starting_airport = st.selectbox("What is your departure airport?", options=airport_mapping.keys())
destination_airport = st.selectbox("What is your destination airport?", options=airport_mapping.keys())
days_between_dates = st.slider("In how many days would you like to travel?", min_value=0, max_value=10)
is_basic_economy = st.slider("Do you want to travel in basic economy?", min_value=0, max_value=1)
number_of_stopovers = st.slider("How many stopovers would you be prepared to do?", min_value=0, max_value=10)
waiting_time = st.slider("How much time would you be prepared to wait between flights?", min_value=0, max_value=10)
numerical_1cabin = st.slider("Choose a cabin type:", min_value=0, max_value=10)
dep_time_only = st.slider("Choose the best time to start the travel:", min_value=0, max_value=10)
# Convert user input to DataFrame with correct types and order
input_features = {
    'numericalStartingAirport': airport_mapping[starting_airport],
    'numericalDestinationAirport': airport_mapping[destination_airport],
    'daysBetweenDates': days_between_dates,
    'isBasicEconomy': is_basic_economy,
    'numberOfStopovers': number_of_stopovers,
    'waitingTime': waiting_time,
    'numerical1cabin': numerical_1cabin,
    '1depTimeOnly': dep_time_only
}
# Convert input features to DataFrame and reorder columns
input_data = pd.DataFrame([input_features])
input_data = input_data[feature_names]
# Prediction button
if st.button("Predict Total Fare"):
    # Make prediction
    prediction = xgb_model.predict(input_data)[0]
    # Display prediction
    st.write(f"The predicted total fare is: ${round(prediction, 2)}")