# from utils import db_connect
# engine = db_connect()

# your code here
import streamlit as st
import pandas as pd
from pickle import load
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# Load your DataFrame and model
even_more_useful_data = pd.read_csv("src/data_app.csv")
model_file = load(open("src/xgb_model.sav", "rb"))
# Streamlit app title
st.title("Total Fare Prediction")
# User input features
input_features = {
    'numericalStartingAirport': st.slider("What is your departure airport?", min_value=0, max_value=10),
    'numericalDestinationAirport': st.slider("What is your destination airport?", min_value=0, max_value=10),
    'daysBetweenDates': st.slider("In how many days would you like to travel?", min_value=0, max_value=10),
    'isBasicEconomy': st.slider("Do you want to travel in basic economy?", min_value=0, max_value=10),
    'numberOfStopovers': st.slider("How many stopovers would you be prepared to do?", min_value=0, max_value=10),
    'waitingTime': st.slider("How much time would you be prepared to wait between flights?", min_value=0, max_value=10),
    'numerical1cabin': st.slider("Choose a cabin type:", min_value=0, max_value=10),
    '1depTimeOnly': st.slider("Choose the best time to start the travel:", min_value=0, max_value=10),
    # Add more sliders for additional features as needed
}
# Prediction button
if st.button("Predict Total Fare"):
    # Convert user input to DataFrame
    input_data = pd.DataFrame(input_features, index=[0])
    # Make prediction
    prediction = model_file.predict(input_data)[0]
    # Display prediction
    st.write(f"The predicted total fare is: {round(prediction, 2)}")