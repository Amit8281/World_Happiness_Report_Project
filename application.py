import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

# Load the trained model
model = pickle.load(open('GradientBoosting.pkl', 'rb'))

# Load the scaler model
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Get user input for the columns
st.title('World Happiness')
Standard_Error = st.number_input('Standard Error')
Economy_GDP_per_Capita = st.number_input('Economy GDP per Capita')
Family = st.number_input('Family')
Freedom = st.number_input('Freedom')
Trust_Government_Corruption = st.number_input('Trust Government Corruption')
Generosity = st.number_input('Generosity')
Dystopia_Residual = st.number_input('Dystopia Residual')

# Add a "Predict" button
predict_button = st.button('Predict')

# Perform the prediction when the button is clicked
if predict_button:
    # Prepare the input data
    input_data = pd.DataFrame({
        'Standard_Error': [Standard_Error],
        'Economy_GDP_per_Capita': [Economy_GDP_per_Capita],
        'Family': [Family],
        'Freedom': [Freedom],
        'Trust_Government_Corruption': [Trust_Government_Corruption],
        'Generosity': [Generosity],
        'Dystopia_Residual': [Dystopia_Residual]
    })

    # Scale the input data
    scaled_input = scaler.transform(input_data)

    # Perform the prediction
    prediction = model.predict(scaled_input)

    # Display the predicted World Happiness Score
    st.subheader('World Happiness Score:')
    st.write(prediction[0])
