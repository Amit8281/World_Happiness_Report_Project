#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pickle
import pandas as pd
import gradio as gr
import warnings
warnings.filterwarnings('ignore')

# Load the trained model
model = pickle.load(open('GradientBoosting.pkl', 'rb'))

def word_happiness(Standard_Error, Economy_GDP_per_Capita, Family, Freedom, Trust_Government_Corruption, Generosity, Dystopia_Residual):
    # Prepare the input data as a DataFrame
    data = pd.DataFrame({
        'Standard_Error': [Standard_Error],
        'Economy_GDP_per_Capita': [Economy_GDP_per_Capita],
        'Family': [Family],
        'Freedom': [Freedom],
        'Trust_Government_Corruption': [Trust_Government_Corruption],
        'Generosity': [Generosity],
        'Dystopia_Residual': [Dystopia_Residual]
    })

    # Perform the prediction
    prediction = model.predict(data)
    return prediction[0]

# Create the input components
input_components = [
    gr.inputs.Number(label="Standard Error"),
    gr.inputs.Number(label="Economy GDP per Capita"),
    gr.inputs.Number(label="Family"),
    gr.inputs.Number(label="Freedom"),
    gr.inputs.Number(label="Trust Government Corruption"),
    gr.inputs.Number(label="Generosity"),
    gr.inputs.Number(label="Dystopia Residual")
]

# Create the interface
interface = gr.Interface(
    fn=word_happiness,
    inputs=input_components,
    outputs="number",
    title="Word Happiness Report Project",
    description="Word Happiness Report Project."
)

# Launch the interface
interface.launch()


# In[ ]:




