import tensorflow as tf
from tensorflow import keras
import numpy as np
import streamlit as st
import os,pickle,re

#2. Function to load the pickle objects and keras models

def load_pickle_file(filepath):
    with open(filepath, 'rb') as f:
        pickle_object = pickle.load(f)
    return pickle_object

def load_model(filepath):
    model_object = keras.models_load()
    return model_object

@st.cache_resource
def load_model(filepath):
    model_object = keras.models.load_model(filepath)
    return model_object

#3. Define filepaths to resources
tokenizer_filepath = "tokenizer.pkl"
label_encoder_path = "label_encoder.pkl"
model_filepath = "nlp_model"

#4. Load the Tokenizer
tokenizer = load_pickle_file(tokenizer_filepath)
label_encoder = load_pickle_file(label_encoder_path)
model = load_model(model_filepath)

#5. Build the sttreamlit app
st.title("Product Categorization")
#(B)
with st.form('input_form'):
    text_area = st.text_area("Input your description here")
    submitted = st.form_submit_button('Predict the product!')

text_inputs = [text_area]

#(C) Process input String
def remove_unwanted_string(text_inputs):
    for index,data in enumerate(text_inputs):
        text_inputs[index] = re.sub('<.*>?'," ",data)
        text_inputs[index] = re.sub("[^a-zA-Z]"," ",data).lower()
    return text_inputs

# Filter unwanted strings
text_filtered = remove_unwanted_string(text_inputs)
# Organizing String
text_token = tokenizer.texts_to_sequences(text_filtered)
# Padding and Truncating
text_padded = keras.preprocessing.sequence.pad_sequences(text_token,maxlen=200,padding='post',truncating='post')
# Model prediction
y_score = model.predict(text_padded)
y_pred = np.argmax(y_score,axis = 1)

#Displaying result
label_map = {i:classes for i,classes in enumerate(label_encoder.classes_)}
result = label_map[y_pred[0]]

#Writing Prediction into StreamLit
st.header("Label List")
st.write(label_encoder.classes_)
st.header("Prediction Score")
st.write(y_score)
st.header("Final Prediction")
st.write(f"The product description falls into the category: {result}")