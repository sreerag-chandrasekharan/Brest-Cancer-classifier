import streamlit as st
import joblib
import pickle
import pandas as pd
import json
import plotly.graph_objects as go
import numpy as np
from PIL import Image

# min and max values for the sliders
with open("data/processed_cytosis/feature_stats.json", "r") as f:
    FEATURE_STATS = json.load(f)

def get_min_max_mean(key):
    return FEATURE_STATS[key]

# Define the slider labels and keys
SLIDER_LABELS = [
    ("Radius (mean)", "radius_mean"),
    ("Texture (mean)", "texture_mean"),
    ("Perimeter (mean)", "perimeter_mean"),
    ("Area (mean)", "area_mean"),
    ("Smoothness (mean)", "smoothness_mean"),
    ("Compactness (mean)", "compactness_mean"),
    ("Concavity (mean)", "concavity_mean"),
    ("Concave points (mean)", "concave points_mean"),
    ("Symmetry (mean)", "symmetry_mean"),
    ("Fractal dimension (mean)", "fractal_dimension_mean"),
    ("Radius (se)", "radius_se"),
    ("Texture (se)", "texture_se"),
    ("Perimeter (se)", "perimeter_se"),
    ("Area (se)", "area_se"),
    ("Smoothness (se)", "smoothness_se"),
    ("Compactness (se)", "compactness_se"),
    ("Concavity (se)", "concavity_se"),
    ("Concave points (se)", "concave points_se"),
    ("Symmetry (se)", "symmetry_se"),
    ("Fractal dimension (se)", "fractal_dimension_se"),
    ("Radius (worst)", "radius_worst"),
    ("Texture (worst)", "texture_worst"),
    ("Perimeter (worst)", "perimeter_worst"),
    ("Area (worst)", "area_worst"),
    ("Smoothness (worst)", "smoothness_worst"),
    ("Compactness (worst)", "compactness_worst"),
    ("Concavity (worst)", "concavity_worst"),
    ("Concave points (worst)", "concave points_worst"),
    ("Symmetry (worst)", "symmetry_worst"),
    ("Fractal dimension (worst)", "fractal_dimension_worst"),
]

#-- add the slidebar

def add_sidebar():

    st.sidebar.header("Cell Nuclei Measurements")

    input_dict = {}

    # Add the sliders
    for label, key in SLIDER_LABELS:
        min_value, max_value = get_min_max_mean(key)
        input_dict[key] = st.sidebar.slider(
            label,
            min_value= min_value,
            max_value= max_value,
            value= min_value,  # Default value is the mean
            format="%.2f")  
    return input_dict


scaler = joblib.load(open("C:/Users/Sreerag/Documents/ML_chellange/Brest-Cancer-classifier/data/processed_cytosis/scaler.pkl", "rb"))

FEATURE_ORDER = [key for _, key in SLIDER_LABELS]

# -- create radar chart ---
def add_radar_chart(input_dict):
    # scale the input values
    features = np.array([input_dict[feature] for feature in FEATURE_ORDER]).reshape(1, -1)
    scaled_features = scaler.transform(features)[0]

    input_data = {feature: scaled_val for feature, scaled_val in zip(FEATURE_ORDER, scaled_features)}

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=[input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
                input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
                input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
                input_data['fractal_dimension_mean']],
            theta=['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', 'Concavity', 'Concave Points',
                   'Symmetry', 'Fractal Dimension'],
            fill='toself',
            name='Mean'

        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=[input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
                input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
                input_data['concave points_se'], input_data['symmetry_se'], input_data['fractal_dimension_se']],
            theta=['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', 'Concavity', 'Concave Points',
                   'Symmetry', 'Fractal Dimension'],
            fill='toself',
            name='Standard Error'
        )
    )

    fig.add_trace(
        go.Scatterpolar(
            r=[input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
                input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
                input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
                input_data['fractal_dimension_worst']],
            theta=['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', 'Concavity', 'Concave Points',
                   'Symmetry', 'Fractal Dimension'],
            fill='toself',
            name='Worst'
        )
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5]
            )
        ),
        showlegend=True,
        autosize=True
    )
    return fig

# ---- make predictions
# Load the models---
model_cytosis = joblib.load(open("C:/Users/Sreerag/Documents/ML_chellange/Brest-Cancer-classifier/models/cytosis/cytosis_model.pkl", "rb"))
with open("C:/Users/Sreerag/Documents/ML_chellange/Brest-Cancer-classifier/models/image_resnet50.pkl", "rb") as f:
    image_model = pickle.load(f)

def display_prediction(input_data, uploaded_file):
    # Cytosis prediction
    row = [input_data[k] for k in FEATURE_ORDER]
    input_array = np.array(row).reshape(1, -1)
    input_data_scaled = scaler.transform(input_array)
    proba_cytosis = model_cytosis.predict_proba(input_data_scaled)[0]

    # Image model prediction
    proba_image = None
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        img = img.resize((224, 224))  # adjust according to your image model
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 224, 224, 3)
        proba_image = image_model.predict_proba(img_array)[0]  # adapt if keras/pytorch

        # Combine predictions
    if proba_image is not None:
        final_proba = (proba_cytosis + proba_image) / 2
    else:
        final_proba = proba_cytosis

    pred = np.argmax(final_proba)

    # Display
    st.subheader('Final Cell Cluster Prediction')
    if pred == 0:
        st.write("<span style='color:green; font-size:24px;'>Benign</span>", unsafe_allow_html=True)
    else:
        st.write("<span style='color:red; font-size:24px;'>Malignant</span>", unsafe_allow_html=True)

    st.write(f"Probability of being benign: {final_proba[0]:.2%}")
    st.write(f"Probability of being malignant: {final_proba[1]:.2%}")




#---- main function
def main():
    st.set_page_config(page_title="Breast Cancer Diagnosis",
        page_icon="👩‍⚕️", 
        layout="wide", 
        initial_sidebar_state="expanded")

    #--- add the sidebar
    data = add_sidebar()
    # ----Set up the structure

    with st.container():
        st.title("Breast Cancer Diagnosis")
        st.write("This app predicts whether a breast mass is benign or malignant using cytology measurements or mamography images. Adjust the sliders and upload an image, then click 'Predict Output' to see the results.")
        col1, col2, col3 = st.columns([3,2,3])
        with col1:
            # ---- Image upload
            uploaded_file = st.file_uploader("Upload mamography image", type=["jpg", "jpeg", "png"])
            radar_chart = add_radar_chart(data)
            st.plotly_chart(radar_chart, use_container_width=True)
        with col2:
            if uploaded_file is not None:
                st.subheader("Uploaded Image")
                st.image(uploaded_file, caption="Uploaded Cell Image", use_column_width=True)
            else:
                st.subheader("No image uploaded")
        with col3:
            # ---- Predict button
            #predict_clicked = st.button("Predict Output", use_container_width=True)
            #if predict_clicked:
            display_prediction(data, uploaded_file)



if __name__ == '__main__':
    main()