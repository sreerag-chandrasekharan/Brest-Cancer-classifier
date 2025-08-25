import streamlit as st
import joblib
import pandas as pd
import json
import plotly.graph_objects as go
import numpy as np

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
model = joblib.load(open("C:/Users/Sreerag/Documents/ML_chellange/Brest-Cancer-classifier/models/cytosis/cytosis_model.pkl", "rb"))

def display_prediction(input_data):
    row = [input_data[k] for k in FEATURE_ORDER]
    input_array = np.array(row).reshape(1, -1)
    input_data_scaled = scaler.transform(input_array)
    proba = model.predict_proba(input_data_scaled)[0]
    pred = model.predict(input_data_scaled)[0]

    st.subheader('Cell cluster prediction')
    st.write("The cell cluster is: ")
    if pred == 0:
        st.write("<span style='color:green; font-size:24px;'>Benign</span>", unsafe_allow_html=True)
    else:
        st.write("<span style='color:red;font-size:24px;'>Malignant</span>", unsafe_allow_html=True)

    st.write(f"Probability of being benign: {proba[0]:.2%}")
    st.write(f"Probability of being malignant: {proba[1]:.2%}")




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
        st.write("Please connect this app to your cytology lab to help diagnose breast cancer form your tissue sample. This app predicts using a machine learning model whether a breast mass is benign or malignant based on the measurements it receives from your cytosis lab. You can also update the measurements by hand using the sliders in the sidebar. ")
        col1, col2 = st.columns([4,2])
        with col1:
            radar_chart = add_radar_chart(data)
            st.plotly_chart(radar_chart, use_container_width=True)
        with col2:
            display_prediction(data)



if __name__ == '__main__':
    main()