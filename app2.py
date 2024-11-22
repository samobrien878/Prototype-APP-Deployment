import streamlit as st
import numpy as np
import pandas as pd
import joblib
import math
import matplotlib.pyplot as plt

st.set_page_config(page_title="Rat Performance Prediction App", page_icon="🐭", layout="centered")

# Define the path to the dataset
file_path = 'hab_data_sessions.csv'

# Load the habituation data
habituation_data = pd.read_csv(file_path)

# Calculate max values for each feature
feature_ranges = habituation_data.describe().loc[['max']].transpose()
feature_ranges.reset_index(inplace=True)
feature_ranges.columns = ['Feature', 'Max']

# Function to round up to the nearest 10
def round_up_to_nearest_10(value):
    return math.ceil(value / 10) * 10

# Map feature ranges to rounded max values and set min to 0
feature_range_dict = feature_ranges.set_index('Feature').to_dict(orient='index')
for feature, ranges in feature_range_dict.items():
    ranges['Min'] = 0  # Set min to 0
    ranges['Max'] = round_up_to_nearest_10(ranges['Max'])  # Round max to nearest 10

# Define the 20 habituation data features used for training
features = [
    "S1 poke event", "S2 poke event", "M1 poke event", "M2 poke event",
    "M3 poke event", "Sp1 corner poke event", "Sp2 corner poke event", "Door event",
    "Match Box event", "Inactive event", "S1 poke duration", "S2 poke duration",
    "M1 poke duration", "M2 poke duration", "M3 poke duration", "Sp1 corner poke duration",
    "Sp2 corner poke duration", "Door duration", "Match Box duration", "Inactive duration"
]

st.markdown(
    """
    <style>
    /* Global Styles */
    body, .main, .stApp {
        background-color: #000000; /* Black background */
        color: #0000ff; /* Dark blue text */
    }
    h1, h2, h3, h4, h5, h6, .stText {
        color: #0000ff; /* Dark blue headers and text */
    }
    .stButton>button {
        color: #ffffff;
        background-color: #0000ff; /* Dark blue buttons */
        border: none;
        padding: 12px 30px;
        font-size: 18px;
        cursor: pointer;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0073e6; /* Lighter blue hover effect */
        transform: scale(1.1);
    }
    .stSlider label {
        color: #0000ff; /* Dark blue labels for sliders */
    }
    .stSlider .st-uy {
        background: #333333 !important; /* Dark slider background */
    }
    .block-container {
        padding-top: 0px; /* Remove padding to make top part black */
    }

    /* Title-Specific Styles */
    .title-container {
        margin-top: 100px; /* Adjust the value to move the title further down */
        text-align: center; /* Center-align the title */
    }
    h1 {
        color: #0000ff; /* Dark blue text for the title */
    }
    </style>
    <div class="title-container">
        <h1>Rat Performance Prediction App</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# Include "Session" as a feature for scaling
all_features = features + ["Session"]

# Load ensemble models
def load_ensemble_models():
    try:
        ensemble_models = joblib.load("performance_classifier_ensemble.pkl")
        return ensemble_models
    except FileNotFoundError:
        st.error("Ensemble model file not found. Please ensure the correct model file is in the directory.")
        return None

# Function to make predictions using the ensemble
def predict_with_ensemble(input_data, ensemble_models):
    predictions = []
    feature_importances = []
    for model_info in ensemble_models:
        rf_model = model_info["model"]
        scaler = model_info["scaler"]
        rfe = model_info["rfe"]

        # Scale and select features using RFE
        input_scaled = scaler.transform(input_data)
        input_selected = input_scaled[:, rfe.support_]

        # Predict with the model
        prediction = rf_model.predict(input_selected)
        predictions.append(prediction[0])

        # Get feature importances
        importance = rf_model.feature_importances_
        full_importances = np.zeros(len(all_features))
        full_importances[rfe.support_] = importance
        feature_importances.append(full_importances)

    # Aggregate ensemble predictions (majority vote)
    final_prediction = int(np.round(np.mean(predictions)))

    # Average feature importances across models
    mean_importances = np.mean(feature_importances, axis=0)
    return final_prediction, predictions, mean_importances

# Streamlit UI setup


st.header("Input Averages of Habituation Data")

# Create sliders dynamically for all 21 features (including "Session")
input_data = {}
for feature in all_features:
    if feature in feature_range_dict:
        min_val = 0  # Set all mins to 0
        max_val = feature_range_dict[feature]['Max']
    else:
        min_val, max_val = 0, 100  # Default values if feature is not in dataset

    # Default slider value set to midpoint of min and max
    default_val = (min_val + max_val) / 2
    input_data[feature] = st.slider(
        f"{feature} average",
        min_value=float(min_val),
        max_value=float(max_val),
        value=float(default_val),
        step=0.1
    )

# Prediction button
if st.button("Predict Performance"):
    st.header("Prediction Results")

    # Prepare input data for the model
    input_values = np.array([input_data[feature] for feature in all_features]).reshape(1, -1)

    # Load ensemble models
    ensemble_models = load_ensemble_models()
    if ensemble_models:
        try:
            # Make predictions using the ensemble
            final_prediction, predictions, mean_importances = predict_with_ensemble(input_values, ensemble_models)

            # Interpret final prediction
            performance = "PROFICIENT PERFORMANCE" if final_prediction == 0 else "LOWER PERFORMANCE"
            color = "green" if final_prediction == 0 else "red"  # Green for Proficient, Red for Lower Performance

            # Use HTML to style the performance message
            st.markdown(
                f"""
                <div style="text-align: center; font-size: 24px; color: {color}; font-weight: bold; margin-top: 20px;">
                    The predicted performance class for the rat is: <br><span style="font-size: 32px;">{performance}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Ensemble Votes Visualization
            st.subheader("Ensemble Vote Distribution")
            vote_counts = [predictions.count(0), predictions.count(1)]
            vote_labels = ["Proficient", "Lower Performance"]
            fig2, ax2 = plt.subplots()
            ax2.pie(vote_counts, labels=vote_labels, autopct="%1.1f%%", startangle=90, colors=['#00ff00', '#ff0000'])
            ax2.set_title("Vote Distribution", color='#0000ff')
            st.pyplot(fig2)

        except ValueError as e:
            st.error(f"Error during prediction: {e}")
