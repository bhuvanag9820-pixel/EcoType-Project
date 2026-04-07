import streamlit as st
import numpy as np
import joblib

# Load model & encoder
model = joblib.load('models/final_model.pkl')
le = joblib.load('models/label_encoder.pkl')

st.title("🌿 Forest Cover Type Prediction")

# User Inputs
Elevation = st.number_input("Elevation", value=2500)
Aspect = st.number_input("Aspect", value=0)
Slope = st.number_input("Slope", value=10)

Horizontal_Distance_To_Hydrology = st.number_input("Distance to Hydrology", value=100)
Vertical_Distance_To_Hydrology = st.number_input("Vertical Distance to Hydrology", value=0)
Horizontal_Distance_To_Roadways = st.number_input("Distance to Roadways", value=500)

Hillshade_9am = st.number_input("Hillshade 9am", value=200)
Hillshade_Noon = st.number_input("Hillshade Noon", value=220)
Hillshade_3pm = st.number_input("Hillshade 3pm", value=150)

Horizontal_Distance_To_Fire_Points = st.number_input("Distance to Fire Points", value=1000)

Wilderness_Area = st.selectbox("Wilderness Area", [1, 2, 3, 4])
Soil_Type = st.selectbox("Soil Type", list(range(1, 41)))

if st.button("Predict"):

    features = np.array([[Elevation, Aspect, Slope,
                          Horizontal_Distance_To_Hydrology,
                          Vertical_Distance_To_Hydrology,
                          Horizontal_Distance_To_Roadways,
                          Hillshade_9am, Hillshade_Noon, Hillshade_3pm,
                          Horizontal_Distance_To_Fire_Points,
                          Wilderness_Area, Soil_Type]])

    # ✅ prediction happens HERE
    prediction = model.predict(features)

    # ✅ mapping also inside
    cover_type_map = {
        0: "Spruce/Fir",
        1: "Lodgepole Pine",
        2: "Ponderosa Pine",
        3: "Cottonwood/Willow",
        4: "Aspen",
        5: "Douglas-fir",
        6: "Krummholz"
    }

    st.success(f"🌲 Predicted Forest Type: {cover_type_map[prediction[0]]}")

   