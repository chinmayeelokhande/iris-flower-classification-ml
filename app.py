import streamlit as st
import numpy as np
import joblib

if "prediction" not in st.session_state:
    st.session_state.prediction = None

iris_images = {
    "Iris-setosa": "https://upload.wikimedia.org/wikipedia/commons/a/a7/Iris_setosa.jpg",
    "Iris-versicolor": "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",
    "Iris-virginica": "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg"
}

# Load model and scaler
scaler = joblib.load("scaler.joblib")
model = joblib.load("logistic_regression_best_model.joblib")

# Page config
st.set_page_config(page_title="Iris Classification App", layout="wide")

# Title
st.title("🌸 Iris Flower Species Classification App")
st.write(
    "Predict the **species of an Iris flower** using its physical measurements."
)

# Sidebar inputs
st.sidebar.header("🌿 Input Features")

sepal_length = st.sidebar.slider("Sepal Length (cm)", 0.0, 8.0, 5.0)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 0.0, 8.0, 3.0)
petal_length = st.sidebar.slider("Petal Length (cm)", 0.0, 8.0, 4.0)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.0, 8.0, 1.0)

# Main layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Selected Feature Values")
    st.write({
        "Sepal Length": sepal_length,
        "Sepal Width": sepal_width,
        "Petal Length": petal_length,
        "Petal Width": petal_width
    })

with col2:
    image_placeholder = st.empty()


# Prediction
if st.session_state.prediction:
    image_placeholder.image(
        iris_images[st.session_state.prediction],
        caption=st.session_state.prediction,
        width=300
    )
else:
    image_placeholder.image(
        "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",
        caption="Iris Flower",
        width=300
    )

if st.button("🔍 Predict Species"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    st.session_state.prediction = prediction

    st.success(f"🌼 Predicted Iris Species: {prediction}")

