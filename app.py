import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Car Price Predictor", layout="centered")

st.title("🚗 Intelligent Car Price Prediction System")
st.write("Enter the vehicle specifications below to estimate its selling price using our tuned XGBoost AI Model.")

@st.cache_resource
def load_model():
    return joblib.load('car_price_prediction_model.pkl')

try:
    model = load_model()
except Exception as e:
    st.error("Model not found! Please run `python train_model.py` first to generate the .pkl file.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    make = st.selectbox("Make", ["Toyota", "Honda", "Ford", "BMW", "Mercedes-Benz", "Chevrolet", "Volkswagen", "Hyundai", "Nissan", "Audi"])
    model_name = st.text_input("Model (e.g., Camry, Civic, X5)", "Camry")
    year = st.number_input("Manufacturing Year", min_value=1990, max_value=2025, value=2015)
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Hybrid", "Electric"])
    transmission = st.selectbox("Transmission", ["Automatic", "Manual"])
    engine_size = st.number_input("Engine Size (L)", min_value=0.5, max_value=8.0, value=2.0, step=0.1)
    mileage = st.number_input("Mileage (Miles)", min_value=0, max_value=500000, value=50000)
    horsepower = st.number_input("Horsepower", min_value=50, max_value=1000, value=150)

with col2:
    torque = st.number_input("Torque", min_value=50, max_value=1000, value=150)
    owners = st.number_input("Previous Owners", min_value=1, max_value=10, value=1)
    accident_history = st.selectbox("Accident History", [0.0, 1.0], format_func=lambda x: "Yes" if x == 1.0 else "No")
    service_history = st.selectbox("Service History", ["Full Service", "Partial Service", "No Service"])
    color = st.selectbox("Color", ["Black", "White", "Silver", "Red", "Blue", "Grey", "Other"])
    body_type = st.selectbox("Body Type", ["Sedan", "SUV", "Hatchback", "Coupe", "Truck", "Wagon"])
    drivetrain = st.selectbox("Drivetrain", ["FWD", "RWD", "AWD", "4WD"])
    fuel_efficiency = st.number_input("Fuel Efficiency (MPG)", min_value=5.0, max_value=150.0, value=30.0)
    location = st.selectbox("Location (State)", ["CA", "TX", "NY", "FL", "IL", "OH", "GA", "PA", "NC", "MI"])

if st.button("Predict Price", type="primary"):
    input_data = {
        'Make': make, 'Model': model_name, 'Year': year, 'Fuel_Type': fuel_type,
        'Transmission': transmission, 'Engine_Size': engine_size, 'Mileage': mileage,
        'Horsepower': horsepower, 'Torque': torque, 'Owners': owners,
        'Accident_History': float(accident_history), 'Service_History': service_history,
        'Color': color, 'Body_Type': body_type, 'Drivetrain': drivetrain,
        'Fuel_Efficiency': fuel_efficiency, 'Location': location
    }
    
    input_df = pd.DataFrame([input_data])
    
    with st.spinner("Calculating..."):
        prediction = model.predict(input_df)[0]
    
    st.success(f"### Estimated Selling Price: ${prediction:,.2f}")
