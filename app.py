import streamlit as st
import pandas as pd
import joblib

# Page config
st.set_page_config(page_title="Smart Car Valuation", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for sleek dark mode styling
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    /* Header Gradient */
    .gradient-text {
        background: -webkit-linear-gradient(45deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .subtext {
        color: #8b949e;
        font-size: 1.2rem;
        margin-bottom: 3rem;
        text-align: center;
    }
    
    /* Card style for inputs */
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        color: #c9d1d9;
    }
    
    /* Labels */
    .stSelectbox label, .stNumberInput label, .stTextInput label {
        color: #58a6ff !important;
        font-weight: 600;
    }

    /* Predict Button Gradient */
    .stButton>button {
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-size: 1.2rem;
        font-weight: bold;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #182848 0%, #4b6cb7 100%);
        box-shadow: 0 4px 15px rgba(0,0,0,0.4);
        color: white;
    }
    
    /* Result Box */
    .result-box {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #30363d;
        margin-top: 10px;
    }
    .result-title {
        color: #4ade80;
        font-size: 1.2rem;
        margin-bottom: 5px;
    }
    .result-price {
        color: #22c55e;
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Main UI Structure
st.markdown('<p class="gradient-text">Smart Car Valuation</p>', unsafe_allow_html=True)
st.markdown('<p class="subtext">Get a fast and reliable estimate based on your vehicle\'s specifications.</p>', unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    return joblib.load('car_price_prediction_model.pkl')

try:
    model = load_model()
except Exception as e:
    st.error("Model not found! Please run the training notebook first to generate the .pkl file.")
    st.stop()

# Layout for user input mimicking the sleek grid
col1, col2 = st.columns(2, gap="large")

with col1:
    make = st.selectbox("Make", ["Toyota", "Honda", "Ford", "BMW", "Mercedes-Benz", "Chevrolet", "Volkswagen", "Hyundai", "Nissan", "Audi"], index=8)
    model_name = st.selectbox("Model", ["GLE", "Camry", "Civic", "X5", "Tucson", "Golf", "Tahoe", "Equinox", "Sentra", "C-Class"], index=0)
    year = st.number_input("Year", min_value=1990, max_value=2025, value=2024)
    
    col1_sub1, col1_sub2 = st.columns(2)
    with col1_sub1:
        fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Hybrid", "Electric"], index=1)
    with col1_sub2:
        transmission = st.selectbox("Transmission", ["Automatic", "Manual"], index=0)
        
    col1_sub3, col1_sub4 = st.columns(2)
    with col1_sub3:
        engine_size = st.number_input("Engine Size (L)", min_value=0.5, max_value=8.0, value=2.30, step=0.1, format="%.2f")
    with col1_sub4:
        body_type = st.selectbox("Body Type", ["Sedan", "SUV", "Hatchback", "Coupe", "Truck", "Wagon"], index=1)
        
    mileage = st.number_input("Mileage (Miles)", min_value=0, max_value=500000, value=100)
    location = st.selectbox("Location (State)", ["CA", "TX", "NY", "FL", "IL", "OH", "GA", "PA", "NC", "MI"], index=4)

with col2:
    torque = st.number_input("Torque (NM)", min_value=50, max_value=1000, value=196)
    owners = st.number_input("Previous Owners", min_value=1, max_value=10, value=1)
    accident_history = st.selectbox("Accident History", ["No", "Yes"], index=0)
    accident_history_val = 0.0 if accident_history == "No" else 1.0
    
    col2_sub1, col2_sub2 = st.columns(2)
    with col2_sub1:
        service_history = st.selectbox("Service History", ["Full Service", "Partial Service", "No Service"], index=2)
    with col2_sub2:
        color = st.selectbox("Color", ["Black", "White", "Silver", "Red", "Blue", "Grey", "Other"], index=2)
        
    drivetrain = st.selectbox("Drive Train", ["FWD", "RWD", "AWD", "4WD"], index=2)
    
    col2_sub3, col2_sub4 = st.columns(2)
    with col2_sub3:
        horsepower = st.number_input("Horsepower (HP)", min_value=50, max_value=1000, value=186)
    with col2_sub4:
        fuel_efficiency = st.number_input("Fuel Efficiency (MPG)", min_value=5.0, max_value=150.0, value=30.0, format="%.2f")
    
    # Prediction Section
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    predict_clicked = st.button("Predict Price", key="predict")
    
    # Place result inside the right column
    if predict_clicked:
        input_data = {
            'Make': make, 'Model': model_name, 'Year': year, 'Fuel_Type': fuel_type,
            'Transmission': transmission, 'Engine_Size': engine_size, 'Mileage': mileage,
            'Horsepower': horsepower, 'Torque': torque, 'Owners': owners,
            'Accident_History': accident_history_val, 'Service_History': service_history,
            'Color': color, 'Body_Type': body_type, 'Drivetrain': drivetrain,
            'Fuel_Efficiency': fuel_efficiency, 'Location': location
        }
        input_df = pd.DataFrame([input_data])
        
        prediction = model.predict(input_df)[0]
        
        st.markdown(f"""
        <div class="result-box">
            <div class="result-title">Estimated Selling Price</div>
            <div class="result-price">${prediction:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; color: #8b949e; margin-top: 50px; font-size: 0.9rem;">
    <p>Powered by XGBoost AI Model</p>
</div>
""", unsafe_allow_html=True)
