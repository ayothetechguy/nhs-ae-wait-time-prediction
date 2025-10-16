"""
NHS A&E Wait Time Prediction - Streamlit Dashboard
===================================================
Interactive web application for A&E wait time predictions
Author: Ayoolumi Melehon
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import time
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="NHS A&E Wait Time Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with LARGER TEXT
st.markdown("""
    <style>
    /* Increase base font size */
    html, body, [class*="css"] {
        font-size: 18px !important;
    }
    
    .main-header {
        font-size: 3.2rem !important;
        color: #2563eb;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .sub-header {
        font-size: 1.6rem !important;
        color: #64748b;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Larger text in main content */
    .stMarkdown, .stText {
        font-size: 1.2rem !important;
    }
    
    /* Larger labels and input text */
    label, .stTextInput label, .stSelectbox label, .stSlider label {
        font-size: 1.3rem !important;
        font-weight: 600 !important;
    }
    
    input, select, .stTextInput input, .stSelectbox select {
        font-size: 1.2rem !important;
    }
    
    /* Larger metric values */
    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1.3rem !important;
    }
    
    /* Larger tabs */
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 1.3rem !important;
        padding: 1rem 2rem !important;
    }
    
    .metric-card {
        background-color: #f8fafc;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2563eb;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        margin: 2rem 0;
    }
    .stButton>button {
        background-color: #2563eb !important;
        color: white !important;
        font-weight: bold !important;
        font-size: 1.3rem !important;
        border-radius: 0.5rem !important;
        padding: 0.8rem 2rem !important;
        width: 100%;
    }
    
    /* Larger info boxes */
    .stAlert, .stInfo, .stSuccess, .stWarning, .stError {
        font-size: 1.2rem !important;
    }
    
    /* Larger expander text */
    .streamlit-expanderHeader {
        font-size: 1.3rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and components
@st.cache_resource
def load_model_components():
    """Load trained model and preprocessing components"""
    try:
        model = joblib.load('output/nhs_ae_model.pkl')
        scaler = joblib.load('output/scaler.pkl')
        le_mode = joblib.load('output/arrival_mode_encoder.pkl')
        le_complaint = joblib.load('output/complaint_encoder.pkl')
        
        with open('output/feature_columns.txt', 'r') as f:
            feature_columns = [line.strip() for line in f.readlines()]
        
        return model, scaler, le_mode, le_complaint, feature_columns
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None, None

model, scaler, le_mode, le_complaint, feature_columns = load_model_components()

# Sidebar navigation
st.sidebar.title("🏥 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Overview", "📊 Visualizations", "🔮 Live Predictions", "🏥 Patient Check-In", "📄 About Project"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Project Info")
st.sidebar.info(
    """
    **NHS A&E Wait Time Predictor**
    
    Machine Learning system for predicting Emergency Department wait times.
    
    **Author:** Ayoolumi Melehon  
    **Tech Stack:** Python, scikit-learn, Streamlit
    """
)

# ==============================================================================
# PAGE 1: OVERVIEW
# ==============================================================================

if page == "🏠 Overview":
    st.markdown('<p class="main-header">🏥 NHS A&E Wait Time Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Machine Learning System for Emergency Department Management</p>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Dataset Size",
            value="5,000",
            delta="Synthetic Records"
        )
    
    with col2:
        st.metric(
            label="🎯 Model Accuracy",
            value="85.67%",
            delta="R² Score"
        )
    
    with col3:
        st.metric(
            label="⏱️ MAE",
            value="17.66 min",
            delta="High Precision"
        )
    
    with col4:
        st.metric(
            label="🤖 Algorithm",
            value="Gradient Boosting",
            delta="Best Model"
        )
    
    st.markdown("---")
    
    # Project overview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📋 Project Overview")
        st.markdown("""
        This machine learning system predicts A&E (Accident & Emergency) wait times based on multiple factors:
        
        **🔑 Key Features:**
        - Real-time wait time predictions
        - 13 predictive features including patient demographics, triage category, and department status
        - 85.67% prediction accuracy
        - Interactive patient check-in simulation
        
        **📊 Model Performance:**
        - **Mean Absolute Error:** 17.66 minutes
        - **Root Mean Squared Error:** 21.95 minutes
        - **R² Score:** 0.8567 (85.67% variance explained)
        
        **🎯 Use Cases:**
        - Hospital resource planning
        - Patient communication systems
        - Department capacity management
        - Triage optimization
        """)
    
    with col2:
        st.markdown("### 🔬 Technical Stack")
        st.code("""
        Language: Python 3.13
        
        ML Libraries:
        • scikit-learn
        • pandas
        • numpy
        
        Models:
        • Random Forest
        • Gradient Boosting
        
        Visualization:
        • matplotlib
        • seaborn
        • plotly
        
        Deployment:
        • Streamlit
        """, language="python")
    
    st.markdown("---")
    
    # Features breakdown
    st.markdown("### 🎯 Predictive Features (13 Total)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**⏰ Temporal Features**")
        st.markdown("""
        - Hour of day
        - Day of week
        - Month
        - Weekend flag
        - Peak hour flag
        - Night shift flag
        """)
    
    with col2:
        st.markdown("**👤 Patient Features**")
        st.markdown("""
        - Age
        - Triage category (1-5)
        - Arrival mode
        - Chief complaint
        - Previous visits (30d)
        """)
    
    with col3:
        st.markdown("**🏥 Department Features**")
        st.markdown("""
        - Department occupancy %
        - Staff available
        """)

# ==============================================================================
# PAGE 2: VISUALIZATIONS
# ==============================================================================

elif page == "📊 Visualizations":
    st.markdown('<p class="main-header">📊 Data Visualizations & Analysis</p>', unsafe_allow_html=True)
    
    # Check if output folder exists
    if not os.path.exists('output'):
        st.error("⚠️ Output folder not found. Please run nhs_ae_model.py first.")
    else:
        # Display all visualizations
        viz_files = [
            ('01_wait_time_distribution.png', 'Wait Time Distribution', 
             'Shows the distribution of wait times across all patients and by triage category.'),
            ('02_wait_time_by_hour.png', 'Wait Time by Hour of Day',
             'Average wait times throughout the day, showing peak and off-peak patterns.'),
            ('03_occupancy_impact.png', 'Occupancy & Staff Impact',
             'Relationship between department occupancy, staff levels, and wait times.'),
            ('04_arrival_mode_analysis.png', 'Arrival Mode Analysis',
             'Wait times and distribution by patient arrival method.'),
            ('05_feature_importance.png', 'Feature Importance',
             'Which factors have the biggest impact on predicting wait times.'),
            ('06_actual_vs_predicted.png', 'Model Performance',
             'How well the model predictions match actual wait times.')
        ]
        
        for viz_file, title, description in viz_files:
            viz_path = f'output/{viz_file}'
            
            if os.path.exists(viz_path):
                st.markdown(f"### {title}")
                st.markdown(f"*{description}*")
                
                try:
                    image = Image.open(viz_path)
                    st.image(image, use_column_width=True)
                except Exception as e:
                    st.error(f"Error loading {viz_file}: {e}")
                
                st.markdown("---")
            else:
                st.warning(f"⚠️ {viz_file} not found")

# ==============================================================================
# PAGE 3: LIVE PREDICTIONS
# ==============================================================================

elif page == "🔮 Live Predictions":
    st.markdown('<p class="main-header">🔮 Live Wait Time Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Enter patient details to predict A&E wait time</p>', unsafe_allow_html=True)
    
    if model is None:
        st.error("⚠️ Model not loaded. Please ensure all model files are in the output folder.")
    else:
        # Prediction form
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 👤 Patient Information")
            
            age = st.slider("Patient Age", 18, 90, 45)
            
            triage = st.select_slider(
                "Triage Category",
                options=[1, 2, 3, 4, 5],
                value=3,
                help="1 = Immediate, 5 = Non-urgent"
            )
            
            arrival_mode = st.selectbox(
                "Arrival Mode",
                ["Walk-in", "Ambulance", "GP Referral"]
            )
            
            complaint = st.selectbox(
                "Chief Complaint",
                ['Chest Pain', 'Breathing Difficulty', 'Abdominal Pain',
                 'Injury/Trauma', 'Fever', 'Mental Health', 'Other']
            )
            
            previous_visits = st.number_input(
                "Previous A&E Visits (Last 30 Days)",
                min_value=0, max_value=10, value=0
            )
        
        with col2:
            st.markdown("### 🏥 Department Status")
            
            now = datetime.now()
            hour = st.slider("Hour of Day", 0, 23, now.hour)
            
            day_of_week = st.selectbox(
                "Day of Week",
                ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                index=now.weekday()
            )
            
            day_map = {
                "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
                "Friday": 4, "Saturday": 5, "Sunday": 6
            }
            day_num = day_map[day_of_week]
            
            occupancy = st.slider(
                "Department Occupancy (%)",
                30, 98, 70,
                help="Current department capacity utilization"
            )
            
            staff = st.slider(
                "Staff Available",
                8, 30, 15,
                help="Number of medical staff currently on duty"
            )
        
        st.markdown("---")
        
        # Predict button
        if st.button("🔮 Predict Wait Time", type="primary"):
            # Calculate derived features
            month = now.month
            is_weekend = 1 if day_num >= 5 else 0
            is_peak_hour = 1 if (10 <= hour <= 13) or (18 <= hour <= 21) else 0
            is_night = 1 if 0 <= hour <= 6 else 0
            
            # Encode categorical variables
            try:
                arrival_encoded = le_mode.transform([arrival_mode])[0]
                complaint_encoded = le_complaint.transform([complaint])[0]
                
                # Create feature vector
                features = pd.DataFrame({
                    'hour': [hour],
                    'day_of_week': [day_num],
                    'month': [month],
                    'is_weekend': [is_weekend],
                    'is_peak_hour': [is_peak_hour],
                    'is_night': [is_night],
                    'triage_category': [triage],
                    'age': [age],
                    'arrival_mode_encoded': [arrival_encoded],
                    'complaint_encoded': [complaint_encoded],
                    'department_occupancy': [occupancy],
                    'staff_available': [staff],
                    'previous_visits_30d': [previous_visits]
                })
                
                # Scale and predict
                features_scaled = scaler.transform(features)
                prediction = model.predict(features_scaled)[0]
                
                # Display prediction
                st.markdown(f"""
                    <div class="prediction-box">
                        <h2 style="font-size: 2rem;">⏱️ Predicted Wait Time</h2>
                        <h1 style="font-size: 5rem; margin: 1rem 0;">{prediction:.0f} minutes</h1>
                        <h3 style="font-size: 2rem;">({prediction/60:.1f} hours)</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                # Additional context
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if prediction < 90:
                        st.success("🟢 **Short Wait**\n\nBelow average wait time")
                    elif prediction < 180:
                        st.warning("🟡 **Moderate Wait**\n\nAverage wait time")
                    else:
                        st.error("🔴 **Long Wait**\n\nAbove average wait time")
                
                with col2:
                    triage_desc = {
                        1: "Immediate - Life threatening",
                        2: "Very Urgent - Critical condition",
                        3: "Urgent - Serious condition",
                        4: "Standard - Non-critical",
                        5: "Non-urgent - Minor condition"
                    }
                    st.info(f"**Triage Level {triage}**\n\n{triage_desc[triage]}")
                
                with col3:
                    if occupancy > 80:
                        dept_status = "🔴 Very Busy"
                    elif occupancy > 60:
                        dept_status = "🟡 Busy"
                    else:
                        dept_status = "🟢 Normal"
                    st.info(f"**Department Status**\n\n{dept_status}")
                
            except Exception as e:
                st.error(f"Prediction error: {e}")

# ==============================================================================
# PAGE 4: PATIENT CHECK-IN SYSTEM
# ==============================================================================

elif page == "🏥 Patient Check-In":
    st.markdown('<p class="main-header">🏥 Patient Self-Service Check-In</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Complete your A&E check-in and get your predicted wait time</p>', unsafe_allow_html=True)
    
    if model is None:
        st.error("⚠️ AI Prediction System offline. Please ensure model files are available.")
    else:
        # Initialize session state for tab navigation
        if 'current_tab' not in st.session_state:
            st.session_state.current_tab = 0
        
        # Welcome message
        with st.expander("⚠️ IMPORTANT SAFETY INFORMATION - Click to Read", expanded=False):
            st.warning("""
            **If you have any of these symptoms, please alert reception staff IMMEDIATELY:**
            - Severe chest pain
            - Difficulty breathing
            - Heavy bleeding
            - Loss of consciousness
            - Stroke symptoms (face drooping, arm weakness, speech difficulty)
            """)
        
        st.markdown("---")
        
        # Create tabs for check-in steps
        tab1, tab2, tab3, tab4 = st.tabs(["👤 Personal Info", "🚨 Urgency", "🏥 Medical Info", "✅ Complete Check-In"])
        
        # Initialize session state
        if 'checkin_data' not in st.session_state:
            st.session_state.checkin_data = {}
        
        # TAB 1: Personal Information
        with tab1:
            st.markdown("### 👤 Your Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Full Name *", value=st.session_state.checkin_data.get('name', ''))
                
                dob = st.date_input(
                    "Date of Birth *",
                    value=None,
                    min_value=datetime(1920, 1, 1),
                    max_value=datetime.now()
                )
                
                if dob:
                    age = (datetime.now() - datetime.combine(dob, datetime.min.time())).days // 365
                    st.info(f"Age: {age} years")
                else:
                    age = None
                
                postcode = st.text_input("Postcode *", value=st.session_state.checkin_data.get('postcode', ''))
            
            with col2:
                nhs_number = st.text_input("NHS Number (optional)", value=st.session_state.checkin_data.get('nhs_number', ''))
                phone = st.text_input("Contact Number *", value=st.session_state.checkin_data.get('phone', ''))
            
            st.markdown("### 👥 Emergency Contact")
            
            col3, col4 = st.columns(2)
            
            with col3:
                emergency_name = st.text_input("Emergency Contact Name *", value=st.session_state.checkin_data.get('emergency_name', ''))
            
            with col4:
                emergency_phone = st.text_input("Emergency Contact Phone *", value=st.session_state.checkin_data.get('emergency_phone', ''))
            
            if st.button("💾 Save & Continue to Urgency Assessment", key="save_personal"):
                if name and dob and postcode and phone and emergency_name and emergency_phone:
                    st.session_state.checkin_data.update({
                        'name': name,
                        'age': age,
                        'dob': dob,
                        'postcode': postcode,
                        'phone': phone,
                        'nhs_number': nhs_number or 'Temporary',
                        'emergency_name': emergency_name,
                        'emergency_phone': emergency_phone
                    })
                    st.success("✅ Personal information saved! Moving to Urgency Assessment...")
                    time.sleep(1)
                    st.session_state.current_tab = 1
                    st.rerun()
                else:
                    st.error("❌ Please fill in all required fields (marked with *)")
        
        # TAB 2: Urgency Assessment
        with tab2:
            st.markdown("### 🚨 Urgency Self-Assessment")
            st.info("This helps us prioritize your care appropriately.")
            
            pain_level = st.slider(
                "📊 Current Pain Level",
                0, 10, 5,
                help="0 = No pain, 10 = Worst pain imaginable"
            )
            
            if pain_level >= 8:
                st.error("🔴 Severe pain detected - High priority")
            elif pain_level >= 5:
                st.warning("🟡 Moderate pain - Standard priority")
            else:
                st.success("🟢 Mild pain - Will be assessed")
            
            duration = st.selectbox(
                "⏱️ How long have you had these symptoms?",
                ["Less than 1 hour", "1-6 hours", "6-24 hours", "More than 24 hours"]
            )
            
            st.markdown("### 🩺 Do you have any of these symptoms?")
            
            col1, col2 = st.columns(2)
            
            with col1:
                chest_pain = st.checkbox("Chest tightness or pressure")
                headache = st.checkbox("Severe headache with vision changes")
                vomiting = st.checkbox("Persistent vomiting")
            
            with col2:
                fever = st.checkbox("High fever (over 39°C)")
                abdominal = st.checkbox("Severe abdominal pain")
                wound = st.checkbox("Deep cut or wound")
            
            mobility = st.radio("🚶 Can you walk without assistance?", ["Yes", "No"])
            
            # Calculate triage score
            triage_score = 5
            
            if pain_level >= 8:
                triage_score = min(triage_score, 3)
            elif pain_level >= 5:
                triage_score = min(triage_score, 4)
            
            if duration == "Less than 1 hour" and pain_level >= 6:
                triage_score = min(triage_score, 3)
            
            if chest_pain or headache:
                triage_score = min(triage_score, 2)
            if vomiting or fever or abdominal:
                triage_score = min(triage_score, 3)
            if wound:
                triage_score = min(triage_score, 4)
            if mobility == "No":
                triage_score = min(triage_score, 3)
            
            triage_colors = {1: "🔴", 2: "🟠", 3: "🟡", 4: "🟢", 5: "🔵"}
            triage_labels = {
                1: "IMMEDIATE - Life threatening",
                2: "VERY URGENT - Critical condition",
                3: "URGENT - Serious condition",
                4: "STANDARD - Non-critical",
                5: "NON-URGENT - Minor condition"
            }
            
            st.info(f"{triage_colors[triage_score]} **Assessed Triage Category {triage_score}:** {triage_labels[triage_score]}")
            
            if st.button("💾 Save & Continue to Medical Information", key="save_urgency"):
                st.session_state.checkin_data.update({
                    'triage': triage_score,
                    'pain_level': pain_level,
                    'duration': duration,
                    'symptoms': {
                        'chest_pain': chest_pain,
                        'headache': headache,
                        'vomiting': vomiting,
                        'fever': fever,
                        'abdominal': abdominal,
                        'wound': wound
                    },
                    'mobility': mobility
                })
                st.success("✅ Urgency assessment saved! Moving to Medical Information...")
                time.sleep(1)
                st.session_state.current_tab = 2
                st.rerun()
        
        # TAB 3: Medical Information
        with tab3:
            st.markdown("### 🏥 Medical Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                complaint = st.selectbox(
                    "📝 Main reason for your visit *",
                    ['Chest Pain', 'Breathing Difficulty', 'Abdominal Pain',
                     'Injury/Trauma', 'Fever', 'Mental Health', 'Other']
                )
                
                arrival_mode = st.selectbox(
                    "🚑 How did you arrive? *",
                    ["Walked in myself", "Brought by family/friend", "Ambulance", "GP referred me"]
                )
                
                # Map to model categories
                if arrival_mode in ["Walked in myself", "Brought by family/friend"]:
                    arrival_mode_model = "Walk-in"
                elif arrival_mode == "Ambulance":
                    arrival_mode_model = "Ambulance"
                else:
                    arrival_mode_model = "GP Referral"
            
            with col2:
                conditions = st.text_area("💊 Existing medical conditions", placeholder="e.g., Diabetes, Asthma (or None)")
                medications = st.text_area("💊 Current medications", placeholder="e.g., Insulin, Ventolin (or None)")
                allergies = st.text_area("⚠️ Known allergies", placeholder="e.g., Penicillin, Latex (or None)")
            
            previous_visits = st.radio("📅 Have you visited A&E in the last 30 days?", ["No", "Yes"])
            previous_visits_num = 1 if previous_visits == "Yes" else 0
            
            if st.button("💾 Save & Continue to Final Step", key="save_medical"):
                if complaint and arrival_mode:
                    st.session_state.checkin_data.update({
                        'complaint': complaint,
                        'arrival_mode': arrival_mode_model,
                        'conditions': conditions or 'None',
                        'medications': medications or 'None',
                        'allergies': allergies or 'None',
                        'previous_visits': previous_visits_num
                    })
                    st.success("✅ Medical information saved! Moving to Complete Check-In...")
                    time.sleep(1)
                    st.session_state.current_tab = 3
                    st.rerun()
                else:
                    st.error("❌ Please fill in all required fields (marked with *)")
        
        # TAB 4: Complete Check-In
        with tab4:
            st.markdown("### ✅ Review & Complete Check-In")
            
            # Check if all required data is present
            required_fields = ['name', 'age', 'triage', 'complaint', 'arrival_mode']
            missing_fields = [field for field in required_fields if field not in st.session_state.checkin_data]
            
            if missing_fields:
                st.warning("⚠️ Please complete all previous tabs before checking in.")
                st.info(f"Missing information: {', '.join(missing_fields)}")
            else:
                # Display summary
                st.markdown("### 📋 Your Information Summary")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    **Personal Details:**
                    - Name: {st.session_state.checkin_data.get('name')}
                    - Age: {st.session_state.checkin_data.get('age')} years
                    - Phone: {st.session_state.checkin_data.get('phone')}
                    
                    **Medical:**
                    - Chief Complaint: {st.session_state.checkin_data.get('complaint')}
                    - Arrival Mode: {st.session_state.checkin_data.get('arrival_mode')}
                    """)
                
                with col2:
                    triage = st.session_state.checkin_data.get('triage')
                    st.markdown(f"""
                    **Urgency Assessment:**
                    - Triage Category: {triage}
                    - Pain Level: {st.session_state.checkin_data.get('pain_level')}/10
                    - Can walk: {st.session_state.checkin_data.get('mobility')}
                    """)
                
                st.markdown("---")
                
                # Complete Check-In Button
                if st.button("🎫 COMPLETE CHECK-IN & GET WAIT TIME", type="primary", key="complete_checkin"):
                    with st.spinner("⏳ Processing your check-in and calculating wait time using AI..."):
                        time.sleep(1.5)
                        
                        # Get current time and calculate features
                        now = datetime.now()
                        hour = now.hour
                        day = now.weekday()
                        month = now.month
                        
                        # Simulate department status
                        base_occupancy = 65
                        if hour in range(10, 14) or hour in range(18, 22):
                            base_occupancy += 20
                        if day in [5, 6]:
                            base_occupancy += 10
                        occupancy = min(95, base_occupancy + np.random.randint(-5, 10))
                        staff = max(8, 25 - occupancy // 5 + np.random.randint(-2, 3))
                        
                        # Calculate temporal features
                        is_weekend = 1 if day in [5, 6] else 0
                        is_peak = 1 if hour in range(10, 14) or hour in range(18, 22) else 0
                        is_night = 1 if hour in range(0, 7) else 0
                        
                        # Encode categorical variables
                        try:
                            arrival_encoded = le_mode.transform([st.session_state.checkin_data['arrival_mode']])[0]
                            complaint_encoded = le_complaint.transform([st.session_state.checkin_data['complaint']])[0]
                            
                            # Create feature vector
                            features = pd.DataFrame({
                                'hour': [hour],
                                'day_of_week': [day],
                                'month': [month],
                                'is_weekend': [is_weekend],
                                'is_peak_hour': [is_peak],
                                'is_night': [is_night],
                                'triage_category': [st.session_state.checkin_data['triage']],
                                'age': [st.session_state.checkin_data['age']],
                                'arrival_mode_encoded': [arrival_encoded],
                                'complaint_encoded': [complaint_encoded],
                                'department_occupancy': [occupancy],
                                'staff_available': [staff],
                                'previous_visits_30d': [st.session_state.checkin_data['previous_visits']]
                            })
                            
                            # Predict wait time
                            features_scaled = scaler.transform(features)
                            predicted_wait = model.predict(features_scaled)[0]
                            predicted_wait = max(5, predicted_wait + np.random.normal(0, 10))
                            
                            # Generate reference number
                            ref_num = f"AE{now.strftime('%Y%m%d')}{np.random.randint(1000, 9999)}"
                            queue_pos = np.random.randint(5, 25)
                            
                            # Display results
                            st.success("✅ CHECK-IN COMPLETE!")
                            
                            # Prediction box
                            hours = int(predicted_wait // 60)
                            minutes = int(predicted_wait % 60)
                            
                            if hours > 0:
                                wait_display = f"{hours} hour{'s' if hours > 1 else ''} {minutes} minutes"
                            else:
                                wait_display = f"{minutes} minutes"
                            
                            st.markdown(f"""
                                <div class="prediction-box">
                                    <h2 style="font-size: 2rem;">⏱️ Your Estimated Wait Time</h2>
                                    <h1 style="font-size: 5rem; margin: 1rem 0;">{wait_display}</h1>
                                    <p style="font-size: 1.5rem;">Predicted using AI based on current conditions</p>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            # Ticket information
                            st.markdown("### 🎫 Your Digital Ticket")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Reference Number", ref_num)
                                st.metric("Check-in Time", now.strftime('%H:%M'))
                            
                            with col2:
                                st.metric("Queue Position", f"~{queue_pos} patients")
                                st.metric("Department Occupancy", f"{occupancy:.0f}%")
                            
                            with col3:
                                triage_labels_short = {1: "IMMEDIATE", 2: "VERY URGENT", 3: "URGENT", 4: "STANDARD", 5: "NON-URGENT"}
                                st.metric("Triage Priority", triage_labels_short[st.session_state.checkin_data['triage']])
                                st.metric("Staff Available", f"{staff} staff")
                            
                            # Status indicator
                            st.markdown("### 📊 Current Department Status")
                            
                            if predicted_wait > 240:
                                st.error("🔴 **Long Wait Expected** - Department is very busy. Consider alternatives if non-urgent.")
                            elif predicted_wait > 120:
                                st.warning("🟡 **Moderate Wait** - Department is busy. Normal wait times.")
                            else:
                                st.success("🟢 **Short Wait** - Below average wait time.")
                            
                            # Instructions
                            st.markdown("---")
                            st.markdown("### ℹ️ What to Do Next")
                            
                            st.info("""
                            **📱 Your reference number will be called on the display screens**
                            
                            While you wait:
                            - 🏪 Refreshments available in the café (Ground Floor)
                            - 🚻 Toilets located near waiting area
                            - 💊 Pharmacy open until 20:00
                            - 📞 Free WiFi: NHS_Guest
                            
                            ⚠️ **Please inform staff immediately if:**
                            - Your condition worsens
                            - You develop new symptoms
                            - You need urgent assistance
                            """)
                            
                            # Download ticket button
                            st.markdown("---")
                            
                            ticket_text = f"""
NHS A&E DIGITAL TICKET
=================================
Reference: {ref_num}
Name: {st.session_state.checkin_data['name']}
Time: {now.strftime('%d/%m/%Y %H:%M')}
Estimated Wait: {predicted_wait:.0f} minutes
Triage: Category {st.session_state.checkin_data['triage']}
=================================
Keep this ticket with you.
Your reference will appear on screens.
                            """
                            
                            st.download_button(
                                label="📄 Download Ticket (Text)",
                                data=ticket_text,
                                file_name=f"ae_ticket_{ref_num}.txt",
                                mime="text/plain"
                            )
                            
                        except Exception as e:
                            st.error(f"❌ Error during check-in: {e}")
                            st.info("Please try again or contact reception for assistance.")

# ==============================================================================
# PAGE 5: ABOUT PROJECT
# ==============================================================================

elif page == "📄 About Project":
    st.markdown('<p class="main-header">📄 About This Project</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 🎯 Project Objective
    
    This project demonstrates the application of machine learning to healthcare operations management,
    specifically predicting A&E (Accident & Emergency) wait times based on patient and department factors.
    
    ## 🔬 Methodology
    
    ### 1. Data Generation
    - Created 5,000 synthetic patient records using realistic distributions
    - Ensured data reflects real-world A&E patterns
    - Privacy-compliant synthetic data for portfolio demonstration
    
    ### 2. Feature Engineering
    Developed 13 predictive features across three categories:
    - **Temporal:** Hour, day, month, peak periods
    - **Patient:** Age, triage, arrival mode, complaint, history
    - **Department:** Occupancy levels, staff availability
    
    ### 3. Model Development
    - Trained two ensemble models: Random Forest and Gradient Boosting
    - Used 80/20 train-test split
    - Applied StandardScaler for feature normalization
    - Selected best model based on MAE (Mean Absolute Error)
    
    ### 4. Model Evaluation
    - **Gradient Boosting** selected as best model
    - MAE: 17.66 minutes (high precision)
    - RMSE: 21.95 minutes
    - R² Score: 0.8567 (85.67% variance explained)
    
    ## 💡 Key Insights
    
    **Most Important Predictive Features:**
    1. Department occupancy percentage
    2. Triage category
    3. Staff availability
    4. Hour of day
    5. Patient age
    
    **Patterns Discovered:**
    - Peak hours (10-13, 18-21) show 30-40% longer wait times
    - Ambulance arrivals prioritized, reducing wait by ~30 minutes
    - Higher triage categories (1-2) get faster attention
    - Weekend occupancy increases wait times by 15-20%
    
    ## 🎯 Business Value
    
    **For Hospital Management:**
    - Optimize staff scheduling based on predicted demand
    - Improve resource allocation
    - Better capacity planning
    
    **For Patients:**
    - Transparent wait time expectations
    - Informed decision-making about seeking care
    - Reduced anxiety through communication
    
    **For Operational Efficiency:**
    - Data-driven triage optimization
    - Identify bottlenecks in patient flow
    - Evidence-based process improvements
    
    ## 🔮 Future Enhancements
    
    - Integration with real hospital data (with proper approvals)
    - Real-time dashboard for hospital staff
    - SMS/email patient notifications
    - Mobile app for patient self-service check-in
    - Integration with NHS systems
    - Expand to multiple departments (X-ray, labs, etc.)
    
    ## 📊 Technical Implementation
```python
    # Model Architecture
    Gradient Boosting Regressor
    - n_estimators: 100
    - max_depth: 7
    - learning_rate: 0.1
    - random_state: 42
    
    # Performance Metrics
    MAE: 17.66 minutes
    RMSE: 21.95 minutes
    R²: 0.8567
```
    
    ## 👨‍💻 About the Author
    
    **Ayoolumi Melehon**
    - MSc Artificial Intelligence (University of Stirling, 2025)
    - CompTIA Data+ Certified
    - Specialization: Data Analytics, ML/AI, Healthcare Analytics
    - Location: Edinburgh, Scotland, UK
    
    ## 📧 Contact & Social Links
    
    - **Email:** [ayoolumimelehon@gmail.com](mailto:ayoolumimelehon@gmail.com)
    - **LinkedIn:** [linkedin.com/in/ayoolumi-melehon](https://linkedin.com/in/ayoolumi-melehon)
    - **GitHub:** [github.com/ayothetechguy](https://github.com/ayothetechguy)
    - **Twitter/X:** [@ayo_olumi](https://twitter.com/ayo_olumi)
    - **Portfolio:** [ayofemimelehon.info](https://ayofemimelehon.info)
    
    ---
    
    *This project uses synthetic data to demonstrate capabilities while maintaining
    privacy and data protection standards. For real-world implementation, proper NHS
    data governance and ethical approvals would be required.*
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #64748b; padding: 2rem; font-size: 1.1rem;'>
        <p><strong style="font-size: 1.3rem;">NHS A&E Wait Time Prediction System</strong></p>
        <p>Built with Python, scikit-learn, and Streamlit</p>
        <p>© 2025 Ayoolumi Melehon</p>
        <p style="margin-top: 1rem;">
            <a href="https://linkedin.com/in/ayoolumi-melehon" target="_blank" style="margin: 0 10px;">LinkedIn</a> |
            <a href="https://github.com/ayothetechguy" target="_blank" style="margin: 0 10px;">GitHub</a> |
            <a href="https://twitter.com/ayo_olumi" target="_blank" style="margin: 0 10px;">Twitter</a> |
            <a href="mailto:ayoolumimelehon@gmail.com" style="margin: 0 10px;">Email</a>
        </p>
    </div>
""", unsafe_allow_html=True)