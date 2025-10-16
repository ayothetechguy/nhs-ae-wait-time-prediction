"""
NHS A&E SELF-SERVICE PATIENT CHECK-IN SYSTEM
=============================================
Interactive patient check-in with real-time wait time predictions
Author: Ayoolumi Melehon
"""

import pandas as pd
import numpy as np
from datetime import datetime
import time
import joblib

print("\n" + "="*80)
print("🏥 NHS A&E DIGITAL CHECK-IN SYSTEM")
print("="*80)
print("Welcome to the Emergency Department")
print("="*80)

# Load trained model and components
try:
    model = joblib.load('output/nhs_ae_model.pkl')
    scaler = joblib.load('output/scaler.pkl')
    le_mode = joblib.load('output/arrival_mode_encoder.pkl')
    le_complaint = joblib.load('output/complaint_encoder.pkl')
    print("✅ AI Prediction System: ONLINE")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

class AEPatientCheckIn:
    """Self-service check-in system for A&E patients"""

    def __init__(self, model, scaler, encoders):
        self.model = model
        self.scaler = scaler
        self.le_mode = encoders['mode']
        self.le_complaint = encoders['complaint']
        self.queue_position = np.random.randint(5, 25)

    def welcome_screen(self):
        """Display welcome information"""
        print("\n" + "="*80)
        print("📱 EMERGENCY DEPARTMENT SELF-SERVICE CHECK-IN")
        print("="*80)
        print("\n⚠️  IMPORTANT: If you have any of these symptoms:")
        print("   • Severe chest pain")
        print("   • Difficulty breathing")
        print("   • Heavy bleeding")
        print("   • Loss of consciousness")
        print("   → Please alert reception staff IMMEDIATELY")
        print("-"*80)
        input("\nPress Enter to continue with check-in...")

    def collect_patient_info(self):
        """Collect patient information"""
        print("\n" + "="*80)
        print("📋 PATIENT REGISTRATION")
        print("="*80)

        patient = {}

        print("\n👤 Your Information:")
        patient['name'] = input("Full Name: ")
        patient['nhs_number'] = input("NHS Number (or press Enter if unknown): ") or "Temporary"

        # Date of birth
        while True:
            try:
                dob = input("Date of Birth (DD/MM/YYYY): ")
                dob_date = datetime.strptime(dob, '%d/%m/%Y')
                age = (datetime.now() - dob_date).days // 365
                patient['dob'] = dob
                patient['age'] = age
                break
            except:
                print("❌ Please enter date in format DD/MM/YYYY")

        patient['postcode'] = input("Postcode: ")
        patient['phone'] = input("Contact Number: ")

        print("\n👥 Emergency Contact:")
        patient['emergency_name'] = input("Contact Name: ")
        patient['emergency_phone'] = input("Contact Phone: ")

        return patient

    def assess_urgency(self):
        """Self-assessment triage questions"""
        print("\n" + "="*80)
        print("🚨 URGENCY ASSESSMENT")
        print("="*80)
        print("Please answer these questions about your condition:")

        triage_score = 5

        # Pain assessment
        print("\n📊 Pain Level:")
        print("Rate your pain from 0-10 (0 = No pain, 10 = Worst pain)")
        while True:
            try:
                pain = int(input("Pain level (0-10): "))
                if 0 <= pain <= 10:
                    break
                print("Please enter a number between 0 and 10")
            except:
                print("Please enter a valid number")

        if pain >= 8:
            triage_score = min(triage_score, 3)
        elif pain >= 5:
            triage_score = min(triage_score, 4)

        # Symptom duration
        print("\n⏱️ How long have you had these symptoms?")
        print("1. Less than 1 hour")
        print("2. 1-6 hours")
        print("3. 6-24 hours")
        print("4. More than 24 hours")
        
        while True:
            try:
                duration = int(input("Select (1-4): "))
                if 1 <= duration <= 4:
                    break
                print("Please select 1, 2, 3, or 4")
            except:
                print("Please enter a valid number")

        if duration == 1 and pain >= 6:
            triage_score = min(triage_score, 3)

        # Specific symptoms
        print("\n🩺 Do you have any of these symptoms? (Y/N)")
        
        symptoms = {
            "Chest tightness or pressure": 2,
            "Severe headache with vision changes": 2,
            "Persistent vomiting": 3,
            "High fever (over 39°C)": 3,
            "Severe abdominal pain": 3,
            "Deep cut or wound": 4
        }

        for symptom, priority in symptoms.items():
            response = input(f"  {symptom}? (Y/N): ").upper()
            if response == 'Y':
                triage_score = min(triage_score, priority)

        # Mobility
        print("\n🚶 Can you walk without assistance?")
        mobile = input("(Y/N): ").upper()
        if mobile == 'N':
            triage_score = min(triage_score, 3)

        return triage_score, pain

    def collect_medical_info(self):
        """Collect medical history and current complaint"""
        print("\n" + "="*80)
        print("🏥 MEDICAL INFORMATION")
        print("="*80)

        medical = {}

        # Chief complaint
        print("\n📝 Main reason for visit:")
        print("1. Chest Pain")
        print("2. Breathing Difficulty")
        print("3. Abdominal Pain")
        print("4. Injury/Trauma")
        print("5. Fever")
        print("6. Mental Health")
        print("7. Other")

        while True:
            try:
                complaint_choice = int(input("Select (1-7): "))
                if 1 <= complaint_choice <= 7:
                    break
                print("Please select a number between 1 and 7")
            except:
                print("Please enter a valid number")

        complaints = ['Chest Pain', 'Breathing Difficulty', 'Abdominal Pain',
                     'Injury/Trauma', 'Fever', 'Mental Health', 'Other']
        medical['complaint'] = complaints[complaint_choice - 1]

        # Arrival mode
        print("\n🚑 How did you arrive?")
        print("1. Walked in myself")
        print("2. Brought by family/friend")
        print("3. Ambulance")
        print("4. GP referred me")

        while True:
            try:
                arrival_choice = int(input("Select (1-4): "))
                if 1 <= arrival_choice <= 4:
                    break
                print("Please select a number between 1 and 4")
            except:
                print("Please enter a valid number")

        arrival_modes = ['Walk-in', 'Walk-in', 'Ambulance', 'GP Referral']
        medical['arrival_mode'] = arrival_modes[arrival_choice - 1]

        # Medical history
        print("\n💊 Medical History:")
        medical['conditions'] = input("Any existing conditions (or None): ")
        medical['medications'] = input("Current medications (or None): ")
        medical['allergies'] = input("Allergies (or None): ")

        # Previous visits
        print("\n📅 Have you visited A&E in the last 30 days?")
        prev = input("(Y/N): ").upper()
        medical['previous_visits'] = 1 if prev == 'Y' else 0

        return medical

    def calculate_wait_time(self, patient_info, triage, medical_info):
        """Calculate predicted wait time using ML model"""

        now = datetime.now()
        hour = now.hour
        day = now.weekday()
        month = now.month

        # Simulate current department status
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
            arrival_encoded = self.le_mode.transform([medical_info['arrival_mode']])[0]
        except:
            arrival_encoded = 0

        try:
            complaint_encoded = self.le_complaint.transform([medical_info['complaint']])[0]
        except:
            complaint_encoded = 0

        # Create feature vector
        features = pd.DataFrame({
            'hour': [hour],
            'day_of_week': [day],
            'month': [month],
            'is_weekend': [is_weekend],
            'is_peak_hour': [is_peak],
            'is_night': [is_night],
            'triage_category': [triage],
            'age': [patient_info['age']],
            'arrival_mode_encoded': [arrival_encoded],
            'complaint_encoded': [complaint_encoded],
            'department_occupancy': [occupancy],
            'staff_available': [staff],
            'previous_visits_30d': [medical_info['previous_visits']]
        })

        # Predict wait time using trained ML model
        features_scaled = self.scaler.transform(features)
        predicted_wait = self.model.predict(features_scaled)[0]

        # Add realistic variation
        predicted_wait = max(5, predicted_wait + np.random.normal(0, 10))

        return predicted_wait, occupancy, staff

    def display_wait_info(self, patient_info, triage, predicted_wait, occupancy, queue_pos):
        """Display wait time and information to patient"""

        print("\n" + "="*80)
        print("✅ CHECK-IN COMPLETE")
        print("="*80)

        # Generate unique reference
        ref_num = f"AE{datetime.now().strftime('%Y%m%d')}{np.random.randint(1000, 9999)}"

        print(f"\n📋 Your Reference Number: {ref_num}")
        print(f"Name: {patient_info['name']}")
        print(f"Check-in Time: {datetime.now().strftime('%H:%M')}")

        # Triage category display
        triage_labels = {
            1: "🔴 IMMEDIATE - You will be seen immediately",
            2: "🟠 VERY URGENT - Priority patient",
            3: "🟡 URGENT - Will be seen soon",
            4: "🟢 STANDARD - Routine case",
            5: "🔵 NON-URGENT - Minor condition"
        }

        print(f"\n🚨 Urgency Level: {triage_labels[triage]}")

        # Wait time information
        print("\n" + "="*80)
        print("⏱️  ESTIMATED WAIT TIME (AI-PREDICTED)")
        print("="*80)

        hours = int(predicted_wait // 60)
        minutes = int(predicted_wait % 60)

        if hours > 0:
            print(f"\n🕐 Estimated wait: {hours} hour{'s' if hours > 1 else ''} {minutes} minutes")
        else:
            print(f"\n🕐 Estimated wait: {minutes} minutes")

        print(f"\n📊 Current Status:")
        print(f"  • Queue position: Approximately {queue_pos} patients ahead")
        print(f"  • Department occupancy: {occupancy:.0f}%")
        print(f"  • Predicted range: {max(5, predicted_wait-20):.0f} - {predicted_wait+20:.0f} minutes")

        # Status indicator
        if predicted_wait > 240:
            print("\n⚠️  STATUS: Long wait expected - Department is very busy")
        elif predicted_wait > 120:
            print("\n🟡 STATUS: Moderate wait - Department is busy")
        else:
            print("\n🟢 STATUS: Normal wait times")

        # While you wait information
        print("\n" + "="*80)
        print("ℹ️  WHILE YOU WAIT")
        print("="*80)
        print("\n📱 Your reference will be called on the display screens")
        print("🏪 Refreshments available in the café (Ground Floor)")
        print("🚻 Toilets located near waiting area")
        print("💊 Pharmacy open until 20:00")
        print("📞 Free WiFi: NHS_Guest")

        print("\n⚠️  Please inform staff immediately if:")
        print("  • Your condition worsens")
        print("  • You develop new symptoms")
        print("  • You need urgent assistance")

        return ref_num

    def generate_ticket(self, ref_num, patient_info, predicted_wait):
        """Generate a ticket/receipt for the patient"""

        print("\n" + "="*80)
        print("🎫 YOUR A&E TICKET")
        print("="*80)
        print(f"Reference: {ref_num}")
        print(f"Name: {patient_info['name']}")
        print(f"Time: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
        print(f"Estimated wait: {predicted_wait:.0f} minutes")
        print("-"*80)
        print("Keep this ticket with you")
        print("Your reference will appear on screens")
        print("="*80)

# Initialize the check-in system
check_in_system = AEPatientCheckIn(model, scaler, {'mode': le_mode, 'complaint': le_complaint})

def run_patient_checkin():
    """Complete patient self-service check-in flow"""

    # Welcome
    check_in_system.welcome_screen()

    # Collect information
    patient_info = check_in_system.collect_patient_info()
    triage_score, pain_level = check_in_system.assess_urgency()
    medical_info = check_in_system.collect_medical_info()

    # Calculate wait time using AI
    print("\n⏳ Calculating your wait time using AI prediction model...")
    time.sleep(2)

    predicted_wait, occupancy, staff = check_in_system.calculate_wait_time(
        patient_info, triage_score, medical_info
    )

    # Display results
    ref_num = check_in_system.display_wait_info(
        patient_info, triage_score, predicted_wait, occupancy,
        check_in_system.queue_position
    )

    # Generate ticket
    check_in_system.generate_ticket(ref_num, patient_info, predicted_wait)

    print("\n📱 Text 'UPDATE' to 66777 for wait time updates")
    print("💬 Or scan QR code on your ticket for live updates")
    print("\n✅ Check-in complete. Please take a seat in the waiting area.")

    return ref_num, predicted_wait

# Main execution
print("\n🚀 SYSTEM READY FOR PATIENT CHECK-IN")
print("\nWould you like to check in a patient? (Yes/No)")
response = input(">>> ").lower()

if response in ['yes', 'y']:
    run_patient_checkin()
else:
    print("\n💤 System on standby. Run this script again when ready.")
    print("="*80)