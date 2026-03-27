import os
import streamlit as st 
import streamlit as st 
import numpy as np



# ΣΥΣΤΗΜΑ ΥΠΟΛΟΓΙΣΤΙΚΗΣ ΤΑΞΙΝΟΜΗΣΗΣ ΒΙΟΪΑΤΡΙΚΩΝ ΔΕΔΟΜΕΝΩΝ
st.set_page_config(page_title="Breast Cancer Diagnostic Assistant", layout="wide")

@st.cache_resource
def load_assets():
    model = joblib.load('neural_network_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    mlp_model, data_scaler = load_assets()
except:
    st.error("Σφάλμα: Δεν βρέθηκαν τα αρχεία 'neural_network_model.pkl' ή 'scaler.pkl'.")


    st.title("🩺 Σύστημα Υποστήριξης Διαγνωστικών Αποφάσεων")
st.markdown("""
Αυτή η εφαρμογή χρησιμοποιεί ένα **Τεχνητό Νευρωνικό Δίκτυο (MLP)** για την ανάλυση μορφολογικών χαρακτηριστικών κυττάρων 
και την πρόβλεψη πιθανής κακοήθειας.
""")

st.sidebar.header("Εισαγωγή Δεδομένων Βιοψίας")
st.sidebar.info("Εισάγετε τις τιμές 'Worst'  για ακριβέστερη πρόβλεψη.")

feature_names = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

user_values = []
col1, col2 = st.columns(2)

for i, name in enumerate(feature_names):

    with col1 if i < 15 else col2:
        val = st.number_input(f"{name}", value=0.0, format="%.4f")
        user_values.append(val)

        if st.button("Εκτέλεση Διάγνωσης"):
    
    input_array = np.array(user_values).reshape(1, -1)
    input_scaled = data_scaler.transform(input_array)

    prediction = mlp_model.predict(input_scaled)
    probability = mlp_model.predict_proba(input_scaled)

    st.divider()

    if prediction[0] == 'M' or prediction[0] == 1:
        st.error(f"### Αποτέλεσμα: **Πιθανή Κακοήθεια (Malignant)**")
        conf = probability[0][1] * 100 
    else:
        st.success(f"### Αποτέλεσμα: **Πιθανή Καλοήθεια (Benign)**")
        conf = probability[0][0] * 100

        st.metric(label="Βεβαιότητα Μοντέλου", value=f"{conf:.2f}%")
    
    st.warning("**Προσοχή:** Το αποτέλεσμα αποτελεί προϊόν τεχνητής νοημοσύνης και δεν αντικαθιστά την ιατρική γνωμάτευση.")

#
with st.expander("Τεχνικές Λεπτομέρειες Μοντέλου"):
    st.write("Αλγόριθμος: Multi-Layer Perceptron (Neural Network)")
    st.write("Επίδοση (AUC): 0.9967")
    st.write("Προεπεξεργασία: StandardScaler")