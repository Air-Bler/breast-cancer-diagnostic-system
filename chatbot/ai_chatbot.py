import os
import streamlit as st 
import numpy as np
import joblib

st.set_page_config(page_title="Breast Cancer Diagnostic Assistant", layout="wide")


@st.cache_resource
def load_assets():
    model = joblib.load('neural_network_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    mlp_model, data_scaler = load_assets()
except Exception as e:
    st.error(f"Σφάλμα: Δεν βρέθηκαν τα αρχεία μοντέλου. {e}")
    st.stop()

feature_names = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

benign_sample = [
    13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.1885, 0.05766,
    0.2699, 0.7886, 2.058, 23.56, 0.008462, 0.0146, 0.02387, 0.01315, 0.0198, 0.0023,
    15.11, 19.26, 99.7, 711.2, 0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259
]

malignant_sample = [
    16.02, 23.24, 102.7, 797.8, 0.08206, 0.06669, 0.03299, 0.03323, 0.1528, 0.05697,
    0.3795, 1.187, 2.466, 40.51, 0.004029, 0.009269, 0.01101, 0.007591, 0.0146, 0.003042,
    19.19, 33.88, 123.8, 1150.0, 0.1181, 0.1551, 0.1459, 0.09975, 0.2948, 0.08452
]


for feat in feature_names:
    if feat not in st.session_state:
        st.session_state[feat] = 0.0


def set_benign():
    for feat, val in zip(feature_names, benign_sample):
        st.session_state[feat] = float(val)

def set_malignant():
    for feat, val in zip(feature_names, malignant_sample):
        st.session_state[feat] = float(val)

def set_zeros():
    for feat in feature_names:
        st.session_state[feat] = 0.0

#  Sidebar
col_left, col_mid, col_right = st.sidebar.columns([1, 2, 1])
with col_mid:
    st.image("../images.png", width=150)
st.sidebar.header("Εισαγωγή Δεδομένων Βιοψίας")
st.sidebar.info("Εισάγετε χειροκίνητα τιμές ή επιλέξτε ένα έτοιμο δείγμα:")

st.sidebar.button("🟢 Δείγμα Καλοήθους (Benign)", on_click=set_benign, use_container_width=True)
st.sidebar.button("🔴 Δείγμα Κακοήθους (Malignant)", on_click=set_malignant, use_container_width=True)
st.sidebar.button("🔄 Επαναφορά", on_click=set_zeros, use_container_width=True)


st.title("Σύστημα Υποστήριξης Διαγνωστικών Αποφάσεων")
st.markdown("""
Αυτή η εφαρμογή χρησιμοποιεί ένα Τεχνητό Νευρωνικό Δίκτυο (MLP) για την ανάλυση μορφολογικών χαρακτηριστικών κυττάρων 
και την πρόβλεψη πιθανής κακοήθειας.
""")


col1, col2 = st.columns(2)

for i, name in enumerate(feature_names):
    with col1 if i < 15 else col2:
        st.number_input(
            f"{name}",
            format="%.4f",
            key=name  
        )

st.divider()


# 6. Εκτέλεση Διάγνωσης
if st.button("Εκτέλεση Διάγνωσης", type="primary", use_container_width=True):
    user_values = [st.session_state[name] for name in feature_names]
    
    input_array = np.array(user_values).reshape(1, -1)
    input_scaled = data_scaler.transform(input_array)

    prediction = mlp_model.predict(input_scaled)
    probability = mlp_model.predict_proba(input_scaled)

    is_malignant = (prediction[0] == 1 or str(prediction[0]).upper() == 'M')
    conf = probability[0][1] * 100 if is_malignant else probability[0][0] * 100

    st.markdown("### Αποτέλεσμα Ανάλυσης")
    res_col1, res_col2 = st.columns([2, 1])

    with res_col1:
        if is_malignant:
            st.error("###  Πιθανή Κακοήθεια (Malignant)")
            st.write("Τα μορφολογικά χαρακτηριστικά παραπέμπουν σε κακοήθη αλλοίωση.")
        else:
            st.success("###  Πιθανή Καλοήθεια (Benign)")
            st.write("Τα μορφολογικά χαρακτηριστικά παραπέμπουν σε καλοήθη αλλοίωση.")

    with res_col2:
        st.metric(label="Βεβαιότητα Μοντέλου", value=f"{conf:.2f}%")
        st.progress(conf / 100.0)

    # Επεξήγηση Αποτελέσματος (Explainability)
    st.markdown("####  Κύριοι Παράγοντες Διαγνωστικής Εκτίμησης")
    
    # Τιμές αναφοράς (μέσοι όροι από το Wisconsin Diagnostic dataset)
    # Καλοήθη: radius ~12.15, concave points ~0.026, texture ~17.91
    # Κακοήθη: radius ~17.46, concave points ~0.088, texture ~21.60
    r_val = st.session_state['radius_mean']
    cp_val = st.session_state['concave points_mean']
    t_val = st.session_state['texture_mean']

    if is_malignant:
        st.write("Η ταξινόμηση ως **πιθανή κακοήθεια** βασίστηκε κυρίως στις παρακάτω αποκλίσεις:")
        if r_val > 14.0:
            st.markdown(f"* **Αυξημένο Μέγεθος Πυρήνα (`radius_mean` = {r_val:.2f}):** Η τιμή υπερβαίνει τον μέσο όρο καλοήθων δειγμάτων (~12.15), υποδεικνύοντας κυτταρική υπερπλασία.")
        if cp_val > 0.05:
            st.markdown(f"* **Ανωμαλία Περιγράμματος (`concave points_mean` = {cp_val:.4f}):** Υψηλός αριθμός κοίλων σημείων, ένδειξη έντονης ανωμαλίας στη μεμβράνη του κυττάρου.")
        if t_val > 20.0:
            st.markdown(f"* **Υψηλή Ανομοιογένεια Υφής (`texture_mean` = {t_val:.2f}):** Μεγάλη διακύμανση στις τιμές φωτεινότητας, χαρακτηριστικό ανομοιόμορφης χρωματίνης.")
        if r_val <= 14.0 and cp_val <= 0.05 and t_val <= 20.0:
            st.markdown("* **Συνδυαστική Πολυπαραμετρική Απόκλιση:** Παρότι τα κύρια μεγέθη παραμένουν ενδιάμεσα, οι δευτερεύουσες παράμετροι (worst/se) συγκλίνουν προς κακοήθη συμπεριφορά.")
    else:
        st.write("Η ταξινόμηση ως **πιθανή καλοήθεια** βασίστηκε κυρίως στα εξής φυσιολογικά ευρήματα:")
        st.markdown(f"* **Φυσιολογικό Μέγεθος Πυρήνα (`radius_mean` = {r_val:.2f}):** Εντός των αναμενόμενων ορίων για μη κακοήθεις ιστούς.")
        st.markdown(f"* **Ομαλό Περίγραμμα (`concave points_mean` = {cp_val:.4f}):** Χαμηλός αριθμός κοιλοτήτων, στοιχείο που υποδηλώνει διατηρημένη κυτταρική δομή.")
        st.markdown(f"* **Ομοιόμορφη Υφή (`texture_mean` = {t_val:.2f}):** Χαμηλή διακύμανση στην πυκνότητα των κυττάρων.")

    st.warning("Προσοχή: Το αποτέλεσμα αποτελεί προϊόν τεχνητής νοημοσύνης και δεν αντικαθιστά την ιατρική γνωμάτευση.")