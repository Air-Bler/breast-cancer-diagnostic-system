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


feature_info = {
    'radius': 'Μέση απόσταση από το κέντρο προς τα σημεία της περιμέτρου του κυτταρικού πυρήνα (Μέγεθος).',
    'texture': 'Τυπική απόκλιση τιμών κλίμακας του γκρι στην εικόνα (Ανομοιογένεια υφής).',
    'perimeter': 'Μέγεθος της περιμέτρου του πυρήνα του κυττάρου.',
    'area': 'Συνολικό εμβαδόν της επιφάνειας του πυρήνα.',
    'smoothness': 'Τοπική διακύμανση στα μήκη των ακτίνων (Ομαλότητα ορίων).',
    'compactness': 'Συμπαγικότητα: (περίμετρος² / εμβαδόν - 1.0).',
    'concavity': 'Βαθμός σοβαρότητας των κοίλων τμημάτων του περιγράμματος.',
    'concave points': 'Πλήθος των κοίλων σημείων στην επιφάνεια του περιγράμματος.',
    'symmetry': 'Συμμετρία του σχήματος του κυτταρικού πυρήνα.',
    'fractal_dimension': 'Κλασματική διάσταση (Προσέγγιση ακτογραμμής - 1.0).'
}

base_names = list(feature_info.keys())
mean_features = [f"{k}_mean" if k != 'concave points' else 'concave points_mean' for k in base_names]
se_features = [f"{k}_se" if k != 'concave points' else 'concave points_se' for k in base_names]
worst_features = [f"{k}_worst" if k != 'concave points' else 'concave points_worst' for k in base_names]

all_features = mean_features + se_features + worst_features

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


for feat in all_features:
    if feat not in st.session_state:
        st.session_state[feat] = 0.0

def set_benign():
    for feat, val in zip(all_features, benign_sample):
        st.session_state[feat] = float(val)

def set_malignant():
    for feat, val in zip(all_features, malignant_sample):
        st.session_state[feat] = float(val)

def set_zeros():
    for feat in all_features:
        st.session_state[feat] = 0.0

#Sidebar 
col_left, col_mid, col_right = st.sidebar.columns([1, 2, 1])
with col_mid:
    st.image("../images.png", width=140)

st.sidebar.markdown("### Εισαγωγή Δεδομένων")
st.sidebar.info("Επιλέξτε ένα δείγμα για αυτόματη συμπλήρωση ή εισάγετε τιμές στις καρτέλες:")

st.sidebar.button("🟢 Δείγμα Καλοήθους (Benign)", on_click=set_benign, use_container_width=True)
st.sidebar.button("🔴 Δείγμα Κακοήθους (Malignant)", on_click=set_malignant, use_container_width=True)
st.sidebar.button("🔄 Επαναφορά σε 0.0", on_click=set_zeros, use_container_width=True)


st.title("Σύστημα Υποστήριξης Διαγνωστικών Αποφάσεων")
st.caption("Ανάλυση μορφολογικών χαρακτηριστικών κυττάρων μέσω Τεχνητού Νευρωνικού Δικτύου (MLP).")


def render_feature_group(features, suffix_label):
    col1, col2 = st.columns(2)
    for i, name in enumerate(features):
        base_metric = name.replace('_mean', '').replace('_se', '').replace('_worst', '')
        tooltip = f"{feature_info.get(base_metric, '')} ({suffix_label})"
        
        target_col = col1 if i < 5 else col2
        with target_col:
            st.number_input(
                f"{name}",
                format="%.4f",
                key=name,
                help=tooltip
            )

tab_mean, tab_se, tab_worst = st.tabs([
    " Μέσες Τιμές (Mean)", 
    " Τυπικά Σφάλματα (Standard Error)", 
    " Χειρότερες Τιμές (Worst / Largest)"
])

with tab_mean:
    render_feature_group(mean_features, "Μέση Τιμή")
with tab_se:
    render_feature_group(se_features, "Τυπικό Σφάλμα")
with tab_worst:
    render_feature_group(worst_features, "Μέση τιμή των 3 μεγαλύτερων τιμών")

st.divider()


if st.button("Εκτέλεση Διάγνωσης", type="primary", use_container_width=True):
    user_values = [st.session_state[name] for name in all_features]
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
            st.error("### Πιθανή Κακοήθεια (Malignant)")
            st.write("Τα μορφολογικά χαρακτηριστικά παραπέμπουν σε κακοήθη αλλοίωση.")
        else:
            st.success("### Πιθανή Καλοήθεια (Benign)")
            st.write("Τα μορφολογικά χαρακτηριστικά παραπέμπουν σε καλοήθη αλλοίωση.")

    with res_col2:
        st.metric(label="Βεβαιότητα Μοντέλου", value=f"{conf:.2f}%")
        st.progress(conf / 100.0)


    st.markdown("#### Κύριοι Παράγοντες Διαγνωστικής Εκτίμησης")
    r_val = st.session_state['radius_mean']
    cp_val = st.session_state['concave points_mean']
    t_val = st.session_state['texture_mean']

    if is_malignant:
        st.write("Η ταξινόμηση ως **πιθανή κακοήθεια** βασίστηκε κυρίως στις παρακάτω αποκλίσεις:")
        if r_val > 14.0:
            st.markdown(f"* **Αυξημένο Μέγεθος Πυρήνα (`radius_mean` = {r_val:.2f}):** Υπέρβαση μέσου όρου καλοήθων δειγμάτων (~12.15).")
        if cp_val > 0.05:
            st.markdown(f"* **Ανωμαλία Περιγράμματος (`concave points_mean` = {cp_val:.4f}):** Υψηλός αριθμός κοίλων σημείων μεμβράνης.")
        if t_val > 20.0:
            st.markdown(f"* **Υψηλή Ανομοιογένεια Υφής (`texture_mean` = {t_val:.2f}):** Έντονη διακύμανση πυκνότητας χρωματίνης.")
        if r_val <= 14.0 and cp_val <= 0.05 and t_val <= 20.0:
            st.markdown("* **Συνδυαστική Πολυπαραμετρική Απόκλιση:** Σύγκλιση των δευτερευόντων παραμέτρων (worst/se) σε κακοήθεια.")
    else:
        st.write("Η ταξινόμηση ως **πιθανή καλοήθεια** βασίστηκε στα εξής φυσιολογικά ευρήματα:")
        st.markdown(f"* **Φυσιολογικό Μέγεθος Πυρήνα (`radius_mean` = {r_val:.2f}):** Εντός φυσιολογικών ορίων.")
        st.markdown(f"* **Ομαλό Περίγραμμα (`concave points_mean` = {cp_val:.4f}):** Ελάχιστες κυτταρικές κοιλότητες.")
        st.markdown(f"* **Ομοιόμορφη Υφή (`texture_mean` = {t_val:.2f}):** Ομαλή κατανομή φωτεινότητας.")

    st.warning("**Προσοχή:** Το αποτέλεσμα αποτελεί προϊόν υπολογιστικού μοντέλου τεχνητής νοημοσύνης και δεν αντικαθιστά την ιατρική γνωμάτευση.")

#  Τεχνικές Λεπτομέρειες
with st.expander("Τεχνικές Λεπτομέρειες Μοντέλου"):
    st.write("**Αλγόριθμος:** Multi-Layer Perceptron (Neural Network)")
    st.write("**Επίδοση (AUC):** 0.99")
    st.write("**Προεπεξεργασία:** StandardScaler")