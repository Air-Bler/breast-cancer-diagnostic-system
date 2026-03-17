import os
import joblib
import numpy as np
import time


# ΣΥΣΤΗΜΑ ΥΠΟΛΟΓΙΣΤΙΚΗΣ ΤΑΞΙΝΟΜΗΣΗΣ ΒΙΟΪΑΤΡΙΚΩΝ ΔΕΔΟΜΕΝΩΝ


def load_system_components():
    try:
        model = joblib.load("../breast_cancer_rf_model.pkl")
        scaler = joblib.load("../scaler.pkl")
        print("[INFO] Οι παράμετροι του μοντέλου φορτώθηκαν επιτυχώς.")
        return model, scaler
    except Exception as e:
        print(f"[ERROR] Αποτυχία φόρτωσης συστήματος: {e}")
        exit()

def validate_input_ranges(val_list):

    # 1. Βασικός έλεγχος: Καμία τιμή δεν μπορεί να είναι αρνητική
    if any(n < 0 for n in val_list):
        print("[VALIDATION ERROR] Εντοπίστηκαν αρνητικές τιμές. Οι μετρήσεις πρέπει να είναι θετικές.")
        return False
    
    # 2. Έλεγχος ορίων για κρίσιμα χαρακτηριστικά 
    # radius_mean (index 0): max ~30 | area_mean (index 3): max ~2500
    if val_list[0] > 40 or val_list[3] > 4000:
        print("[WARNING] Προσοχή: Εντοπίστηκαν ακραίες τιμές (Outliers).")
        print("Ελέγξτε αν οι μετρήσεις είναι σε σωστή μονάδα κλίμακας.")
        
    
    return True

def generate_analytical_report(prediction_label, data_vector, prob):
    r_mean = data_vector[0]
    c_mean = data_vector[6]
    
    header = "--- ΤΕΧΝΙΚΗ ΑΝΑΦΟΡΑ ΤΑΞΙΝΟΜΗΤΗ ---"
    analysis = (
        f"Αποτέλεσμα: {prediction_label} (Βεβαιότητα: {prob:.2f}%)\n"
        f"Στατιστική Τεκμηρίωση: Η ανάλυση βασίστηκε σε 30 παραμέτρους.\n"
        f"Κρίσιμοι Δείκτες: Radius Mean: {r_mean}, Concavity Mean: {c_mean}."
    )
    disclaimer = "\n\n[ΠΡΩΤΟΚΟΛΛΟ]: Το παρόν αποτελεί προϊόν πρόβλεψης τεχνιτής νοημοσύνης και απαιτεί κλινική αξιολόγηση."
    return f"{header}\n{analysis}{disclaimer}"

def main():
    clf, transformer = load_system_components()
    
    print("\n====================================================")
    print("  INTERFACE ΠΡΟΓΝΩΣΤΙΚΗΣ ΜΟΝΤΕΛΟΠΟΙΗΣΗΣ (v1.0)")
    print("====================================================")
    print("Εισάγετε το διάνυσμα των 30 χαρακτηριστικών (CSV format):")

    while True:
        raw_data = input("\nΔεδομένα Εισόδου: ")
        
        if raw_data.lower() == 'exit':
            break
            
        try:
            val_list = [float(x.strip()) for x in raw_data.split(',')]
            
            if len(val_list) != 30:
                print(f"[WARNING] Απαιτούνται 30 παράμετροι (Ελήφθησαν: {len(val_list)})")
                continue

            # --- Προσθήκη Ελέγχου Ορίων ---
            if not validate_input_ranges(val_list):
                continue

            # Μετασχηματισμός και Πρόβλεψη
            input_vector = np.array([val_list])
            scaled_vector = transformer.transform(input_vector)
            
            print("[SYSTEM] Εκτέλεση αλγορίθμου ταξινόμησης...")
            time.sleep(0.7)
            
            prediction = clf.predict(scaled_vector)
            
            prob = clf.predict_proba(scaled_vector)[0].max() * 100
            
            label = "Malignant (Κακοήθης)" if prediction[0] == 1 else "Benign (Καλοήθης)"
            
            print(f"\nΑποτέλεσμα: {label}")
            print("-" * 40)
            print(generate_analytical_report(label, val_list, prob))
            
        except ValueError:
            print("[ERROR] Μη έγκυρος τύπος δεδομένων.")
        except Exception as e:
            print(f"[ERROR] Σφάλμα: {e}")

if __name__ == "__main__":
    main()