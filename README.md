Project Overview
This repository contains the implementation of a computational system for the classification of breast cancer tumors using Machine Learning techniques. 
The project was developed as part of a Bachelor's Thesis, focusing on the analysis of cytological features extracted from breast mass biopsies.

Dataset: Breast Cancer Wisconsin (Diagnostic) Dataset from UCI Machine Learning Repository
Key Libraries: Scikit-Learn, Pandas, NumPy, Joblib
Primary Algorithm: Random Forest Classifier

Project Structure:

breast_cancer_project.ipynb: The primary research notebook containing data exploration, preprocessing, and model training.

chatbot/ai_chatbot.py: A command-line interface (CLI) that allows users to input clinical parameters and receive real-time diagnostic predictions.

*.pkl: Serialized files containing the trained model and the feature scaler (StandardScaler).

Functionality:
The system processes 30 distinct  features from digital images of fine needle aspirates. 
Upon data entry, it performs feature scaling and executes the classification algorithm to determine whether a tumor is Benign or Malignant, providing a statistical confidence score for each prediction.



Επισκόπηση Έργου
Αυτό το έργο περιλαμβάνει την εφαρμογή ενός υπολογιστικού συστήματος για την ταξινόμηση των όγκων καρκίνου του μαστού χρησιμοποιώντας τεχνικές Μηχανικής Μάθησης.
Το έργο αναπτύχθηκε ως μέρος μιας πτυχιακής εργασίας, με επίκεντρο την ανάλυση κυτταρολογικών χαρακτηριστικών που εξάγονται από βιοψίες μαστικών μαστών.

Σύνολο Δεδομένων: Breast Cancer Wisconsin (Diagnostic) Dataset από το UCI Machine Learning Repository
Βασικές Βιβλιοθήκες: Scikit-Learn, Pandas, NumPy, Joblib
Κύριος Αλγόριθμος: Random Forest Classifier


Δομή του Έργου:

breast_cancer_project.ipynb: Το κύριο αρχείο έρευνας που περιλαμβάνει την εξερεύνηση δεδομένων, την προεπεξεργασία και την εκπαίδευση του μοντέλου.

chatbot/ai_chatbot.py: Μια διεπαφή γραμμής εντολών (CLI) που επιτρέπει στον χρήστη την εισαγωγή κλινικών παραμέτρων και τη λήψη διαγνωστικών προβλέψεων σε πραγματικό χρόνο.

*.pkl: Αρχεία σειριοποίησης που περιέχουν το εκπαιδευμένο μοντέλο και τις παραμέτρους κανονικοποίησης (StandardScaler).

Λειτουργικότητα:
Το σύστημα επεξεργάζεται 30 διαφορετικά  χαρακτηριστικά από ψηφιοποιημένες εικόνες δειγμάτων βιοψίας.
Με την εισαγωγή των δεδομένων, πραγματοποιείται κανονικοποίηση των χαρακτηριστικών και εκτελείται ο αλγόριθμος ταξινόμησης για τον προσδιορισμό του όγκου ως Καλοήθη ή Κακοήθη, παρέχοντας παράλληλα έναν συντελεστή στατιστικής βεβαιότητας για κάθε πρόβλεψη.
