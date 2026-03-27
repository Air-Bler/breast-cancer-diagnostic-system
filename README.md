Project Overview | Επισκόπηση Έργου
This repository contains a comprehensive computational system for breast cancer tumor classification using advanced Machine Learning techniques. Developed as a Bachelor's Thesis, the project focuses on the diagnostic analysis of cytological features from breast mass biopsies.

Αυτό το project περιλαμβάνει ένα ολοκληρωμένο υπολογιστικό σύστημα για την ταξινόμηση όγκων καρκίνου του μαστού. Αναπτύχθηκε στο πλαίσιο πτυχιακής εργασίας και επικεντρώνεται στη διαγνωστική ανάλυση κυτταρολογικών χαρακτηριστικών από βιοψίες μαστού.


Technical Specifications | Τεχνικές Προδιαγραφές
-Dataset: Breast Cancer Wisconsin (Diagnostic) - UCI Machine Learning Repository.
-Key Libraries: Scikit-Learn, Pandas, NumPy, Streamlit, Joblib.
-Algorithms Evaluated: Logistic Regression, Decision Tree, Random Forest, SVM, k-NN, Neural Network (MLP).
-Primary Model: Multi-Layer Perceptron (Neural Network) with 99.67% AUC.



Functionality | Λειτουργικότητα
-The system processes 30 distinct morphological features. Upon entry via the web interface:
  1.Feature Scaling: Inputs are normalized in real-time.
  2.Classification: The Neural Network engine classifies the tumor as Benign or Malignant.
  3.Confidence Score: Provides a statistical probability for the diagnostic outcome.

-Το σύστημα επεξεργάζεται 30 μορφολογικά χαρακτηριστικά. Μέσω της web διεπαφής:
  1.Κανονικοποίηση: Τα δεδομένα εισόδου εξομαλύνονται σε πραγματικό χρόνο.
  2.Ταξινόμηση: Το Νευρωνικό Δίκτυο κατηγοριοποιεί τον όγκο ως Καλοήθη ή Κακοήθη.
  3.Βεβαιότητα: Παρέχεται το ποσοστό στατιστικής πιθανότητας για το αποτέλεσμα.
