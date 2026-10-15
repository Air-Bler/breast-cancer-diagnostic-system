# Σύστημα Υποστήριξης Διάγνωσης Καρκίνου του Μαστού

Πτυχιακή εργασία: ταξινόμηση όγκων του μαστού σε **καλοήθεις (B)** ή **κακοήθεις (M)** με τεχνικές Μηχανικής Μάθησης, βάσει κυτταρολογικών χαρακτηριστικών από βιοψίες. Περιλαμβάνει την πλήρη ανάλυση σε Jupyter Notebook και μια web εφαρμογή (Streamlit) για πρόβλεψη σε πραγματικό χρόνο.

> *Bachelor's thesis: breast tumor classification (benign / malignant) using machine learning on the Breast Cancer Wisconsin (Diagnostic) dataset, with a Streamlit web app for real-time prediction.*

---

## Dataset

**Breast Cancer Wisconsin (Diagnostic)** — UCI Machine Learning Repository (`brca.csv`)

- **569** δείγματα — 357 καλοήθη, 212 κακοήθη
- **30** χαρακτηριστικά: 10 μορφολογικά μεγέθη του κυτταρικού πυρήνα (ακτίνα, υφή, περίμετρος, εμβαδόν, ομαλότητα, συμπαγικότητα, κοιλότητα, κοίλα σημεία, συμμετρία, κλασματική διάσταση), το καθένα σε τρεις μορφές: μέση τιμή (`mean`), τυπικό σφάλμα (`se`) και χειρότερη τιμή (`worst`)

## Μοντέλα

Συγκρίθηκαν οι αλγόριθμοι:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- k-Nearest Neighbors (k-NN)
- **Neural Network (MLP)** — τελικό μοντέλο, **AUC = 99.67%**

Η αξιολόγηση γίνεται με confusion matrices, ROC/AUC και ανάλυση overfitting (σύγκριση απόδοσης train/test).

## Δομή του project

```
breast-cancer-ml/
├── breast_cancer_project.ipynb   # Ανάλυση δεδομένων, εκπαίδευση & αξιολόγηση μοντέλων
├── brca.csv                      # Dataset
├── chatbot/
│   ├── ai_chatbot.py             # Web εφαρμογή Streamlit
│   ├── neural_network_model.pkl  # Εκπαιδευμένο MLP που χρησιμοποιεί η εφαρμογή
│   └── scaler.pkl                # StandardScaler που χρησιμοποιεί η εφαρμογή
├── breast_cancer_rf_model.pkl    # Εκπαιδευμένο Random Forest
├── images.png                    # Λογότυπο εφαρμογής
├── pyproject.toml                # Εξαρτήσεις (uv)
└── uv.lock                       # Κλειδωμένες εκδόσεις πακέτων
```

## Εγκατάσταση

Απαιτείται **Python 3.13** και ο διαχειριστής πακέτων [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Air-Bler/breast-cancer-diagnostic-system.git
cd breast-cancer-diagnostic-system
uv sync
```

Το `uv sync` δημιουργεί αυτόματα το virtual environment (`.venv/`) και εγκαθιστά όλες τις εξαρτήσεις από το `uv.lock`.

## Χρήση

### Web εφαρμογή

```bash
cd chatbot
uv run streamlit run ai_chatbot.py
```

Η εφαρμογή ανοίγει στο `http://localhost:8501`. Ο χρήστης εισάγει τα 30 χαρακτηριστικά (ή φορτώνει έτοιμο παράδειγμα καλοήθους/κακοήθους όγκου) και το σύστημα:

1. **Κανονικοποιεί** τις τιμές με τον αποθηκευμένο scaler
2. **Ταξινομεί** τον όγκο ως καλοήθη ή κακοήθη με το νευρωνικό δίκτυο
3. **Εμφανίζει** την πιθανότητα (βεβαιότητα) του αποτελέσματος

> Η εφαρμογή πρέπει να εκτελείται από τον φάκελο `chatbot/`, γιατί φορτώνει τα `.pkl` και το λογότυπο με σχετικές διαδρομές.

### Notebook

Ανοίξτε το `breast_cancer_project.ipynb` στο VS Code και επιλέξτε ως kernel το `.venv` (Python 3.13).

Το notebook αποθηκεύει το εκπαιδευμένο μοντέλο και τον scaler στον κεντρικό φάκελο. Για να τα χρησιμοποιήσει η εφαρμογή, αντιγράψτε τα στον φάκελο `chatbot/`.

## Τεχνολογίες

Python · scikit-learn · pandas · NumPy · Matplotlib · Seaborn · Streamlit · Joblib

## Αποποίηση ευθύνης

Το σύστημα αναπτύχθηκε για ακαδημαϊκούς σκοπούς και **δεν** αποτελεί ιατρικό εργαλείο. Δεν αντικαθιστά τη διάγνωση από εξειδικευμένο ιατρό.
