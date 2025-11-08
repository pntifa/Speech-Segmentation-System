import os
import csv
import librosa
import numpy as np

# Εισαγωγή εργαλείων μηχανικής μάθησης από το scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import shuffle, resample

# Ορισμός βασικών παραμέτρων για την εξαγωγή χαρακτηριστικών ήχου
SAMPLE_RATE = 16000                # Ρυθμός δειγματοληψίας (Hz)
N_MFCC = 13                        # Αριθμός MFCC χαρακτηριστικών
FRAME_DURATION = 0.025            # Διάρκεια καρέ σε δευτερόλεπτα
HOP_DURATION = 0.010              # Απόσταση μετατόπισης καρέ (hop) σε δευτερόλεπτα
N_FFT = int(SAMPLE_RATE * FRAME_DURATION)     # Μέγεθος FFT παραθύρου
HOP_LENGTH = int(SAMPLE_RATE * HOP_DURATION)  # Απόσταση μετατόπισης σε δείγματα
FRAME_DURATION_SEC = HOP_LENGTH / SAMPLE_RATE # Διάρκεια καρέ σε δευτερόλεπτα (πραγματικός χρόνος)

# Ορισμός διαδρομών για φάκελο εκπαίδευσης και δοκιμών
train_speech_dir = os.path.join("train","speech", "librivox")
train_noise_dir = os.path.join("train","noise", "free-sound")
test_dir = os.path.join("test")
output_csv_path = "results.csv"

# Συνάρτηση εξαγωγής χαρακτηριστικών MFCC από αρχείο ήχου
def extract_feautures(filepath, label):
    y, sr = librosa.load(filepath, sr=SAMPLE_RATE)  # Φόρτωση ήχου με συγκεκριμένο ρυθμό δειγματοληψίας
    y = y / np.max(np.abs(y))  # Κανονικοποίηση τιμών σήματος

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH).T
    features = mfcc
    labels = np.full((features.shape[0],), label)  # Δημιουργία ετικετών για κάθε καρέ

    return features, labels

# Επεξεργασία αρχείων φωνής ή θορύβου για την εκπαίδευση
def process_dir_train(directory, label):
    all_X, all_y = [], []
    for fname in os.listdir(directory):
        if fname.endswith(".wav"):
            filepath = os.path.join(directory, fname)
            try: 
                X, y = extract_feautures(filepath, label)
                all_X.append(X)
                all_y.append(y)
            except Exception as e:
                print(f"Σφάλμα στην επεξεργασία του αρχείου {fname}: {e}")
    return np.vstack(all_X), np.hstack(all_y)

# Επεξεργασία αρχείων δοκιμής
def process_dir_test(directory):
    X_test_list = []
    filenames = []

    for fname in sorted(os.listdir(directory)):
        if fname.endswith(".wav"):
            filepath = os.path.join(directory, fname)
            try: 
                X, _ = extract_feautures(filepath, label=1)  # Η ετικέτα 1 δεν επηρεάζει τη δοκιμή
                X_test_list.append(X)
                filenames.append(fname)
            except Exception as e:
                print(f"Σφάλμα στην επεξεργασία του αρχείου {fname}: {e}")
    return X_test_list, filenames

# Αξιολόγηση του μοντέλου με μετρικές απόδοσης
def eval_model(y_true, y_pred, model_name=""):
    print(f" ")
    print(f" Αξιολόγηση μοντέλου {model_name}:")
    print(f" Accurancy: {accuracy_score(y_true, y_pred):.4f}")     # Ακρίβεια
    print(f" Precision: {precision_score(y_true, y_pred):.4f}")   # Ευστοχία
    print(f" Recall: {recall_score(y_true, y_pred):.4f}")         # Ανάκληση
    print(f" F1 Δείκτης: {f1_score(y_true, y_pred):.4f}")         # F1 score

# Μετατροπή προβλέψεων καρέ σε διαστήματα με ετικέτες foreground/background
def get_labeled_segments(predictions, frame_duration=FRAME_DURATION_SEC):
    segments = []
    if len(predictions) == 0:
        return segments
    
    current_label = predictions[0]
    start_frame = 0
    for i in range(1, len(predictions)):
        if predictions[i] != current_label:
            end_frame = i
            label = "foreground" if current_label == 1 else "background"
            segments.append((round(start_frame * frame_duration, 2), round(end_frame * frame_duration, 2), label))
            start_frame = i
            current_label = predictions[i]
    
    # Προσθήκη τελευταίου διαστήματος
    end_frame = len(predictions)
    label = "foreground" if current_label == 1 else "background"
    segments.append((round(start_frame * frame_duration, 2), round(end_frame * frame_duration, 2), label))

    return segments

# Επεξεργασία δεδομένων εκπαίδευσης
X_speech, y_speech = process_dir_train(train_speech_dir, label=1)
X_noise, y_noise = process_dir_train(train_noise_dir, label=0)
X_test_list, filenames = process_dir_test(test_dir)

# Συνένωση και ανακάτεμα δεδομένων
X = np.vstack([X_speech, X_noise])
y = np.hstack([y_speech, y_noise])
X, y = shuffle(X, y, random_state=42)
X_test_full = np.vstack(X_test_list)

# Διαχωρισμός φωνής/θορύβου
X_speech_only = X[y == 1]
X_noise_only = X[y == 0]

# Εξισορρόπηση δεδομένων (ίσο πλήθος από κάθε κατηγορία)
n_samples = min(len(X_speech_only), len(X_noise_only))
X_speech_bal = resample(X_speech_only, n_samples=n_samples, random_state=0)
X_noise_bal = resample(X_noise_only, n_samples=n_samples, random_state=0)
X_balanced = np.vstack([X_speech_bal, X_noise_bal])
y_balanced = np.hstack([np.ones(n_samples), np.zeros(n_samples)])

# Διαχωρισμός σε εκπαίδευση/επικύρωση και κανονικοποίηση
X_train, X_val, y_train, y_val = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test_full)

# Εκπαίδευση μοντέλων
ls_model = LinearRegression()
ls_model.fit(X_train_scaled, y_train)

mlp_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
mlp_model.fit(X_train_scaled, y_train)

# Προβλέψεις για επικύρωση και δοκιμή
y_pred_ls = ls_model.predict(X_val_scaled)
y_pred_ls_bin = (y_pred_ls >= 0.5).astype(int)  # Δυαδική μετατροπή

y_pred_mlp = mlp_model.predict(X_val_scaled)

y_pred_ls_test = ls_model.predict(X_test_scaled)
y_pred_ls_bin_test = (y_pred_ls_test >= 0.5).astype(int)
y_pred_mlp_test = mlp_model.predict(X_test_scaled)

# Εμφάνιση βασικών πληροφοριών
print(" ")
print("Η εξαγωγή χαρακτηριστικών ολοκληρώθηκε!")
print("Διάσταση χαρακτηριστικών:", X.shape)
print("Foreground καρέ:", np.sum(y == 1))
print("Background καρέ:", np.sum(y == 0))
print(" ")
print("Διάσταση εξισορροπημένων χαρακτηριστικών:", X_balanced.shape)
print("Κατανομή εξισορροπημένων ετικετών:", np.bincount(y_balanced.astype(int)))

# Αξιολόγηση των δύο μοντέλων
eval_model(y_val, y_pred_ls_bin, model_name="Least Squares")
eval_model(y_val, y_pred_mlp, model_name="MLP")

# Εξαγωγή αποτελεσμάτων σε CSV
with open(output_csv_path, mode="w", newline='') as f:
    write = csv.writer(f)
    write.writerow(["Audiofile", "start", "end", "class"])  # Επικεφαλίδες

    index = 0
    for fname, feats in zip(filenames, X_test_list):
        n_frames = feats.shape[0]
        pred_mlp = y_pred_mlp_test[index:index + n_frames]

        segments = get_labeled_segments(pred_mlp)

        for start, end, label in segments:
            write.writerow([fname, f"{start:.2f}", f"{end:2f}", label])
        
        index += n_frames

print(f" ")
print(f" Τα αποτελέσματα γράφτηκαν στο {output_csv_path}")
