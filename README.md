# Speech-Segmentation-System
**Voice vs Noise Classification — Audio Processing Project

This project performs audio analysis, feature extraction, and machine learning model training to detect whether segments of an audio file are speech (foreground) or noise (background). It uses Librosa for audio feature extraction and Scikit-learn for model training and evaluation.**

Overview:

The pipeline:
1. Loads .wav audio files.
2. Extracts MFCC features for each frame.
3. Trains two models:
   - Linear Regression
   - MLP (Multi-Layer Perceptron) Neural Network
4. Classifies each frame as speech or noise.
5. Groups predictions into time segments.
6. Saves results in a CSV file (results.csv).

Dependencies:

Install the required libraries:
librosa, scikit-learn, numpy, torch

Note: Ensure you are using Python version 3.10–3.13 (Librosa does not yet support Python 3.14).

Project Structure:

project/
│
├── train/
│   ├── speech/
│   │   └── librivox/         # Speech training samples (.wav)
│   └── noise/
│       └── free-sound/       # Noise training samples (.wav)
│
├── test/                     # Test audio files (.wav)
│
├── epe3ergasia2025.py        # Main Python script
└── results.csv               # Output file

Feature Extraction Parameters:

SAMPLE_RATE: 16000 Hz
N_MFCC: 13
FRAME_DURATION: 0.025 s
HOP_DURATION: 0.010 s
N_FFT: 400
HOP_LENGTH: 160

Functions Explained:

extract_feautures(filepath, label)
- Loads an audio file, extracts its MFCC features, and assigns a label to each frame (1 = speech, 0 = noise).

process_dir_train(directory, label)
- Iterates through all .wav files in a training folder, extracting features and stacking them into arrays for model training.

process_dir_test(directory)
- Processes test audio files, extracting MFCCs for prediction.
- Labels are not required (set to 1 by default).

eval_model(y_true, y_pred, model_name="")
- Prints evaluation metrics: Accuracy, Precision, Recall, F1 Score

get_labeled_segments(predictions, frame_duration)
- Converts frame-by-frame predictions into labeled time segments.

Data Preparation:

1. Load and extract features from speech and noise training sets.
2. Combine and shuffle the datasets.
3. Balance them using resampling to ensure equal representation of both classes.

Training and Validation:

The data is split into training and validation sets (80/20) and normalized using StandardScaler() for better model performance.

Two models are trained:
- Linear Regression (baseline)
- MLP Neural Network with 64 and 32 hidden units, max_iter=300

Predictions and Evaluation:

Both models make predictions on the validation set. Linear Regression predictions are converted to binary (>=0.5 → 1).

Models are evaluated using accuracy, precision, recall, and F1 score.

Output to CSV:

Predicted segments are saved into results.csv with columns:
Audiofile, Start (s), End (s), Class (foreground/background)

Example:
test1.wav,0.00,1.25,foreground
test1.wav,1.25,3.80,background

Console Output Example:

Η εξαγωγή χαρακτηριστικών ολοκληρώθηκε!         # Feature extraction complete!
Διάσταση χαρακτηριστικών: (125000, 13)           # Feature shape: (125000, 13)
Foreground καρέ: 63000                           # Foreground frames: 63000
Background καρέ: 62000                           # Background frames: 62000

Διάσταση εξισορροπημένων χαρακτηριστικών: (124000, 13)  # Balanced feature shape: (124000, 13)
Κατανομή εξισορροπημένων ετικετών: [62000 62000]      # Balanced label distribution: [62000 62000]

Αξιολόγηση μοντέλου Least Squares:               # Evaluation for model Least Squares:
Accurancy: 0.8750                                 # Accuracy: 0.8750
Precision: 0.8602                                 # Precision: 0.8602
Recall: 0.8891                                    # Recall: 0.8891
F1 Δείκτης: 0.8743                                 # F1 Score: 0.8743

Αξιολόγηση μοντέλου MLP:                          # Evaluation for model MLP:
Accurancy: 0.9425                                 # Accuracy: 0.9425
Precision: 0.9378                                 # Precision: 0.9378
Recall: 0.9481                                    # Recall: 0.9481
F1 Δείκτης: 0.9429                                 # F1 Score: 0.9429

Τα αποτελέσματα γράφτηκαν στο results.csv          # Results written to results.csv

Notes:

- Ensure your dataset folders follow the structure shown above.
- Audio files must be mono and .wav format.
- You can change model parameters or MFCC count to experiment with performance.

Results:

After execution, the program prints evaluation metrics and creates a results.csv file listing the start, end, and class (foreground/background) for each segment in the test audio files.

Author:

Developed as part of an Audio Processing & Machine Learning project. The code demonstrates a complete workflow for supervised classification of sound signals.
