

import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from collections import deque
import logging
from scipy.ndimage import median_filter
import shutil
from google.colab import drive
from sklearn.model_selection import train_test_split
import random

# Configure logging for informational messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Mount Google Drive
drive.mount('/content/drive')

# Define input/output paths
celeb_real_dir = '/content/drive/MyDrive/Celeb-real'
celeb_synth_dir = '/content/drive/MyDrive/Celeb-synthesis'
output_base     = '/content/drive/MyDrive/blinksss_all'
train_dir       = os.path.join(output_base, 'train')
test_dir        = os.path.join(output_base, 'test')

# Ensure output directories exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir,  exist_ok=True)

# Calibration & detection parameters
INITIAL_CALIB_FRAMES    = 30      # frames for initial eye-aspect-ratio baseline
DYNAMIC_BUFFER_SIZE     = 30      # buffer length for dynamic thresholding
EAR_OPEN_SENSITIVITY    = 0.80    # proportion of baseline EAR for open-eye
CLOSED_FACTOR           = 0.70    # proportion of threshold to detect closure
SMOOTH_KERNEL           = 3       # median filter size for smoothing

# Augmentation settings
AUGMENT_PROBABILITY     = 0.3     # chance to augment each frame
MAX_BRIGHTNESS_DELTA    = 30      # brightness shift range
MAX_ROTATION_DEG        = 5       # rotation angle range

# Face ROI margin and eye landmark indices (MediaPipe)
FACE_MARGIN_RATIO       = 0.05
LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Buffer for dynamic thresholding
ear_buffer = deque(maxlen=DYNAMIC_BUFFER_SIZE)


def augment_frame(frame):
    """Apply random brightness shift or small rotation."""
    if random.random() < 0.5:
        delta = random.randint(-MAX_BRIGHTNESS_DELTA, MAX_BRIGHTNESS_DELTA)
        frame = cv2.convertScaleAbs(frame, alpha=1, beta=delta)
    angle = random.uniform(-MAX_ROTATION_DEG, MAX_ROTATION_DEG)
    h, w = frame.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(frame, M, (w, h))


def compute_eye_aspect_ratio(landmarks, indices):
    """Compute the eye aspect ratio from six landmarks."""
    try:
        pts = [np.array((landmarks[i].x, landmarks[i].y)) for i in indices]
    except IndexError:
        return 0.0
    vertical_1 = np.linalg.norm(pts[1] - pts[5])
    vertical_2 = np.linalg.norm(pts[2] - pts[4])
    horizontal = np.linalg.norm(pts[0] - pts[3])
    return ((vertical_1 + vertical_2) / (2 * horizontal)) if horizontal > 1e-6 else 0.0


def adapt_threshold(current_ear, prev_threshold):
    """Update threshold using a running median of recent EAR values."""
    ear_buffer.append(current_ear)
    if len(ear_buffer) >= DYNAMIC_BUFFER_SIZE // 2:
        baseline = np.median(ear_buffer)
        target = baseline * EAR_OPEN_SENSITIVITY
        return prev_threshold * 0.9 + target * 0.1
    return prev_threshold


def median_smooth(sequence):
    """Apply median filtering to smooth the EAR sequence."""
    return median_filter(sequence, size=SMOOTH_KERNEL)


def extract_face_roi(frame, landmarks):
    """Crop a bounding box around the face with a small margin."""
    h, w = frame.shape[:2]
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    min_x, max_x = max(min(xs), 0), min(max(xs), 1)
    min_y, max_y = max(min(ys), 0), min(max(ys), 1)

    dx = (max_x - min_x) * FACE_MARGIN_RATIO
    dy = (max_y - min_y) * FACE_MARGIN_RATIO
    x1, x2 = int((min_x - dx) * w), int((max_x + dx) * w)
    y1, y2 = int((min_y - dy) * h), int((max_y + dy) * h)

    return frame[y1:y2, x1:x2]


def extract_blink_features(video_path, dest_dir, label):
    """
    Process a single video to compute:
      - EAR time series
      - total blink count
      - blink durations
      - average blink duration
      - blink frequency
    Saves a .npz file with these features.
    """
    cap = cv2.VideoCapture(video_path)
    ear_vals = []
    blink_count = 0
    blink_in_progress = False
    blink_len = 0
    durations = []
    frame_idx = 0
    threshold = 0.21
    last_valid = threshold
    calib_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if random.random() < AUGMENT_PROBABILITY:
            frame = augment_frame(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            ear_vals.append(last_valid)
            continue

        landmarks = results.multi_face_landmarks[0].landmark
        left_ear  = compute_eye_aspect_ratio(landmarks, LEFT_EYE_IDX)
        right_ear = compute_eye_aspect_ratio(landmarks, RIGHT_EYE_IDX)
        avg_ear   = (left_ear + right_ear) / 2 or last_valid
        last_valid = avg_ear

        if frame_idx <= INITIAL_CALIB_FRAMES:
            calib_data.append(avg_ear)
            threshold = np.median(calib_data) * EAR_OPEN_SENSITIVITY
        else:
            if avg_ear > threshold:
                threshold = adapt_threshold(avg_ear, threshold)
            close_th = threshold * CLOSED_FACTOR
            open_th  = threshold

            if not blink_in_progress and avg_ear < close_th:
                blink_in_progress = True
                blink_len = 1
            elif blink_in_progress and avg_ear > open_th:
                if blink_len >= 2:
                    blink_count += 1
                    durations.append(blink_len)
                blink_in_progress = False
                blink_len = 0
            elif blink_in_progress:
                blink_len += 1

        ear_vals.append(avg_ear)

    cap.release()

    ear_array = median_smooth(np.array(ear_vals))
    total_frames = max(frame_idx, 1)
    freq = blink_count / total_frames
    avg_dur = float(np.mean(durations)) if durations else 0.0

    base = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(dest_dir, f"{base}.npz")
    np.savez(
        out_path,
        ear_sequence=ear_array,
        total_blinks=blink_count,
        blink_durations=np.array(durations, dtype=np.int32),
        avg_blink_duration=avg_dur,
        blink_frequency=freq,
        label=label
    )
    logging.info(f"Saved {out_path}: blinks={blink_count}, freq={freq:.3f}")


def batch_process_videos(source_dirs, labels, dest_dir):
    """Process all .mp4/.avi videos in each source directory."""
    for folder, lbl in zip(source_dirs, labels):
        if not os.path.isdir(folder):
            logging.warning(f"Skipping missing folder: {folder}")
            continue
        vids = [f for f in os.listdir(folder) if f.lower().endswith(('.mp4', '.avi'))]
        logging.info(f"Found {len(vids)} videos in {folder} (label={lbl})")
        for vid in tqdm(vids, desc=f"Processing {os.path.basename(folder)}"):
            extract_blink_features(os.path.join(folder, vid), dest_dir, lbl)


def split_dataset(src_dir, train_dir, test_dir, test_frac=0.2):
    """Split .npz files into train/test sets and move them accordingly."""
    files = [f for f in os.listdir(src_dir) if f.endswith('.npz')]
    if not files:
        logging.error("No .npz files to split.")
        return
    train, test = train_test_split(files, test_size=test_frac, random_state=42)
    for f in train:
        shutil.move(os.path.join(src_dir, f), os.path.join(train_dir, f))
    for f in test:
        shutil.move(os.path.join(src_dir, f), os.path.join(test_dir, f))
    logging.info(f"Split {len(files)} files: {len(train)} train, {len(test)} test")


if __name__ == '__main__':
    # Define inputs and execute the pipeline
    inputs = [celeb_real_dir, celeb_synth_dir]
    labels = [0, 1]
    batch_process_videos(inputs, labels, output_base)
    split_dataset(output_base, train_dir, test_dir, test_frac=0.2)
    logging.info("Pipeline completed successfully.")

import os
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, roc_auc_score,
                             precision_recall_curve, auc)
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

BLINKS_DIR  = "/content/drive/MyDrive/blinkss_all"
# Subdirectories for train/test split
TRAIN_DIR   = os.path.join(BLINKS_DIR, "train")
TEST_DIR    = os.path.join(BLINKS_DIR, "test")
# Sequence length and feature dimensionality
MAX_LEN     = 200   # pad/truncate sequences to this length
FEATURE_DIM = 1     # dimensionality of each time step
# Validation split fraction and batch size
VAL_SPLIT   = 0.2
BATCH_SIZE  = 8
# Hyperparameter tuning and epochs
TUNER_TRIALS = 20
BASE_EPOCHS  = 25
FINAL_EPOCHS = 60
# Toggle for data augmentation in training
DO_AUGMENT   = True


def load_npz_data(folder, max_len=MAX_LEN, feature_dim=FEATURE_DIM):
    npz_files = [f for f in os.listdir(folder) if f.lower().endswith(".npz")]
    X_seq, X_blink, y = [], [], []

    for fname in npz_files:
        path = os.path.join(folder, fname)
        data = np.load(path)
        # Skip files missing required keys
        if not all(k in data for k in ("ear_sequence", "total_blinks", "label")):
            print(f"[WARN] Missing keys in {fname}. Skipping.")
            continue

        ear_seq   = data["ear_sequence"]
        tot_blink = int(data["total_blinks"].item())
        label     = int(data["label"].item())

        # Pad or truncate 1D sequence and reshape to (MAX_LEN, feature_dim)
        if ear_seq.ndim == 1 and feature_dim == 1:
            length = len(ear_seq)
            if length < max_len:
                ear_seq = np.pad(ear_seq, (0, max_len - length), mode='constant')
            else:
                ear_seq = ear_seq[:max_len]
            ear_seq = ear_seq.reshape(max_len, 1)
        # Pad or truncate 2D sequence
        elif ear_seq.ndim == 2 and ear_seq.shape[1] == feature_dim:
            length = ear_seq.shape[0]
            if length < max_len:
                pad_len = max_len - length
                ear_seq = np.pad(ear_seq, ((0, pad_len), (0, 0)), mode='constant')
            else:
                ear_seq = ear_seq[:max_len, :]
        else:
            print(f"[WARN] Shape mismatch in {fname} => {ear_seq.shape}. Skipping.")
            continue

        X_seq.append(ear_seq)
        X_blink.append([float(tot_blink)])
        y.append(label)

    # Return None if no valid data found
    if not X_seq:
        return None, None, None

    return (
        np.array(X_seq, dtype=np.float32),
        np.array(X_blink, dtype=np.float32),
        np.array(y, dtype=np.int32)
    )

def random_augment(ear_seq, total_blink):
    max_shift = 5
    shift = np.random.randint(-max_shift, max_shift + 1)
    ear_seq = np.roll(ear_seq, shift, axis=0)
    # Fill rolled-over values
    if shift > 0:
        ear_seq[:shift] = ear_seq[shift]
    elif shift < 0:
        ear_seq[shift:] = ear_seq[shift - 1]

    # Add small Gaussian noise
    noise_scale = 0.01
    ear_seq += np.random.normal(0, noise_scale, ear_seq.shape)

    # Randomly adjust blink count by ±1 (50% chance)
    if np.random.rand() < 0.5:
        total_blink = max(0, total_blink + np.random.choice([-1, 1]))

    return ear_seq, total_blink

class AugmentSequence(tf.keras.utils.Sequence):
    def __init__(self, X_seq, X_blink, y, batch_size=BATCH_SIZE, shuffle=True):
        self.X_seq   = X_seq
        self.X_blink = X_blink
        self.y       = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(X_seq))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.X_seq) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        seq_batch, blink_batch, labels = [], [], []

        for i in batch_idx:
            seq = self.X_seq[i].copy()
            bl  = float(self.X_blink[i, 0])
            lb  = self.y[i]

            if DO_AUGMENT:
                seq, bl = random_augment(seq, bl)

            seq_batch.append(seq)
            blink_batch.append([bl])
            labels.append(lb)

        return (
            {"seq_input": np.array(seq_batch, dtype=np.float32),
             "blink_input": np.array(blink_batch, dtype=np.float32)},
            np.array(labels, dtype=np.int32)
        )

class Attention(layers.Layer):
    def build(self, input_shape):
        d = input_shape[-1]
        # Weight matrices for computing attention scores
        self.W = self.add_weight("W", shape=(d, d), initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight("b", shape=(d,),    initializer="zeros",          trainable=True)
        self.u = self.add_weight("u", shape=(d, 1),  initializer="glorot_uniform", trainable=True)
        super().build(input_shape)

    def call(self, x):
        # x: (batch, time, features)
        uit = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)  # (batch, time, features)
        ait = tf.tensordot(uit, self.u, axes=1)                  # (batch, time)
        a   = tf.nn.softmax(ait, axis=1)                         # attention weights
        return tf.reduce_sum(x * tf.expand_dims(a, -1), axis=1) # (batch, features)

class DynamicParamCallback(keras.callbacks.Callback):
    def __init__(self, min_dropout=0.1, change_factor=0.05, patience=2):
        super().__init__()
        self.min_dropout       = min_dropout
        self.change_factor     = change_factor
        self.patience          = patience
        self.best_val_loss     = np.inf
        self.epochs_no_improve = 0

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss')
        if val_loss is None:
            return

        if val_loss < self.best_val_loss:
            self.best_val_loss     = val_loss
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1

        if self.epochs_no_improve >= self.patience:
            self.epochs_no_improve = 0
            # Reduce each registered dropout layer's rate
            if hasattr(self.model, "_dp_layers"):
                for dp in self.model._dp_layers:
                    new_rate = max(self.min_dropout, dp.rate - self.change_factor)
                    dp.rate = new_rate
                print(f"[DynamicParamCallback] Dropout reduced to {new_rate:.3f}")
            else:
                print("[WARN] No _dp_layers attribute found on model.")

def build_super_model(hp):
    # Hyperparameter definitions
    use_bi         = hp.Boolean("bidirectional", default=False)
    cnn_filters    = hp.Int("cnn_filters", 16, 128, step=16)
    cnn_kernel     = hp.Choice("cnn_kernel", [3, 5, 7])
    lstm1_units    = hp.Int("lstm1_units", 32, 256, step=32)
    lstm2_units    = hp.Int("lstm2_units", 0, 128, step=32)
    use_attention  = hp.Boolean("use_attention", default=True)
    dropout_rate   = hp.Float("dropout_rate", 0.0, 0.5, step=0.1)
    dense_units    = hp.Int("dense_units", 16, 128, step=16)
    l2_reg         = hp.Float("l2_reg", 1e-5, 1e-2, sampling="log")
    lr             = hp.Choice("learning_rate", [1e-4, 3e-4, 1e-3])

    dp_layers = []  # Collect Dropout layers for dynamic adjustment

    # Sequence input branch
    seq_in = keras.Input((MAX_LEN, FEATURE_DIM), name="seq_input")
    x = layers.Conv1D(cnn_filters, cnn_kernel, activation='relu',
                      kernel_regularizer=regularizers.l2(l2_reg))(seq_in)
    x = layers.MaxPooling1D(2)(x)

    # Define an LSTM (or Bidirectional) subroutine
    def lstm_block(units, return_seq):
        layer = layers.LSTM(units, return_sequences=return_seq,
                            kernel_regularizer=regularizers.l2(l2_reg))
        return layers.Bidirectional(layer) if use_bi else layer

    # First LSTM layer
    x = lstm_block(lstm1_units, return_seq=(lstm2_units > 0))(x)
    if dropout_rate:
        do1 = layers.Dropout(dropout_rate); dp_layers.append(do1); x = do1(x)

    # Optional second LSTM + attention or flatten
    if lstm2_units > 0:
        x = lstm_block(lstm2_units, return_seq=True)(x)
        if dropout_rate:
            do2 = layers.Dropout(dropout_rate); dp_layers.append(do2); x = do2(x)
        x = Attention()(x) if use_attention else layers.Flatten()(x)
    else:
        x = layers.Flatten()(x)

    # Blink-count input branch
    blink_in = keras.Input((1,), name="blink_input")
    g = layers.Dense(16, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_reg))(blink_in)
    if dropout_rate:
        do_b = layers.Dropout(dropout_rate); dp_layers.append(do_b); g = do_b(g)

    # Merge branches and final classification head
    merged = layers.Concatenate()([x, g])
    h = layers.Dense(dense_units, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_reg))(merged)
    if dropout_rate:
        do_h = layers.Dropout(dropout_rate); dp_layers.append(do_h); h = do_h(h)
    out = layers.Dense(1, activation='sigmoid')(h)

    model = keras.Model([seq_in, blink_in], out, name="BlinkClassifier")
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="binary_crossentropy", metrics=["accuracy"])
    model._dp_layers = dp_layers
    return model

def main():
    # Load and inspect training data
    print("[INFO] Loading data from:", TRAIN_DIR)
    X_seq, X_blink, y = load_npz_data(TRAIN_DIR)
    if X_seq is None:
        print("[ERROR] No valid data found. Exiting.")
        return

    print(f"[INFO] Data shapes — X_seq: {X_seq.shape}, X_blink: {X_blink.shape}, y: {y.shape}")
    print("Label distribution:", np.unique(y, return_counts=True))

    # Split into train/validation sets
    Xs_tr, Xs_val, Xb_tr, Xb_val, y_tr, y_val = train_test_split(
        X_seq, X_blink, y, test_size=VAL_SPLIT, random_state=42
    )

    # Instantiate data generators
    train_gen = AugmentSequence(Xs_tr, Xb_tr, y_tr, batch_size=BATCH_SIZE, shuffle=True)
    val_gen   = AugmentSequence(Xs_val, Xb_val, y_val, batch_size=BATCH_SIZE, shuffle=False)

    # Hyperparameter search with Keras Tuner
    tuner = kt.RandomSearch(
        build_super_model,
        objective="val_accuracy",
        max_trials=TUNER_TRIALS,
        executions_per_trial=1,
        project_name="BlinkAdvancedTuning",
        overwrite=True
    )
    tuner.search(train_gen, validation_data=val_gen, epochs=BASE_EPOCHS)
    best_hp = tuner.get_best_hyperparameters(1)[0]
    print("[INFO] Best hyperparameters:", best_hp.values)

    # Build and summarize the best model
    model = tuner.hypermodel.build(best_hp)
    model.summary()

    # Define callbacks for training
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        DynamicParamCallback(min_dropout=0.1, change_factor=0.05, patience=2)
    ]

    # Final model training
    history = model.fit(train_gen, validation_data=val_gen,
                        epochs=FINAL_EPOCHS, callbacks=callbacks)

    # Plot loss and accuracy curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("Loss vs. Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Binary Crossentropy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Train Acc")
    plt.plot(history.history["val_accuracy"], label="Val Acc")
    plt.title("Accuracy vs. Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Evaluate on validation set without augmentation
    val_gen_no_aug = AugmentSequence(Xs_val, Xb_val, y_val, batch_size=BATCH_SIZE, shuffle=False)
    probs = model.predict(val_gen_no_aug).ravel()
    preds = (probs >= 0.5).astype(int)

    # Print classification metrics
    print("\n[INFO] Validation Classification Report:")
    print(classification_report(y_val, preds, target_names=["Real", "Fake"]))

    # Display confusion matrix
    cm = confusion_matrix(y_val, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Real","Fake"], yticklabels=["Real","Fake"])
    plt.title("Validation Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # Compute and plot ROC and PR curves
    fpr, tpr, _ = roc_curve(y_val, probs)
    roc_auc = roc_auc_score(y_val, probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Validation)")
    plt.legend()
    plt.show()

    precision, recall, _ = precision_recall_curve(y_val, probs)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Validation)")
    plt.legend()
    plt.show()

    # Inference on test set .npz files
    if not os.path.isdir(TEST_DIR):
        print("[WARN] Test directory not found:", TEST_DIR)
        return

    print("\n[INFO] Running inference on test data:")
    for fname in os.listdir(TEST_DIR):
        if not fname.lower().endswith(".npz"):
            continue
        data = np.load(os.path.join(TEST_DIR, fname))
        if not all(k in data for k in ("ear_sequence", "total_blinks", "label")):
            print(f"[WARN] Skipping {fname} (missing keys)")
            continue

        ear_seq = data["ear_sequence"]
        # Pad/truncate test sequence
        if ear_seq.ndim == 1 and FEATURE_DIM == 1:
            L = len(ear_seq)
            pad = MAX_LEN - L
            ear_seq = np.pad(ear_seq, (0, pad), mode="constant") if L < MAX_LEN else ear_seq[:MAX_LEN]
            ear_seq = ear_seq.reshape(1, MAX_LEN, 1)
        elif ear_seq.ndim == 2 and ear_seq.shape[1] == FEATURE_DIM:
            L = ear_seq.shape[0]
            pad = MAX_LEN - L
            ear_seq = (np.pad(ear_seq, ((0, pad), (0, 0)), mode="constant") if L < MAX_LEN
                       else ear_seq[:MAX_LEN, :])
            ear_seq = ear_seq[np.newaxis, ...]
        else:
            print(f"[WARN] Shape mismatch for {fname}. Skipping.")
            continue

        blink_arr = np.array([[float(data["total_blinks"].item())]], dtype=np.float32)
        prob = model.predict({"seq_input": ear_seq, "blink_input": blink_arr})[0][0]
        label_pred = "Fake" if prob >= 0.5 else "Real"
        print(f"{fname} => {label_pred} (score={prob:.4f})")

if __name__ == "__main__":
    main()