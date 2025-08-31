

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Conv1D, MaxPooling1D,
    BatchNormalization, Concatenate,
    MultiHeadAttention, Add, LayerNormalization,
    Bidirectional, LSTM
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers.schedules import CyclicalLearningRate
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix, roc_curve, cohen_kappa_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive


drive.mount('/content/drive')


BASE_DIR       = "/content/drive/MyDrive"
DATA_SAVE_DIR  = os.path.join(BASE_DIR, "LoadedArrays")
OUTPUT_DIR     = os.path.join(BASE_DIR, "Processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("[INFO] Loading pre-saved data...")
X_train_full     = np.load(os.path.join(DATA_SAVE_DIR, "X_train_full.npy"))
y_train_full     = np.load(os.path.join(DATA_SAVE_DIR, "y_train_full.npy"))
X_test           = np.load(os.path.join(DATA_SAVE_DIR, "X_test.npy"))
y_test           = np.load(os.path.join(DATA_SAVE_DIR, "y_test.npy"))
names_train_full = np.load(os.path.join(DATA_SAVE_DIR, "names_train_full.npy"))
names_test       = np.load(os.path.join(DATA_SAVE_DIR, "names_test.npy"))

print(f"X_train_full shape: {X_train_full.shape}, y_train_full shape: {y_train_full.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")


RANDOM_SEED = 42
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.2, random_state=RANDOM_SEED, stratify=y_train_full
)
print(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}")


# Build the Super Advanced Model

def build_super_advanced_model(input_shape):
    inputs = Input(shape=input_shape)

    # Multi-Scale Convolutional Branches
    def conv_branch(kernel_size):
        x = Conv1D(64, kernel_size, activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x)
        return Dropout(0.3)(x)

    b1 = conv_branch(3)
    b2 = conv_branch(5)
    b3 = conv_branch(7)
    merged = Concatenate()([b1, b2, b3])
    merged = BatchNormalization()(merged)
    merged = Dropout(0.3)(merged)

    # Transformer Block 1
    attn = MultiHeadAttention(num_heads=8, key_dim=64)(merged, merged)
    attn = Dropout(0.3)(attn)
    attn = Add()([merged, attn])
    attn = LayerNormalization(epsilon=1e-6)(attn)

    # Conv + Transformer Block 2
    conv = Conv1D(128, 3, activation='relu', padding='same')(attn)
    conv = BatchNormalization()(conv)
    conv = MaxPooling1D(2)(conv)
    conv = Dropout(0.3)(conv)

    attn2 = MultiHeadAttention(num_heads=8, key_dim=128)(conv, conv)
    attn2 = Dropout(0.3)(attn2)
    attn2 = Add()([conv, attn2])
    attn2 = LayerNormalization(epsilon=1e-6)(attn2)

    # Bi-LSTM Layers
    lstm = Bidirectional(LSTM(128, return_sequences=True))(attn2)
    lstm = Dropout(0.4)(lstm)
    lstm = Bidirectional(LSTM(64))(lstm)
    lstm = Dropout(0.4)(lstm)

    # Dense Classifier
    x = Dense(256, activation='relu')(lstm)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    outputs = Dense(1, activation='sigmoid')(x)
    return Model(inputs=inputs, outputs=outputs)

input_shape = X_train.shape[1:]
model = build_super_advanced_model(input_shape)
model.summary()


# Cyclical Learning Rate Setup

batch_size   = 32
train_steps  = len(X_train) // batch_size

clr = CyclicalLearningRate(
    initial_learning_rate=1e-6,
    maximal_learning_rate=1e-3,
    step_size=train_steps * 2,   # half-cycle = 2 epochs
    scale_fn=lambda x: 1.0,      # triangular policy
    scale_mode='cycle'
)


#Compile & Train the Model

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=clr),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(
        monitor='val_loss', patience=10,
        restore_best_weights=True, verbose=1
    ),
    ModelCheckpoint(
        filepath=os.path.join(OUTPUT_DIR, 'best_super_advanced_model.h5'),
        monitor='val_loss', save_best_only=True, verbose=1
    )
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=batch_size,
    callbacks=callbacks,
    verbose=1
)


#Evaluate on Validation Set & Plot Metrics

# Evaluate
val_loss, _ = model.evaluate(X_val, y_val, verbose=0)
y_val_prob  = model.predict(X_val)
y_val_pred  = (y_val_prob > 0.5).astype(int).ravel()

# Compute metrics
val_accuracy  = accuracy_score(y_val, y_val_pred)
val_precision = precision_score(y_val, y_val_pred)
val_recall    = recall_score(y_val, y_val_pred)
val_f1        = f1_score(y_val, y_val_pred)
val_roc_auc   = roc_auc_score(y_val, y_val_prob)
val_kappa     = cohen_kappa_score(y_val, y_val_pred)

print("\nValidation Metrics:")
print(f"Loss:        {val_loss:.4f}")
print(f"Accuracy:    {val_accuracy*100:.2f}%")
print(f"Precision:   {val_precision*100:.2f}%")
print(f"Recall:      {val_recall*100:.2f}%")
print(f"F1 Score:    {val_f1*100:.2f}%")
print(f"ROC-AUC:     {val_roc_auc:.4f}")
print(f"Cohen's Kappa: {val_kappa:.4f}\n")

print("Classification Report:")
print(classification_report(y_val, y_val_pred, target_names=['Real','Fake']))

# Confusion Matrix
cm_val = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Real','Fake'], yticklabels=['Real','Fake'])
plt.title("Validation Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_val, y_val_prob)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {val_roc_auc:.4f})")
plt.plot([0,1],[0,1],'--', lw=2, color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Validation ROC Curve")
plt.legend(loc="lower right")
plt.show()

# Loss Curves
plt.figure(figsize=(8,6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs. Validation Loss")
plt.legend()
plt.show()

#Evaluate on Test Data & Plot Metrics

if X_test is not None:
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    y_test_prob = model.predict(X_test)
    y_test_pred = (y_test_prob > 0.5).astype(int).ravel()

    test_accuracy  = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall    = recall_score(y_test, y_test_pred)
    test_f1        = f1_score(y_test, y_test_pred)
    test_roc_auc   = roc_auc_score(y_test, y_test_prob)
    test_kappa     = cohen_kappa_score(y_test, y_test_pred)

    print("\nTest Metrics:")
    print(f"Loss:         {test_loss:.4f}")
    print(f"Accuracy:     {test_accuracy*100:.2f}%")
    print(f"Precision:    {test_precision*100:.2f}%")
    print(f"Recall:       {test_recall*100:.2f}%")
    print(f"F1 Score:     {test_f1*100:.2f}%")
    print(f"ROC-AUC:      {test_roc_auc:.4f}")
    print(f"Cohen's Kappa:{test_kappa:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=['Real','Fake']))

    # Test Confusion Matrix
    cm_test = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Real','Fake'], yticklabels=['Real','Fake'])
    plt.title("Test Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # Test ROC Curve
    fpr_t, tpr_t, _ = roc_curve(y_test, y_test_prob)
    plt.figure(figsize=(5,4))
    plt.plot(fpr_t, tpr_t, lw=2, label=f"ROC (AUC = {test_roc_auc:.4f})")
    plt.plot([0,1],[0,1],'--', lw=2, color='navy')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Test ROC Curve")
    plt.legend(loc="lower right")
    plt.show()
else:
    print("\nNo test data found; skipping test evaluation.")


#Save Final Model and Training History

final_model_path = os.path.join(OUTPUT_DIR, "final_super_advanced_model.h5")
model.save(final_model_path)
print(f"\nFinal model saved to: {final_model_path}")

history_path = os.path.join(OUTPUT_DIR, "super_training_history.npy")
np.save(history_path, history.history)
print(f"Training history saved to: {history_path}")

!pip install mediapipe mtcnn opencv-python-headless tqdm

import os
import cv2
import numpy as np
from mtcnn import MTCNN
import mediapipe as mp
from tqdm import tqdm
from google.colab import drive
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed


drive.mount('/content/drive')


BASE_DIR = '/content/drive/MyDrive'
test_dir = os.path.join(BASE_DIR, 'Test')
if not os.path.exists(test_dir):
    raise ValueError(f"Directory {test_dir} does not exist. Please check the path.")


logging.basicConfig(
    filename='preprocessing.log',
    filemode='a',  # Append mode so logs are preserved across runs
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


IMPORTANT_LANDMARKS = [
    # Eyes
    33, 133, 362, 263,
    # Eyebrows
    70, 63, 105, 66, 107, 336,
    # Nose
    168, 6, 197, 195, 5, 4,
    # Mouth
    78, 308, 14, 17, 287, 57, 318, 402, 324,
    # Cheeks and Additional Key Points
    61, 291, 199, 405, 310, 311, 312, 13, 82, 81
]


def compute_bbme_flow(prev_gray: np.ndarray,
                      curr_gray: np.ndarray,
                      block_size: int = 16,
                      search_range: int = 4) -> np.ndarray:
    """
    Compute block-based motion estimation between two grayscale frames.
    Returns an HSV-encoded BGR image of the dense flow.
    """
    h, w = prev_gray.shape
    # Number of blocks
    nb_y = h // block_size
    nb_x = w // block_size

    # Initialize flow vector fields
    flow_u = np.zeros((nb_y, nb_x), dtype=np.float32)
    flow_v = np.zeros((nb_y, nb_x), dtype=np.float32)

    # For each block
    for by in range(nb_y):
        for bx in range(nb_x):
            y0 = by * block_size
            x0 = bx * block_size
            block = prev_gray[y0:y0+block_size, x0:x0+block_size]

            best_sad = float('inf')
            best_dx, best_dy = 0, 0

            # Search in ±search_range
            for dy in range(-search_range, search_range+1):
                for dx in range(-search_range, search_range+1):
                    y1 = y0 + dy
                    x1 = x0 + dx
                    # Check bounds
                    if (y1 < 0 or y1+block_size > h or
                        x1 < 0 or x1+block_size > w):
                        continue
                    candidate = curr_gray[y1:y1+block_size, x1:x1+block_size]
                    sad = np.sum(np.abs(block.astype(np.int16) - candidate.astype(np.int16)))
                    if sad < best_sad:
                        best_sad = sad
                        best_dx, best_dy = dx, dy

            flow_u[by, bx] = best_dx
            flow_v[by, bx] = best_dy

    # Create HSV image
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    # Compute magnitude & angle per block, then fill each block
    mag = np.sqrt(flow_u**2 + flow_v**2)
    ang = np.arctan2(flow_v, flow_u)  # radians

    # Normalize angle to [0, 180] for hue
    hue_block = ((ang + np.pi) / (2 * np.pi) * 180).astype(np.uint8)
    # Normalize magnitude to [0, 255] for value
    mag_norm = np.clip((mag / mag.max()) * 255, 0, 255).astype(np.uint8)

    # Fill HSV blocks
    for by in range(nb_y):
        for bx in range(nb_x):
            y0 = by * block_size
            x0 = bx * block_size
            hsv[y0:y0+block_size, x0:x0+block_size, 0] = hue_block[by, bx]
            hsv[y0:y0+block_size, x0:x0+block_size, 1] = 255
            hsv[y0:y0+block_size, x0:x0+block_size, 2] = mag_norm[by, bx]

    # Convert HSV to BGR for visualization
    flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return flow_bgr


def extract_frames(video_path: str, desired_fps: int = 60) -> list:
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps == 0:
        logging.warning(f"Video {video_path} FPS=0; skipping.")
        return []
    frame_skip = max(int(original_fps / desired_fps), 1)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i % frame_skip == 0:
            frames.append(frame)
    cap.release()
    logging.info(f"Extracted {len(frames)} frames from {video_path}")
    return frames

def enhance_frame(frame: np.ndarray) -> np.ndarray:
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    return cv2.GaussianBlur(enhanced, (3, 3), 0)

def align_face(cropped_face: np.ndarray, face_mesh) -> np.ndarray:
    rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return None
    lm = res.multi_face_landmarks[0].landmark
    h, w, _ = cropped_face.shape
    pts = [(int(p.x*w), int(p.y*h)) for p in lm]
    l, r = pts[33], pts[362]
    center = ((l[0]+r[0])//2, (l[1]+r[1])//2)
    dy, dx = r[1]-l[1], r[0]-l[0]
    angle = np.degrees(np.arctan2(dy, dx))
    desired_w, desired_h = 256, 256
    left_eye = (0.35, 0.35)
    dist = np.hypot(dx, dy)
    desired_dist = (1.0 - 2*left_eye[0])*desired_w
    scale = desired_dist / dist
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0,2] += desired_w*0.5 - center[0]
    M[1,2] += desired_h*left_eye[1] - center[1]
    return cv2.warpAffine(cropped_face, M, (desired_w, desired_h), flags=cv2.INTER_CUBIC)

def extract_landmarks(aligned: np.ndarray, face_mesh, important_idxs: list) -> np.ndarray:
    rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return None
    lm = res.multi_face_landmarks[0].landmark
    h, w, _ = aligned.shape
    pts = np.array([[lm[i].x*w, lm[i].y*h] for i in important_idxs])
    norm = np.stack([pts[:,0]/w, pts[:,1]/h], axis=1).flatten()
    return norm

def segment_sequences(landmarks_list: list, sequence_length: int = 16, step: int = 4) -> list:
    seqs = []
    for i in range(0, len(landmarks_list)-sequence_length+1, step):
        seqs.append(np.concatenate(landmarks_list[i:i+sequence_length]))
    return seqs


def process_single_video(video: str, input_dir: str, output_dir: str, important_landmarks: list):
    name = os.path.splitext(video)[0]
    video_out = os.path.join(output_dir, name)
    faces_dir      = os.path.join(video_out, 'faces')
    landmarks_dir  = os.path.join(video_out, 'landmarks')
    sequences_dir  = os.path.join(video_out, 'sequences')
    flow_dir       = os.path.join(video_out, 'optical_flow')

    # Skip if already processed
    if os.path.exists(sequences_dir) and os.listdir(sequences_dir):
        print(f"Skipping {video}: already processed.")
        return

    # Create dirs
    for d in (faces_dir, landmarks_dir, sequences_dir, flow_dir):
        os.makedirs(d, exist_ok=True)

    frames = extract_frames(os.path.join(input_dir, video))
    prev_gray = None
    landmarks_list = []

    with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:

        detector = MTCNN()

        for idx, frame in enumerate(frames):
            det = detector.detect_faces(frame)
            if not det:
                logging.warning(f"No face in frame {idx} of {video}")
                continue
            x,y,w_box,h_box = det[0]['box']
            x,y = max(0,x), max(0,y)
            crop = frame[y:y+h_box, x:x+w_box]
            if crop.size == 0:
                continue

            enhanced = enhance_frame(crop)
            aligned = align_face(enhanced, face_mesh)
            if aligned is None:
                continue

            # Save aligned face
            cv2.imwrite(os.path.join(faces_dir, f"frame_{idx:05d}.jpg"), aligned)

            # Landmarks
            lm = extract_landmarks(aligned, face_mesh, important_landmarks)
            if lm is None:
                continue
            landmarks_list.append(lm)
            np.save(os.path.join(landmarks_dir, f"frame_{idx:05d}.npy"), lm)

            # Optical flow (BBME) between previous and current aligned frames
            gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                flow_img = compute_bbme_flow(prev_gray, gray)
                cv2.imwrite(os.path.join(flow_dir, f"flow_{idx:05d}.jpg"), flow_img)
            prev_gray = gray

    # Sequence segmentation
    if len(landmarks_list) >= 16:
        seqs = segment_sequences(landmarks_list, sequence_length=16, step=4)
        for i, seq in enumerate(seqs):
            np.save(os.path.join(sequences_dir, f"sequence_{i:05d}.npy"), seq)
        print(f"Processed video: {video}")
    else:
        logging.warning(f"Not enough frames for sequences in {video}")


def main():
    processed_test_dir = os.path.join(BASE_DIR, 'Processed', 'Test')
    os.makedirs(processed_test_dir, exist_ok=True)

    videos = [f for f in os.listdir(test_dir)
              if f.lower().endswith(('.mp4','.avi','.mov','.mkv'))]
    print(f"Total videos to process: {len(videos)}")

    with ThreadPoolExecutor(max_workers=2) as exec:
        futures = [
            exec.submit(process_single_video, v, test_dir, processed_test_dir, IMPORTANT_LANDMARKS)
            for v in videos
        ]
        for _ in tqdm(as_completed(futures), total=len(videos), desc="Processing Test Videos"):
            pass

    print("Preprocessing complete for Test folder.")

if __name__ == "__main__":
    main()