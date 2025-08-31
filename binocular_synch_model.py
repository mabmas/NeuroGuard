

import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import math
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from tensorflow.keras.models import Model
from multiprocessing import Pool

BASE_PATH = '/content/drive/MyDrive'
FF_DIR = os.path.join(BASE_PATH, 'FF++')
FF_REAL_DIR = os.path.join(FF_DIR, 'real')
FF_FAKE_DIR = os.path.join(FF_DIR, 'fake')
TEST_DIR = os.path.join(FF_DIR, 'test')
PREPROCESSED_DIR = os.path.join(BASE_PATH, 'synch_preprocessed')
os.makedirs(PREPROCESSED_DIR, exist_ok=True)
SEQUENCE_LENGTH = 30
CNN_MODEL_INPUT_SIZE = (224, 224)
TEST_SPLIT = 0.20

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

KEY_LANDMARKS = {
    'left_iris': 468,
    'right_iris': 473,
    'nose_tip': 1,
    'left_eye_outer': 33,
    'left_eye_inner': 133,
    'right_eye_outer': 362,
    'right_eye_inner': 263,
    'mouth_left': 61,
    'mouth_right': 291,
    'left_cheek': 234,
    'right_cheek': 454,
    'chin': 152
}

ADDITIONAL_LANDMARKS = {
    'left_eye_center': 468,
    'right_eye_center': 473,
}

def extract_landmarks(video_path):
    cap = cv2.VideoCapture(video_path)
    landmarks = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark
            h, w, _ = frame.shape
            frame_landmarks = {}
            for key, idx in {**KEY_LANDMARKS, **ADDITIONAL_LANDMARKS}.items():
                lm = face_landmarks[idx]
                frame_landmarks[key] = (int(lm.x * w), int(lm.y * h))
            landmarks.append(frame_landmarks)
        else:
            landmarks.append(None)
    cap.release()
    return landmarks

def calculate_angle(a, b, c):
    try:
        ba = (a[0] - b[0], a[1] - b[1])
        bc = (c[0] - b[0], c[1] - b[1])
        dot_product = ba[0]*bc[0] + ba[1]*bc[1]
        magnitude_ba = math.sqrt(ba[0]**2 + ba[1]**2)
        magnitude_bc = math.sqrt(bc[0]**2 + bc[1]**2)
        if magnitude_ba * magnitude_bc == 0:
            return 0
        angle_rad = math.acos(dot_product / (magnitude_ba * magnitude_bc))
        return math.degrees(angle_rad)
    except:
        return 0

def extract_features(landmarks):
    features = []
    previous_angles = {}
    for idx, frame_lm in enumerate(landmarks):
        if frame_lm is None:
            features.append({'left_angle': 0, 'right_angle': 0, 'mouth_distance': 0, 'cheek_distance': 0, 'angle_change_left': 0, 'angle_change_right': 0, 'sync_metric': 0})
            continue
        left_angle = calculate_angle(frame_lm['nose_tip'], frame_lm['left_iris'], frame_lm['right_iris'])
        right_angle = calculate_angle(frame_lm['nose_tip'], frame_lm['right_iris'], frame_lm['left_iris'])
        mouth_distance = math.hypot(frame_lm['mouth_right'][0] - frame_lm['mouth_left'][0], frame_lm['mouth_right'][1] - frame_lm['mouth_left'][1])
        cheek_distance = math.hypot(frame_lm['right_cheek'][0] - frame_lm['left_cheek'][0], frame_lm['right_cheek'][1] - frame_lm['left_cheek'][1])
        if idx == 0 or not previous_angles:
            angle_change_left = 0
            angle_change_right = 0
        else:
            angle_change_left = abs(left_angle - previous_angles['left_angle'])
            angle_change_right = abs(right_angle - previous_angles['right_angle'])
        previous_angles['left_angle'] = left_angle
        previous_angles['right_angle'] = right_angle
        sync_metric = abs(angle_change_left - angle_change_right)
        features.append({'left_angle': left_angle, 'right_angle': right_angle, 'mouth_distance': mouth_distance, 'cheek_distance': cheek_distance, 'angle_change_left': angle_change_left, 'angle_change_right': angle_change_right, 'sync_metric': sync_metric})
    return pd.DataFrame(features)

def extract_cnn_features(frame, cnn_model):
    img = cv2.resize(frame, CNN_MODEL_INPUT_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = xception_preprocess(img)
    img = np.expand_dims(img, axis=0)
    feats = cnn_model.predict(img, verbose=0)
    return feats.flatten()

def extract_combined_features(video_path, cnn_model):
    cap = cv2.VideoCapture(video_path)
    landmarks_list = []
    cnn_feats_list = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            face_lm = results.multi_face_landmarks[0].landmark
            h, w, _ = frame.shape
            frame_dict = {}
            for key, idx in KEY_LANDMARKS.items():
                lm = face_lm[idx]
                frame_dict[key] = (int(lm.x * w), int(lm.y * h))
            landmarks_list.append(frame_dict)
        else:
            landmarks_list.append(None)
        cnn_feat = extract_cnn_features(frame, cnn_model)
        cnn_feats_list.append(cnn_feat)
    cap.release()
    handcrafted_df = extract_features(landmarks_list)
    cnn_feats_df = pd.DataFrame(cnn_feats_list, columns=[f'cnn_feat_{i}' for i in range(len(cnn_feats_list[0]))])
    combined_df = pd.concat([handcrafted_df.reset_index(drop=True), cnn_feats_df.reset_index(drop=True)], axis=1)
    combined_df = combined_df.interpolate(method='linear', limit_direction='forward').fillna(0)
    return combined_df

def create_sequences(features_df, label, sequence_length=SEQUENCE_LENGTH):
    sequences = []
    labels = []
    num_sequences = len(features_df) // sequence_length
    for i in range(num_sequences):
        seq = features_df.iloc[i*sequence_length:(i+1)*sequence_length].values
        sequences.append(seq)
        labels.append(label)
    return sequences, labels

def process_and_save_video(video_path, label, cnn_model, output_dir):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    X_file = os.path.join(output_dir, f"{base_name}_X.npy")
    y_file = os.path.join(output_dir, f"{base_name}_y.npy")
    if os.path.exists(X_file) and os.path.exists(y_file):
        return
    combined_df = extract_combined_features(video_path, cnn_model)
    sequences, labels = create_sequences(combined_df, label)
    X_video = np.array(sequences)
    y_video = np.array(labels)
    np.save(X_file, X_video)
    np.save(y_file, y_video)

def process_folder_separately(folder_path, label, cnn_model, output_dir):
    video_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi', '.mov'))]
    for vf in tqdm(video_files, desc=f"processing {'real' if label==1 else 'fake'}"):
        video_path = os.path.join(folder_path, vf)
        process_and_save_video(video_path, label, cnn_model, output_dir)

if __name__ == '__main__':
    cnn_base = Xception(weights='imagenet', include_top=False, pooling='avg')
    cnn_base.trainable = False
    process_folder_separately(FF_REAL_DIR, 1, cnn_base, PREPROCESSED_DIR)
    process_folder_separately(FF_FAKE_DIR, 0, cnn_base, PREPROCESSED_DIR)
    process_folder_separately(TEST_DIR, TEST_SPLIT, cnn_base, PREPROCESSED_DIR)

!pip install mediapipe opencv-python-headless numpy pandas tqdm tensorflow keras keras_tuner


from google.colab import drive
drive.mount('/content/drive')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc)
from sklearn.utils import class_weight
import keras_tuner as kt  # For hyperparameter tuning


BASE_PATH = '/content/drive/MyDrive'
# Preprocessed data files (after an 80/20 split) are stored in:
PROCESSED_DATA_DIR = os.path.join(BASE_PATH, 'synch_preprocessed')

SEQUENCE_LENGTH = 30  # Must match pre-processing
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.5

# For the 3D CNN, we assume the feature vector per frame is of length 2055.
# We'll reshape it to an image of shape (15, 137, 1) because 15*137 = 2055.
FEATURE_IMG_SHAPE = (15, 137, 1)

#Loading Pre-Processed Data
print("Loading pre-processed training and test data from Google Drive...")
X_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_train.npy'))
y_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'))
X_test = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_test.npy'))
y_test = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_test.npy'))
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

#Visualization and Evaluation Functions
def plot_metrics(history, model_name):
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def plot_confusion_matrix_custom(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def plot_roc_curve_custom(y_true, y_pred_prob, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {model_name}')
    plt.legend(loc="lower right")
    plt.show()

#Handling Class Imbalance
print("\nComputing class weights to handle class imbalance...")
class_weights_values = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights_values))
print(f"Class Weights: {class_weights_dict}")

#Hyperparameter Tuning for Transformer Model
def build_transformer_model(hp):
    """
    Hypermodel builder function for the Transformer-based model.
    """
    inputs = layers.Input(shape=(SEQUENCE_LENGTH, X_train.shape[2]))

    # Tune hyperparameters:
    dense_units = hp.Int('dense_units', min_value=64, max_value=256, step=32)
    dropout_rate = hp.Float('dropout_rate', min_value=0.3, max_value=0.7, step=0.1)
    num_transformer_blocks = hp.Int('num_transformer_blocks', min_value=3, max_value=7, step=1)
    num_heads = hp.Int('num_heads', min_value=4, max_value=8, step=2)
    ff_dim = hp.Int('ff_dim', min_value=128, max_value=512, step=64)

    # Project input features
    x = layers.Dense(dense_units, activation='relu')(inputs)
    # Create positional embeddings with output dim equal to dense_units
    position_indices = tf.range(start=0, limit=SEQUENCE_LENGTH, delta=1)
    position_embeddings = layers.Embedding(input_dim=SEQUENCE_LENGTH, output_dim=dense_units)(position_indices)
    x = x + position_embeddings

    # Transformer blocks
    for _ in range(num_transformer_blocks):
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=dense_units)(x, x)
        attention_output = layers.Dropout(dropout_rate)(attention_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attention_output)
        ff_output = layers.Dense(ff_dim, activation='relu')(x)
        ff_output = layers.Dense(dense_units)(ff_output)
        ff_output = layers.Dropout(dropout_rate)(ff_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ff_output)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model

print("\nStarting hyperparameter search for the Transformer-Based Model...")
tuner = kt.RandomSearch(
    build_transformer_model,
    objective='val_loss',
    max_trials=5,
    executions_per_trial=1,
    directory='kt_transformer_dir',
    project_name='transformer_tuning'
)

tuner.search(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=BATCH_SIZE, verbose=1)
best_hp = tuner.get_best_hyperparameters()[0]
print("\nBest Hyperparameters for Transformer-Based Model:")
print(best_hp.values)

# Building Final Models
print("\nBuilding final Transformer-Based Model with best hyperparameters...")
transformer_model = build_transformer_model(best_hp)
history_transformer = transformer_model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
               ModelCheckpoint(os.path.join(PROCESSED_DATA_DIR, 'best_transformer_model.h5'), monitor='val_loss', save_best_only=True, verbose=1),
               ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)],
    class_weight=class_weights_dict,
    verbose=1
)

# For the 3D CNN, we use the same architecture as before.
def get_3dcnn_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv3D(64, kernel_size=(3,3,3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D(pool_size=(2,2,2))(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Conv3D(128, kernel_size=(3,3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D(pool_size=(2,2,2))(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Conv3D(256, kernel_size=(3,3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D(pool_size=(2,2,2))(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs, outputs)
    return model

print("\nBuilding and training 3D CNN Model...")
# Reshape X_train from (num_sequences, 30, 2055) to (num_sequences, 30, 15, 137, 1)
X_train_3dcnn = X_train.reshape(-1, SEQUENCE_LENGTH, FEATURE_IMG_SHAPE[0], FEATURE_IMG_SHAPE[1], FEATURE_IMG_SHAPE[2])
X_test_3dcnn = X_test.reshape(-1, SEQUENCE_LENGTH, FEATURE_IMG_SHAPE[0], FEATURE_IMG_SHAPE[1], FEATURE_IMG_SHAPE[2])

cnn_3d_model = get_3dcnn_model(input_shape=(SEQUENCE_LENGTH, FEATURE_IMG_SHAPE[0], FEATURE_IMG_SHAPE[1], FEATURE_IMG_SHAPE[2]))
cnn_3d_model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                     loss='binary_crossentropy',
                     metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

history_3dcnn = cnn_3d_model.fit(
    X_train_3dcnn, y_train,
    validation_data=(X_test_3dcnn, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
               ModelCheckpoint(os.path.join(PROCESSED_DATA_DIR, 'best_3dcnn_model.h5'), monitor='val_loss', save_best_only=True, verbose=1),
               ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)],
    class_weight=class_weights_dict,
    verbose=1
)

# Loading Best Models (Weights)
print("\nLoading best Transformer-Based Model weights...")
transformer_model = build_transformer_model(best_hp)
transformer_model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                          loss='binary_crossentropy',
                          metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
transformer_model.load_weights(os.path.join(PROCESSED_DATA_DIR, 'best_transformer_model.h5'))

print("\nLoading best 3D CNN Model weights...")
cnn_3d_model = get_3dcnn_model(input_shape=(SEQUENCE_LENGTH, FEATURE_IMG_SHAPE[0], FEATURE_IMG_SHAPE[1], FEATURE_IMG_SHAPE[2]))
cnn_3d_model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                     loss='binary_crossentropy',
                     metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
cnn_3d_model.load_weights(os.path.join(PROCESSED_DATA_DIR, 'best_3dcnn_model.h5'))

# Ensemble Evaluation on Test Data
print("\nEvaluating Ensemble Model on Test Data...")
y_test_pred_prob_transformer = transformer_model.predict(X_test, verbose=0).ravel()
y_test_pred_prob_3dcnn = cnn_3d_model.predict(X_test_3dcnn, verbose=0).ravel()

# Ensemble: average the predicted probabilities
y_test_pred_prob_ensemble = (y_test_pred_prob_transformer + y_test_pred_prob_3dcnn) / 2
y_test_pred_ensemble = (y_test_pred_prob_ensemble >= 0.5).astype(int)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, classification_report
acc_ens = accuracy_score(y_test, y_test_pred_ensemble)
f1_ens = f1_score(y_test, y_test_pred_ensemble)
prec_ens = precision_score(y_test, y_test_pred_ensemble)
rec_ens = recall_score(y_test, y_test_pred_ensemble)
roc_ens = roc_auc_score(y_test, y_test_pred_prob_ensemble)

print(f"\nEnsemble Model - Accuracy: {acc_ens:.4f}")
print(f"Ensemble Model - F1 Score: {f1_ens:.4f}")
print(f"Ensemble Model - Precision: {prec_ens:.4f}")
print(f"Ensemble Model - Recall: {rec_ens:.4f}")
print(f"Ensemble Model - ROC AUC Score: {roc_ens:.4f}")

print("\nClassification Report for Ensemble Model:")
print(classification_report(y_test, y_test_pred_ensemble, target_names=['Fake', 'Real']))

plot_confusion_matrix_custom(y_test, y_test_pred_ensemble, model_name='Ensemble Model')
plot_roc_curve_custom(y_test, y_test_pred_prob_ensemble, model_name='Ensemble Model')

# Saving Best Hyperparameters
print("\nBest Hyperparameters for the Transformer-Based Model:")
print(best_hp.values)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                             classification_report, confusion_matrix, roc_curve, auc, cohen_kappa_score,
                             matthews_corrcoef, balanced_accuracy_score)
from sklearn.utils import class_weight

#Defining Paths and Parameters
BASE_PATH = '/content/drive/MyDrive'
# Preprocessed data files (after an 80/20 split) are stored here:
PROCESSED_DATA_DIR = os.path.join(BASE_PATH, 'synch_preprocessed')

SEQUENCE_LENGTH = 30  # Must match pre-processing
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.5

# For the 3D CNN: reshape flattened feature vector (length 2055) to an image shape.
# Here we use (15, 137, 1) because 15*137 = 2055.
FEATURE_IMG_SHAPE = (15, 137, 1)

# Loading Pre-Processed Data
print("Loading pre-processed training and test data from Google Drive...")
X_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_train.npy'))
y_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'))
X_test = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_test.npy'))
y_test = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_test.npy'))
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Visualization and Evaluation Functions
def plot_metrics(history, model_name):
    plt.figure(figsize=(14, 5))
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def plot_confusion_matrix_custom(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def plot_roc_curve_custom(y_true, y_pred_prob, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {model_name}')
    plt.legend(loc="lower right")
    plt.show()

# Handling Class Imbalance
print("\nComputing class weights to handle class imbalance...")
class_weights_values = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights_values))
print(f"Class Weights: {class_weights_dict}")

# Defining Advanced Models
def get_transformer_model(input_shape, num_heads=8, ff_dim=256, num_transformer_blocks=6, dropout=0.3):
    inputs = layers.Input(shape=input_shape)
    # Dense projection and positional encoding
    x = layers.Dense(128, activation='relu')(inputs)
    # Create positional embeddings with the same dimension as the projection (128)
    position_indices = tf.range(start=0, limit=input_shape[0], delta=1)
    position_embeddings = layers.Embedding(input_dim=input_shape[0], output_dim=128)(position_indices)
    x = x + position_embeddings
    for _ in range(num_transformer_blocks):
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=128)(x, x)
        attention_output = layers.Dropout(dropout)(attention_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attention_output)
        ff_output = layers.Dense(ff_dim, activation='relu')(x)
        ff_output = layers.Dense(128)(ff_output)
        ff_output = layers.Dropout(dropout)(ff_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ff_output)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs, outputs)
    return model

def get_3dcnn_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv3D(64, kernel_size=(3,3,3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D(pool_size=(2,2,2))(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Conv3D(128, kernel_size=(3,3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D(pool_size=(2,2,2))(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Conv3D(256, kernel_size=(3,3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D(pool_size=(2,2,2))(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs, outputs)
    return model

# Initializing Models
print("\nInitializing the Transformer-Based Model...")
transformer_model = get_transformer_model(input_shape=(SEQUENCE_LENGTH, X_train.shape[2]))
transformer_model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                          loss='binary_crossentropy',
                          metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

print("\nInitializing the 3D CNN Model...")
# Reshape X_train from (num_sequences, 30, 2055) to (num_sequences, 30, 15, 137, 1)
X_train_3dcnn = X_train.reshape(-1, SEQUENCE_LENGTH, FEATURE_IMG_SHAPE[0], FEATURE_IMG_SHAPE[1], FEATURE_IMG_SHAPE[2])
X_test_3dcnn = X_test.reshape(-1, SEQUENCE_LENGTH, FEATURE_IMG_SHAPE[0], FEATURE_IMG_SHAPE[1], FEATURE_IMG_SHAPE[2])
cnn_3d_model = get_3dcnn_model(input_shape=(SEQUENCE_LENGTH, FEATURE_IMG_SHAPE[0], FEATURE_IMG_SHAPE[1], FEATURE_IMG_SHAPE[2]))
cnn_3d_model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                     loss='binary_crossentropy',
                     metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

# Defining Callbacks
transformer_ckpt_path = os.path.join(PROCESSED_DATA_DIR, 'best_transformer_model.h5')
cnn3d_ckpt_path = os.path.join(PROCESSED_DATA_DIR, 'best_3dcnn_model.h5')

early_stopping_transformer = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
checkpoint_transformer = ModelCheckpoint(transformer_ckpt_path, monitor='val_loss', save_best_only=True, verbose=1)
reduce_lr_transformer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

early_stopping_3dcnn = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
checkpoint_cnn3d = ModelCheckpoint(cnn3d_ckpt_path, monitor='val_loss', save_best_only=True, verbose=1)
reduce_lr_3dcnn = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

# Training Models
print("\nTraining the Transformer-Based Model...")
history_transformer = transformer_model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping_transformer, checkpoint_transformer, reduce_lr_transformer],
    class_weight=class_weights_dict,
    verbose=1
)

print("\nTraining the 3D CNN Model...")
history_3dcnn = cnn_3d_model.fit(
    X_train_3dcnn, y_train,
    validation_data=(X_test_3dcnn, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping_3dcnn, checkpoint_cnn3d, reduce_lr_3dcnn],
    class_weight=class_weights_dict,
    verbose=1
)

# Loading Best Models (Weights)
print("\nLoading best Transformer-Based Model weights...")
transformer_model = get_transformer_model(input_shape=(SEQUENCE_LENGTH, X_train.shape[2]))
transformer_model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                          loss='binary_crossentropy',
                          metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
transformer_model.load_weights(transformer_ckpt_path)

print("\nLoading best 3D CNN Model weights...")
cnn_3d_model = get_3dcnn_model(input_shape=(SEQUENCE_LENGTH, FEATURE_IMG_SHAPE[0], FEATURE_IMG_SHAPE[1], FEATURE_IMG_SHAPE[2]))
cnn_3d_model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                     loss='binary_crossentropy',
                     metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
cnn_3d_model.load_weights(cnn3d_ckpt_path)

# Ensemble Evaluation on Test Data
print("\nEvaluating Ensemble Model on Test Data...")
y_test_pred_prob_transformer = transformer_model.predict(X_test, verbose=0).ravel()
y_test_pred_prob_3dcnn = cnn_3d_model.predict(X_test_3dcnn, verbose=0).ravel()

# Ensemble: average the predicted probabilities
y_test_pred_prob_ensemble = (y_test_pred_prob_transformer + y_test_pred_prob_3dcnn) / 2
y_test_pred_ensemble = (y_test_pred_prob_ensemble >= 0.5).astype(int)

acc_ens = accuracy_score(y_test, y_test_pred_ensemble)
f1_ens = f1_score(y_test, y_test_pred_ensemble)
prec_ens = precision_score(y_test, y_test_pred_ensemble)
rec_ens = recall_score(y_test, y_test_pred_ensemble)
roc_ens = roc_auc_score(y_test, y_test_pred_prob_ensemble)
kappa = tf.keras.metrics.CategoricalAccuracy()  # Not ideal for binary; instead use:
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef, balanced_accuracy_score
kappa = cohen_kappa_score(y_test, y_test_pred_ensemble)
mcc = matthews_corrcoef(y_test, y_test_pred_ensemble)
balanced_acc = balanced_accuracy_score(y_test, y_test_pred_ensemble)

print(f"\nEnsemble Model - Accuracy: {acc_ens:.4f}")
print(f"Ensemble Model - F1 Score: {f1_ens:.4f}")
print(f"Ensemble Model - Precision: {prec_ens:.4f}")
print(f"Ensemble Model - Recall: {rec_ens:.4f}")
print(f"Ensemble Model - ROC AUC Score: {roc_ens:.4f}")
print(f"Ensemble Model - Cohen's Kappa: {kappa:.4f}")
print(f"Ensemble Model - Matthews Correlation Coefficient: {mcc:.4f}")
print(f"Ensemble Model - Balanced Accuracy: {balanced_acc:.4f}")

print("\nClassification Report for Ensemble Model:")
print(classification_report(y_test, y_test_pred_ensemble, target_names=['Fake', 'Real']))

plot_confusion_matrix_custom(y_test, y_test_pred_ensemble, model_name='Ensemble Model')
plot_roc_curve_custom(y_test, y_test_pred_prob_ensemble, model_name='Ensemble Model')

# Saving Best Hyperparameters
print("\nBest Hyperparameters for the Transformer-Based Model:")
print(best_hp.values)