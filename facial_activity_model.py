

from google.colab import drive
drive.mount('/content/drive')

import os
import glob
import random
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm


#Face Detection with Mediapipe

mp_face_detection = mp.solutions.face_detection

def crop_face_with_mediapipe(frame, face_detector):

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb_frame)
    if not results.detections:
        return None, False

    # Pick the first detection as "largest" (Mediapipe doesn't guarantee ordering)
    detection = results.detections[0]
    bbox = detection.location_data.relative_bounding_box
    h, w, _ = frame.shape

    x_min = max(int(bbox.xmin * w), 0)
    y_min = max(int(bbox.ymin * h), 0)
    box_w = max(int(bbox.width * w), 0)
    box_h = max(int(bbox.height * h), 0)

    x_max = min(x_min + box_w, w)
    y_max = min(y_min + box_h, h)

    face_roi = frame[y_min:y_max, x_min:x_max]
    return face_roi, True


# Contour Variation–Based Muscle Activity (CVMA)

def compute_edge_map(gray_img):

    return cv2.Canny(gray_img, threshold1=100, threshold2=200)

def xor_edge_maps(edge_map1, edge_map2):

    return cv2.bitwise_xor(edge_map1, edge_map2)


#Advanced Preprocessing: Full Video => .npy

def advanced_preprocess_video_cvma(input_path,
                                   output_dir,
                                   face_detector,
                                   face_size=(224, 224)):

    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Could not open video: {input_path}")
        return

    face_frames = []
    edge_maps = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect & crop face
        face_roi, found = crop_face_with_mediapipe(frame, face_detector)
        if not found:
            # Skip frames where no face is detected
            continue

        # Resize face
        face_resized = cv2.resize(face_roi, face_size, interpolation=cv2.INTER_AREA)
        face_frames.append(face_resized)

        # Convert to grayscale => compute edges
        gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        edge_map = compute_edge_map(gray)
        edge_maps.append(edge_map)

    cap.release()

    if len(face_frames) == 0:
        print(f"No faces found in {input_path}. Skipping this video.")
        return

    face_frames_arr = np.stack(face_frames, axis=0)  # shape [T, H, W, 3]
    edge_maps_arr = np.stack(edge_maps, axis=0)      # shape [T, H, W]

    # XOR consecutive edge maps => shape [T-1, H, W]
    variation_list = []
    for i in range(1, len(edge_maps_arr)):
        prev_map = edge_maps_arr[i-1]
        curr_map = edge_maps_arr[i]
        diff_map = xor_edge_maps(prev_map, curr_map)
        variation_list.append(diff_map)

    if len(variation_list) == 0:
        print(f"Only 1 face frame found in {input_path}, can't compute variations.")
        return

    edge_variation_arr = np.stack(variation_list, axis=0)  # shape [T-1, H, W]

    # Save the results
    np.save(os.path.join(output_dir, "face_frames.npy"), face_frames_arr)
    np.save(os.path.join(output_dir, "edge_variation.npy"), edge_variation_arr)

    print(f"Video: {input_path}")
    print(f" -> Cropped frames: {face_frames_arr.shape}")
    print(f" -> Edge Variation: {edge_variation_arr.shape}")
    print(f"Saved to: {output_dir}")


#Train/Test Split & Processing

def collect_videos_and_split(real_dir, fake_dir, train_ratio=0.8):
    """
    Gather .mp4 from real_dir, fake_dir, shuffle, do train/test split.
    Returns (train_items, test_items), each a list of (path, label).
    """
    real_videos = glob.glob(os.path.join(real_dir, "*.mp4"))
    fake_videos = glob.glob(os.path.join(fake_dir, "*.mp4"))

    all_items = [(rv, "real") for rv in real_videos] + [(fv, "fake") for fv in fake_videos]
    random.shuffle(all_items)

    split_idx = int(len(all_items) * train_ratio)
    train_items = all_items[:split_idx]
    test_items  = all_items[split_idx:]
    return train_items, test_items

def process_dataset(items, split_folder, face_detector, out_root):
    """
    For each video in items => (vid_path, label),
    produce face_frames.npy & edge_variation.npy.
    Store them in out_root/split_folder/label/videoName/
    """
    for (vid_path, lbl) in tqdm(items, desc=f"Processing {split_folder}", unit="video"):
        base_name = os.path.splitext(os.path.basename(vid_path))[0]
        out_dir = os.path.join(out_root, split_folder, lbl, base_name)

        advanced_preprocess_video_cvma(
            input_path=vid_path,
            output_dir=out_dir,
            face_detector=face_detector,
            face_size=(224, 224)
        )


#Execute the Entire Pipeline

# Adjust these paths to match your Google Drive structure:
FFPP_ROOT = "/content/drive/MyDrive/FF++"
REAL_DIR = os.path.join(FFPP_ROOT, "real")
FAKE_DIR = os.path.join(FFPP_ROOT, "fake")

OUTPUT_ROOT = "/content/drive/MyDrive/facial_pre_processed"

# Create subdirs for train/test => real/fake
for split in ["train", "test"]:
    for lbl in ["real", "fake"]:
        os.makedirs(os.path.join(OUTPUT_ROOT, split, lbl), exist_ok=True)

# 1) Split
train_items, test_items = collect_videos_and_split(REAL_DIR, FAKE_DIR, train_ratio=0.8)
print(f"Train items: {len(train_items)} | Test items: {len(test_items)}")

# 2) Initialize Mediapipe Face Detection
with mp_face_detection.FaceDetection(
    model_selection=0,  # 0 => short range, 1 => full range
    min_detection_confidence=0.5
) as face_detector:

    # 3) Process train videos
    process_dataset(train_items, "train", face_detector, OUTPUT_ROOT)

    # 4) Process test videos
    process_dataset(test_items, "test", face_detector, OUTPUT_ROOT)

print("All done! Contour Variation-based Facial Muscle Activity (CVMA) pre-processing complete.")
print(f"Check '{OUTPUT_ROOT}' in your Drive for face_frames.npy & edge_variation.npy.")

# --- Install required libraries ---
!pip install mediapipe torch torchvision tqdm scikit-learn opencv-python matplotlib

# --- Mount Google Drive ---
from google.colab import drive
drive.mount('/content/drive')

# --- Import Statements ---
import os
import glob
import random
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score,
    matthews_corrcoef, roc_auc_score, roc_curve
)


# Helper Functions


def fix_length_sequence(arr, fixed_length, pad_last=True):
    T_actual = arr.shape[0]
    if T_actual >= fixed_length:
        start = random.randint(0, T_actual - fixed_length)
        return arr[start:start+fixed_length]
    else:
        pad_amount = fixed_length - T_actual
        pad_array = np.repeat(arr[-1][np.newaxis, ...], pad_amount, axis=0)
        return np.concatenate([arr, pad_array], axis=0)

def filter_valid_folders(folder_list):
    valid_folders = []
    for folder in tqdm(folder_list, desc="Filtering valid folders"):
        face_path = os.path.join(folder, "face_frames.npy")
        edge_path = os.path.join(folder, "edge_variation.npy")
        if not os.path.exists(face_path) or not os.path.exists(edge_path):
            continue
        try:
            face_frames = np.load(face_path)
            edge_variation = np.load(edge_path)
        except Exception:
            continue
        if face_frames.shape[0] == 0 or edge_variation.shape[0] == 0:
            continue
        valid_folders.append(folder)
    return valid_folders


#Custom Dataset for Preprocessed Data


class CVMADataset(Dataset):
    """
    Expects each sample folder to contain:
      - face_frames.npy (shape: [T, H, W, 3])
      - edge_variation.npy (shape: [T-1, H, W])
    'labels' is 0 for real, 1 for fake.
    Each sample is forced to a fixed temporal length.
    """
    def __init__(self, data_folders, labels, fixed_length=64):
        self.data_folders = data_folders
        self.labels = labels
        self.fixed_length = fixed_length

    def __len__(self):
        return len(self.data_folders)

    def __getitem__(self, idx):
        folder = self.data_folders[idx]
        label = self.labels[idx]
        face_frames = np.load(os.path.join(folder, "face_frames.npy"))
        edge_variation = np.load(os.path.join(folder, "edge_variation.npy"))
        face_frames = fix_length_sequence(face_frames, self.fixed_length, pad_last=True)
        edge_variation = fix_length_sequence(edge_variation, self.fixed_length - 1, pad_last=True)
        face_frames = torch.from_numpy(face_frames).float()        # [T, H, W, 3]
        edge_variation = torch.from_numpy(edge_variation).float()  # [T-1, H, W]
        label_tensor = torch.tensor([label], dtype=torch.float32)
        return face_frames, edge_variation, label_tensor


#Model Components


class TemporalGatedAggregator(nn.Module):
    """
    A custom aggregator for sequential data.

    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.Wg = nn.Linear(input_dim, hidden_dim)
        self.bg = nn.Parameter(torch.zeros(hidden_dim))
        self.Wh = nn.Linear(input_dim, hidden_dim)
        self.bh = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x_sequence):
        B, T, _ = x_sequence.shape
        h = torch.zeros(B, self.bh.shape[0], device=x_sequence.device)
        for t in range(T):
            x_t = x_sequence[:, t, :]
            gate_t = torch.sigmoid(self.Wg(x_t) + self.bg)
            cand_t = self.Wh(x_t) + self.bh
            h = (1 - gate_t) * h + gate_t * cand_t
        return h  # [B, hidden_dim]

class CVMAAdvancedModel(nn.Module):
    """
    Advanced model combining face_frames and edge_variation.
    """
    def __init__(self,
                 face_h=224, face_w=224,
                 edge_h=224, edge_w=224,
                 face_reduced_dim=128,
                 edge_reduced_dim=64,
                 aggregator_hidden=64):
        super().__init__()
        # Face branch
        self.face_input_dim = face_h * face_w * 3
        self.face_reducer = nn.Linear(self.face_input_dim, face_reduced_dim)
        self.face_aggregator = TemporalGatedAggregator(face_reduced_dim, aggregator_hidden)
        # Edge branch
        self.edge_input_dim = edge_h * edge_w
        self.edge_reducer = nn.Linear(self.edge_input_dim, edge_reduced_dim)
        self.edge_aggregator = TemporalGatedAggregator(edge_reduced_dim, aggregator_hidden)
        # Fusion & output
        fusion_in = aggregator_hidden * 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, fusion_in),
            nn.ReLU(),
            nn.Linear(fusion_in, 1)
        )

    def forward(self, face_frames, edge_variation):
        B, T, H, W, C = face_frames.shape
        # Face path
        f = face_frames.view(B * T, -1)
        f = self.face_reducer(f).view(B, T, -1)
        fh = self.face_aggregator(f)
        # Edge path
        B_e, Tm, EH, EW = edge_variation.shape
        e = edge_variation.view(B_e * Tm, -1)
        e = self.edge_reducer(e).view(B_e, Tm, -1)
        eh = self.edge_aggregator(e)
        # Fuse & predict
        x = torch.cat((fh, eh), dim=-1)
        logits = self.fusion(x)
        return logits


#Training, Evaluation & Hyperparameter Search


def train_eval_model(model, train_loader, val_loader,
                     epochs=5, lr=1e-4, weight_decay=1e-4, clip_grad=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # --- SGD w/ Nesterov momentum µ=0.9 ---
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        nesterov=True,
        weight_decay=weight_decay
    )

    criterion = nn.BCEWithLogitsLoss()

    # Cosine Annealing over epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=0.0
    )

    train_losses, val_losses = [], []
    val_accs, val_f1s, val_kappas, val_mccs, val_aucs = [], [], [], [], []

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for face_frames, edge_variation, labels in train_loader:
            face_frames = face_frames.to(device)
            edge_variation = edge_variation.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(face_frames, edge_variation)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            running_train_loss += loss.item()

        # Step cosine scheduler once per epoch
        scheduler.step()

        epoch_train_loss = running_train_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        # Validation
        model.eval()
        running_val_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for face_frames, edge_variation, labels in val_loader:
                face_frames = face_frames.to(device)
                edge_variation = edge_variation.to(device)
                labels = labels.to(device)

                logits = model(face_frames, edge_variation)
                loss = criterion(logits, labels)
                running_val_loss += loss.item()

                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        epoch_val_loss = running_val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)

        all_preds = np.concatenate(all_preds).ravel()
        all_labels = np.concatenate(all_labels).ravel()
        all_probs  = np.concatenate(all_probs).ravel()

        val_accs.append(accuracy_score(all_labels, all_preds))
        val_f1s.append(f1_score(all_labels, all_preds))
        val_kappas.append(cohen_kappa_score(all_labels, all_preds))
        val_mccs.append(matthews_corrcoef(all_labels, all_preds))
        try:
            val_aucs.append(roc_auc_score(all_labels, all_probs))
        except:
            val_aucs.append(0.0)

        print(f"Epoch {epoch+1}/{epochs} — "
              f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | "
              f"AUC: {val_aucs[-1]:.4f}")

    # Plot losses
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend(); plt.show()

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f"AUC={val_aucs[-1]:.4f}")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.legend(); plt.show()

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accs": val_accs,
        "val_f1s": val_f1s,
        "val_kappas": val_kappas,
        "val_mccs": val_mccs,
        "val_aucs": val_aucs
    }

# 5) Data Loading & 6) Run Search


OUTPUT_ROOT = "/content/drive/MyDrive/facial_pre_processed"
# … (data loading as before) …

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=4, shuffle=False)

best_params, search_results = hyperparameter_search(train_loader, test_loader, num_epochs=5)
print("Final Best Parameters:", best_params)