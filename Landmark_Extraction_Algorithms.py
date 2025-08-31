import os
import cv2
import numpy as np
import pandas as pd
import math
import mediapipe as mp
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import tensorflow as tf
import random
from collections import deque
from scipy.ndimage import median_filter
from scipy.stats import rankdata
import traceback

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

DESIRED_FPS = 30
ALIGNED_SIZE = 256
BLOCK_SIZE = 16
SEARCH_RANGE = 4
TARGET_SEQ_LENGTH = 16
SEQUENCE_LENGTH = 30
CNN_MODEL_INPUT_SIZE = (224, 224)
INITIAL_CALIBRATION_FRAMES = 30
DYNAMIC_WINDOW_SIZE = 30
EAR_SENSITIVITY = 0.80
CLOSED_THRESHOLD_FACTOR = 0.70
OPEN_THRESHOLD_FACTOR = 1.0
SMOOTHING_KERNEL_SIZE = 3
FACE_MARGIN = 0.05
AUGMENT_PROB = 0.3
MAX_BRIGHTNESS_SHIFT = 30
MAX_ROTATION_ANGLE = 5
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
dynamic_ear_buffer = deque(maxlen=DYNAMIC_WINDOW_SIZE)

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
    'right_eye_center': 473
}

def detect_and_align_face(frame, aligned_size=ALIGNED_SIZE):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None
    lm = results.multi_face_landmarks[0].landmark
    h, w = frame.shape[:2]
    xs = [int(p.x * w) for p in lm]
    ys = [int(p.y * h) for p in lm]
    x1, y1 = max(min(xs), 0), max(min(ys), 0)
    x2, y2 = min(max(xs), w-1), min(max(ys), h-1)
    mw, mh = x2 - x1, y2 - y1
    m = 0.1
    x1 = max(0, x1 - int(mw * m))
    y1 = max(0, y1 - int(mh * m))
    x2 = min(w-1, x2 + int(mw * m))
    y2 = min(h-1, y2 + int(mh * m))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    le = (int(lm[33].x * w), int(lm[33].y * h))
    re = (int(lm[362].x * w), int(lm[362].y * h))
    dx, dy = re[0] - le[0], re[1] - le[1]
    angle = np.degrees(np.arctan2(dy, dx))
    dist = np.hypot(dx, dy)
    if dist < 1e-5:
        return None
    scale = (aligned_size * 0.4) / dist
    center = ((le[0] + re[0]) // 2, (le[1] + re[1]) // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0,2] += aligned_size/2 - center[0]
    M[1,2] += aligned_size*0.35 - center[1]
    aligned = cv2.warpAffine(frame, M, (aligned_size, aligned_size), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return aligned

def random_augment_frame(frame):
    if random.random() < 0.5:
        shift = random.randint(-MAX_BRIGHTNESS_SHIFT, MAX_BRIGHTNESS_SHIFT)
        frame = cv2.convertScaleAbs(frame, alpha=1, beta=shift)
    ang = random.uniform(-MAX_ROTATION_ANGLE, MAX_ROTATION_ANGLE)
    h, w = frame.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), ang, 1.0)
    return cv2.warpAffine(frame, M, (w, h))

def calculate_ear(lm, idxs):
    try:
        pts = [(lm[i].x, lm[i].y) for i in idxs]
    except:
        return 0.0
    v1 = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    v2 = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    h = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    if h < 1e-6:
        return 0.0
    return (v1 + v2) / (2 * h)

def update_threshold(value, thresh):
    dynamic_ear_buffer.append(value)
    if len(dynamic_ear_buffer) >= DYNAMIC_WINDOW_SIZE//2:
        base = np.median(dynamic_ear_buffer)
        new_t = base * EAR_SENSITIVITY
        return 0.9 * thresh + 0.1 * new_t
    return thresh

def smooth_ear_sequence(seq, kernel_size=SMOOTHING_KERNEL_SIZE):
    return median_filter(seq, size=kernel_size)

def get_face_roi(frame, lm):
    h, w = frame.shape[:2]
    xs = [p.x for p in lm]
    ys = [p.y for p in lm]
    min_x, max_x = max(min(xs), 0), min(max(xs), 1)
    min_y, max_y = max(min(ys), 0), min(max(ys), 1)
    mx = (max_x - min_x) * FACE_MARGIN
    my = (max_y - min_y) * FACE_MARGIN
    min_x, max_x = max(min_x - mx,0), min(max_x + mx,1)
    min_y, max_y = max(min_y - my,0), min(max_y + my,1)
    return frame[int(min_y*h):int(max_y*h), int(min_x*w):int(max_x*w)]

def extract_frames(video_path, desired_fps=DESIRED_FPS):
    cap = cv2.VideoCapture(video_path)
    orig = cap.get(cv2.CAP_PROP_FPS) or desired_fps
    skip = max(int(orig/desired_fps),1)
    frames, idx = [], 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % skip == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames

def block_based_motion_estimation(prev_frame, curr_frame, block_size=BLOCK_SIZE, search_range=SEARCH_RANGE):
    h, w = prev_frame.shape
    vecs = []
    for y in range(0, h - block_size + 1, block_size):
        for x in range(0, w - block_size + 1, block_size):
            best, dx, dy = float('inf'), 0, 0
            blk = prev_frame[y:y+block_size, x:x+block_size]
            for dy_ in range(-search_range, search_range+1):
                for dx_ in range(-search_range, search_range+1):
                    rx, ry = x+dx_, y+dy_
                    if 0 <= rx < w-block_size and 0 <= ry < h-block_size:
                        cand = curr_frame[ry:ry+block_size, rx:rx+block_size]
                        sad = np.sum(np.abs(blk - cand))
                        if sad < best:
                            best, dx, dy = sad, dx_, dy_
            vecs.append((dx, dy))
    return vecs

def enhance_face(face):
    yuv = cv2.cvtColor(face, cv2.COLOR_BGR2YUV)
    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
    eq = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    return cv2.GaussianBlur(eq, (3,3), 0)

def adjust_feature_dim(seqs, target_dim=12288):
    b, sl, cd = seqs.shape
    if cd == target_dim:
        return seqs
    factor = int(np.ceil(target_dim/cd))
    tiled = np.tile(seqs, (1,1,factor))
    return tiled[:,:,:target_dim]

def run_modelA_pipeline(video_path, output_dir="micro_output"):
    os.makedirs(output_dir, exist_ok=True)
    frames = extract_frames(video_path)
    prev_gray = None
    seqs = []
    for frame in frames:
        aligned = detect_and_align_face(frame)
        if aligned is None:
            continue
        enhanced = enhance_face(aligned)
        curr_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            vecs = block_based_motion_estimation(prev_gray, curr_gray)
            arr = np.array(vecs)
            dx, dy = arr[:,0], arr[:,1]
            feats = [np.mean(dx), np.std(dx), np.mean(dy), np.std(dy), np.mean(np.hypot(dx,dy)), np.std(np.hypot(dx,dy)), np.max(np.hypot(dx,dy))]
            seqs.append(np.array(feats))
        prev_gray = curr_gray
    if len(seqs) > TARGET_SEQ_LENGTH:
        seqs = seqs[:TARGET_SEQ_LENGTH]
    else:
        for _ in range(TARGET_SEQ_LENGTH - len(seqs)):
            seqs.append(np.zeros(7))
    inp = np.expand_dims(np.array(seqs, dtype=np.float32), axis=0)
    return adjust_feature_dim(inp, target_dim=12288)

def run_modelB_pipeline(video_path):
    cap = cv2.VideoCapture(video_path)
    ear_vals, total_blinks = [], 0
    blink_counter, frame_count = 0, 0
    calibration, blink_in_progress = [], False
    thresh, last_ear = 0.21, 0.21
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if random.random() < AUGMENT_PROB:
            frame = random_augment_frame(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            ear_vals.append(last_ear)
            continue
        lm = res.multi_face_landmarks[0].landmark
        left = calculate_ear(lm, LEFT_EYE)
        right = calculate_ear(lm, RIGHT_EYE)
        avg = (left + right)/2.0
        if avg < 1e-3:
            avg = last_ear
        last_ear = avg
        if frame_count <= INITIAL_CALIBRATION_FRAMES:
            calibration.append(avg)
            thresh = np.median(calibration)*EAR_SENSITIVITY
        else:
            if avg > thresh:
                thresh = update_threshold(avg, thresh)
            closed_t = CLOSED_THRESHOLD_FACTOR*thresh
            open_t = OPEN_THRESHOLD_FACTOR*thresh
            if not blink_in_progress and avg < closed_t:
                blink_in_progress, blink_counter = True, 1
            elif blink_in_progress and avg > open_t:
                if blink_counter >= 2:
                    total_blinks += 1
                blink_in_progress, blink_counter = False, 0
            elif blink_in_progress:
                blink_counter += 1
        ear_vals.append(avg)
    cap.release()
    ear_arr = smooth_ear_sequence(np.array(ear_vals))
    return ear_arr, total_blinks, np.array([total_blinks, ear_arr.mean()]).reshape(1,-1)

def extract_landmarks(video_path):
    cap = cv2.VideoCapture(video_path)
    lm_list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        aligned = detect_and_align_face(frame)
        if aligned is None:
            lm_list.append(None)
            continue
        rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        if res.multi_face_landmarks:
            fl = res.multi_face_landmarks[0].landmark
            h, w = aligned.shape[:2]
            d = {}
            for k, idx in {**KEY_LANDMARKS, **ADDITIONAL_LANDMARKS}.items():
                p = fl[idx]
                d[k] = (int(p.x*w), int(p.y*h))
            lm_list.append(d)
        else:
            lm_list.append(None)
    cap.release()
    return lm_list

def calculate_angle(a, b, c):
    ba = (a[0]-b[0], a[1]-b[1])
    bc = (c[0]-b[0], c[1]-b[1])
    dp = ba[0]*bc[0] + ba[1]*bc[1]
    mag = math.sqrt(ba[0]**2+ba[1]**2)*math.sqrt(bc[0]**2+bc[1]**2)
    if mag == 0:
        return 0
    return math.degrees(math.acos(dp/mag))

def extract_features(lm_list):
    feats = []
    # Initialize prev so prev['left_angle'] and prev['right_angle'] always exist:
    prev = {'left_angle': 0.0, 'right_angle': 0.0}

    for idx, fl in enumerate(lm_list):
        if fl is None:
            feats.append({
                'left_angle': 0.0,
                'right_angle': 0.0,
                'mouth_distance': 0.0,
                'cheek_distance': 0.0,
                'angle_change_left': 0.0,
                'angle_change_right': 0.0,
                'sync_metric': 0.0
            })
            continue

        # Compute current angles and distances
        la = calculate_angle(fl['nose_tip'], fl['left_iris'], fl['right_iris'])
        ra = calculate_angle(fl['nose_tip'], fl['right_iris'], fl['left_iris'])
        md = math.hypot(fl['mouth_right'][0] - fl['mouth_left'][0],
                        fl['mouth_right'][1] - fl['mouth_left'][1])
        cd = math.hypot(fl['right_cheek'][0] - fl['left_cheek'][0],
                        fl['right_cheek'][1] - fl['left_cheek'][1])

        # On the very first valid frame, treat angle changes as zero
        if idx == 0:
            acl, acr = 0.0, 0.0
        else:
            acl = abs(la - prev['left_angle'])
            acr = abs(ra - prev['right_angle'])

        # Update prev for next iteration
        prev['left_angle'], prev['right_angle'] = la, ra

        sync = abs(acl - acr)
        feats.append({
            'left_angle': la,
            'right_angle': ra,
            'mouth_distance': md,
            'cheek_distance': cd,
            'angle_change_left': acl,
            'angle_change_right': acr,
            'sync_metric': sync
        })

    return pd.DataFrame(feats)


def extract_cnn_features(frame, cnn_model):
    img = cv2.resize(frame, CNN_MODEL_INPUT_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = tf.keras.applications.xception.preprocess_input(img)
    img = np.expand_dims(img,0)
    return cnn_model.predict(img, verbose=0).flatten()

def extract_combined_features(video_path, cnn_model):
    cap = cv2.VideoCapture(video_path)
    lm_list, cnn_list = [], []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        aligned = detect_and_align_face(frame)
        if aligned is None:
            lm_list.append(None)
            cnn_list.append(np.zeros(2048))
            continue
        rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        if res.multi_face_landmarks:
            fl = res.multi_face_landmarks[0].landmark
            h, w = aligned.shape[:2]
            fd = {}
            for k, idx in KEY_LANDMARKS.items():
                p = fl[idx]
                fd[k] = (int(p.x*w), int(p.y*h))
            lm_list.append(fd)
        else:
            lm_list.append(None)
        cnn_list.append(extract_cnn_features(aligned, cnn_model))
    cap.release()
    handmade = extract_features(lm_list)
    cnn_df = pd.DataFrame(cnn_list, columns=[f'cnn_feat_{i}' for i in range(len(cnn_list[0]))])
    combined = pd.concat([handmade.reset_index(drop=True), cnn_df.reset_index(drop=True)], axis=1)
    combined = combined.interpolate(method='linear', limit_direction='forward').fillna(0)
    return combined

def create_sequences(df, sequence_length=SEQUENCE_LENGTH):
    seqs = []
    n = len(df)//sequence_length
    for i in range(n):
        seqs.append(df.iloc[i*sequence_length:(i+1)*sequence_length].values)
    return np.array(seqs)

def run_modelC_pipeline(video_path):
    cnn = tf.keras.applications.Xception(weights='imagenet', include_top=False, pooling='avg')
    cnn.trainable = False
    combined = extract_combined_features(video_path, cnn)
    seqs = create_sequences(combined)
    seqs = adjust_feature_dim(seqs, target_dim=2055)
    return seqs

def run_modelD_pipeline(video_path):
    # 1) extract frames at DESIRED_FPS
    frames = extract_frames(video_path, desired_fps=DESIRED_FPS)
    prev_edge = None
    variations = []

    # 2) for each frame: align, grayscale, canny, XOR with previous
    for frame in frames:
        aligned = detect_and_align_face(frame, ALIGNED_SIZE)
        if aligned is None:
            edge = np.zeros(CNN_MODEL_INPUT_SIZE, dtype=np.uint8)
        else:
            face = cv2.resize(aligned, CNN_MODEL_INPUT_SIZE)
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            edge = cv2.Canny(gray, 100, 200)

        if prev_edge is not None:
            var_map = cv2.bitwise_xor(prev_edge, edge)
            variations.append(var_map.flatten().astype(np.float32))

        prev_edge = edge

    # 3) truncate or pad to TARGET_SEQ_LENGTH (16)
    if len(variations) > TARGET_SEQ_LENGTH:
        variations = variations[:TARGET_SEQ_LENGTH]
    else:
        # pad with zeros
        pad = np.zeros_like(variations[0]) if variations else np.zeros(CNN_MODEL_INPUT_SIZE[0]*CNN_MODEL_INPUT_SIZE[1], dtype=np.float32)
        variations += [pad] * (TARGET_SEQ_LENGTH - len(variations))

    # 4) form a batch of one, shape (1,16,224*224)
    seq = np.expand_dims(np.stack(variations, axis=0), axis=0)

    # 5) adjust features to 12288 per step
    seq = adjust_feature_dim(seq, target_dim=12288)

    # now seq.shape == (1, 16, 12288)
    return seq


modelA = tf.keras.models.load_model("modelA.keras")
modelB = tf.keras.models.load_model("modelB.keras")
modelC = tf.keras.models.load_model("modelC.keras")
modelD = tf.keras.models.load_model("modelD.keras")

def robust_rank_weighted_ensemble(preds):
    ranks = rankdata(preds, method='ordinal')
    weights = ranks/np.sum(ranks)
    return np.sum(weights*preds)

def integrated_prediction(video_path):
    """Run each pipeline+model in turn, catching and printing any errors."""
    try:
        flow_input_A = run_modelA_pipeline(video_path)
        print("Model A input shape:", flow_input_A.shape)
        predA = modelA.predict(flow_input_A)[0][0].item()
    except Exception as e:
        print("❌ Error in Model A pipeline or prediction:")
        traceback.print_exc()
        return

    try:
        seqD = run_modelD_pipeline(video_path)
        print("Model D input shape:", seqD.shape)
        predD = modelD.predict(seqD)[0][0].item()
    except Exception as e:
        print("❌ Error in Model D pipeline or prediction:")
        traceback.print_exc()
        return

    try:
        ear_seq, blink_count, feature_B = run_modelB_pipeline(video_path)
        print("Model B EAR sequence length:", len(ear_seq), "blink_feature shape:", feature_B.shape)
        ear_input = np.expand_dims(np.expand_dims(ear_seq[:200] if len(ear_seq)>200 else np.pad(ear_seq, (0,200-len(ear_seq))), 0), -1)
        blink_input = np.array([[blink_count]], dtype=np.float32)
        print("Model B ear_input shape:", ear_input.shape, "blink_input shape:", blink_input.shape)
        predB = modelB.predict([ear_input, blink_input])[0][0].item()
    except Exception as e:
        print("❌ Error in Model B pipeline or prediction:")
        traceback.print_exc()
        return

    try:
        seqs_C = run_modelC_pipeline(video_path)
        print("Model C seqs shape:", seqs_C.shape)
        if len(seqs_C) == 0:
            predC = 0.5
        else:
            predC = modelC.predict(seqs_C[0:1])[0][0].item()
    except Exception as e:
        print("❌ Error in Model C pipeline or prediction:")
        traceback.print_exc()
        return

    # Ensemble & final verdict
    preds = [predA, predB, predC, predD]
    avg_pred = robust_rank_weighted_ensemble(preds)
    if avg_pred < 0.5:
        print(f"Final Verdict: Video is Real with {(1-avg_pred)*100:.1f}% confidence")
    else:
        print(f"Final Verdict: Video is Fake with {avg_pred*100:.1f}% confidence")
    return avg_pred, ("Fake" if avg_pred >= 0.5 else "Real")

if __name__ == "__main__":
    video_path = "video.mp4"
    try:
        integrated_prediction(video_path)
    except Exception as e:
        print(f"\n❌ Unhandled error during processing of `{video_path}`:")
        print(f"{type(e).__name__}: {e}")
        traceback.print_exc()
