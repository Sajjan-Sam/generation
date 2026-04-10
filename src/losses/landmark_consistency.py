from pathlib import Path
import numpy as np
import torch

FEATURE_NAMES = ['eye_to_nose_ratio', 'mouth_width_ratio', 'nose_to_mouth_ratio', 'left_brow_to_eye_ratio', 'right_brow_to_eye_ratio', 'eye_symmetry', 'brow_symmetry', 'jaw_symmetry', 'face_hw_ratio']

def features_from_landmarks_numpy(lm):
    lm = np.asarray(lm, dtype=np.float32)

    left_eye = lm[36:42].mean(axis=0)
    right_eye = lm[42:48].mean(axis=0)
    nose = lm[30]
    mouth_l = lm[48]
    mouth_r = lm[54]
    mouth_center = (mouth_l + mouth_r) / 2.0
    jaw_l = lm[0]
    jaw_r = lm[16]
    chin = lm[8]
    brow_l = lm[17:22].mean(axis=0)
    brow_r = lm[22:27].mean(axis=0)
    eye_center = (left_eye + right_eye) / 2.0

    inter_ocular = np.linalg.norm(left_eye - right_eye) + 1e-8
    jaw_width = np.linalg.norm(jaw_l - jaw_r) + 1e-8
    face_height = np.linalg.norm(eye_center - chin) + 1e-8

    eye_to_nose_ratio = ((np.linalg.norm(left_eye - nose) + np.linalg.norm(right_eye - nose)) / 2.0) / inter_ocular
    mouth_width_ratio = np.linalg.norm(mouth_l - mouth_r) / inter_ocular
    nose_to_mouth_ratio = np.linalg.norm(nose - mouth_center) / inter_ocular
    left_brow_to_eye_ratio = np.linalg.norm(brow_l - left_eye) / inter_ocular
    right_brow_to_eye_ratio = np.linalg.norm(brow_r - right_eye) / inter_ocular
    eye_symmetry = abs(np.linalg.norm(left_eye - nose) - np.linalg.norm(right_eye - nose)) / inter_ocular
    brow_symmetry = abs(np.linalg.norm(brow_l - left_eye) - np.linalg.norm(brow_r - right_eye)) / inter_ocular
    jaw_symmetry = abs(np.linalg.norm(jaw_l - nose) - np.linalg.norm(jaw_r - nose)) / jaw_width
    face_hw_ratio = face_height / jaw_width

    return np.array([
        eye_to_nose_ratio,
        mouth_width_ratio,
        nose_to_mouth_ratio,
        left_brow_to_eye_ratio,
        right_brow_to_eye_ratio,
        eye_symmetry,
        brow_symmetry,
        jaw_symmetry,
        face_hw_ratio,
    ], dtype=np.float32)

def load_real_landmark_stats(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    return {
        "feature_names": list(d["feature_names"]),
        "mean": d["mean"].astype(np.float32),
        "std": d["std"].astype(np.float32),
        "cov": d["cov"].astype(np.float32),
        "precision": d["precision"].astype(np.float32),
    }

def mahalanobis_landmark_loss_torch(fake_features, real_mean, real_precision):
    """
    fake_features: (B, F) torch tensor
    real_mean: (F,) torch tensor
    real_precision: (F, F) torch tensor
    """
    diffs = fake_features - real_mean.unsqueeze(0)
    d2 = torch.einsum("bf,fg,bg->b", diffs, real_precision, diffs)
    return d2.mean()

def zscore_landmark_loss_torch(fake_features, real_mean, real_std):
    z = (fake_features - real_mean.unsqueeze(0)) / (real_std.unsqueeze(0) + 1e-8)
    return (z ** 2).mean()
