# data_utils.py  — BAT-Mamba / PartialSpoof v1.2 edition
#
# Key facts about PartialSpoof v1.2:
#   - Audio: .wav files under {split}/con_wav/
#   - Audio list: {split}/{split}.lst
#   - CM protocol: protocols/PartialSpoof_LA_cm_protocols/PartialSpoof.LA.cm.{split}.trl.txt
#     format: SPEAKER_ID  AUDIO_NAME  -  SYSTEM_ID  KEY
#   - Segment labels: segment_labels/{split}_seglab_0.02.npy
#     format: dict { audio_name: np.array(['1','0',...]) }  '1'=bonafide, '0'=spoof
#   - Frame alignment: SR=16000, XLSR stride=320, cut=66800 -> 208 frames @ 20ms each

import numpy as np
import torch
from torch import Tensor
import soundfile as sf
import librosa
from torch.utils.data import Dataset
from RawBoost import process_Rawboost_feature
from utils import pad
import os
import io

try:
    import torchaudio
    _TORCHAUDIO_AVAILABLE = True
except ImportError:
    _TORCHAUDIO_AVAILABLE = False


def _load_audio(path, sr=16000):
    """Fast audio loader: tries soundfile first (10x faster), falls back to librosa."""
    try:
        X, sr_native = sf.read(path, dtype='float32', always_2d=False)
        if sr_native != sr:
            X = librosa.resample(X, orig_sr=sr_native, target_sr=sr)
    except Exception:
        X, _ = librosa.load(path, sr=sr)
    return X


def _robustness_augment(X, sr=16000):
    """Simulate real-media processing distortions found in RADAR Challenge.
    Randomly applies a subset of: MP3 compression, resampling, additive noise.
    Ref: RADAR Challenge 2026 — test data undergoes unknown combination of transforms.
    Each augmentation is applied with 40% probability independently.
    """
    import random
    # 1. Additive white/pink noise (SNR 15-35 dB)
    if random.random() < 0.4:
        snr_db = random.uniform(15, 35)
        sig_power = np.mean(X ** 2) + 1e-9
        noise_power = sig_power / (10 ** (snr_db / 10))
        noise = np.random.randn(len(X)).astype(np.float32) * np.sqrt(noise_power)
        X = X + noise
    # 2. Resample to random intermediate rate then back (simulates codec/upload)
    if random.random() < 0.4:
        target_sr = random.choice([8000, 22050, 44100])
        X = librosa.resample(X, orig_sr=sr, target_sr=target_sr)
        X = librosa.resample(X, orig_sr=target_sr, target_sr=sr)
    # 3. MP3 compression via torchaudio (if available)
    if _TORCHAUDIO_AVAILABLE and random.random() < 0.4:
        try:
            bitrate = random.choice(['32k', '64k', '128k'])
            waveform = torch.from_numpy(X).unsqueeze(0)  # [1, T]
            buf = io.BytesIO()
            torchaudio.save(buf, waveform, sr, format='mp3',
                            compression=torchaudio.io.CodecConfig(bit_rate=int(bitrate[:-1]) * 1000))
            buf.seek(0)
            orig_len = len(X)
            waveform_mp3, _ = torchaudio.load(buf, format='mp3')
            X = waveform_mp3.squeeze(0).numpy()
            # 严格对齐原始长度，多退少补
            if len(X) > orig_len:
                X = X[:orig_len]
            elif len(X) < orig_len:
                X = np.pad(X, (0, orig_len - len(X)), 'constant')
        except Exception:
            pass  # Skip MP3 aug if torchaudio mp3 not supported
    # Clip to [-1, 1] after augmentation
    X = np.clip(X, -1.0, 1.0)
    return X.astype(np.float32)

# ============================================================
# Constants
# ============================================================
SR         = 16000
CUT        = 66800           # samples (~4.175 s)
STRIDE     = 320             # XLSR stride in samples
NUM_FRAMES = CUT // STRIDE   # = 208
FRAME_DUR  = STRIDE / SR     # = 0.02 s  (matches seglab_0.02.npy)


# ============================================================
# Load PartialSpoof .npy segment labels
# Returns: { audio_name: np.array of '0'/'1' strings }
#   '1' = bonafide,  '0' = spoof  (PartialSpoof convention)
# ============================================================
def load_seglab(npy_path):
    """Load a PartialSpoof segment-label .npy file.
    Returns dict: { audio_name (str): np.array(['1','0',...]) }
    """
    return np.load(npy_path, allow_pickle=True).item()


# ============================================================
# Convert seglab array -> frame_labels + boundary_labels tensors
# seglab values: '1'=bonafide->0, '0'=spoof->1  (flip for our convention)
# ============================================================
def seglab_to_frame_labels(seglab_arr, num_frames=NUM_FRAMES):
    """
    Args:
        seglab_arr: np.array of strings ['1','0',...] from .npy file
        num_frames: int, target number of frames (default 208)
    Returns:
        frame_labels    : [num_frames] torch.long   0=bonafide, 1=spoof
        boundary_labels : [num_frames] torch.float32  1 at transition frames
    """
    # Truncate or pad seglab to num_frames
    arr = np.array(seglab_arr, dtype=np.int32)  # '1'->1 (bonafide), '0'->0 (spoof)
    # Flip: PartialSpoof '1'=bonafide->our 0, '0'=spoof->our 1
    arr = 1 - arr  # now: 0=bonafide, 1=spoof

    if len(arr) >= num_frames:
        arr = arr[:num_frames]
    else:
        # Pad with last label value (or 0 if empty)
        pad_val = int(arr[-1]) if len(arr) > 0 else 0
        arr = np.concatenate([arr, np.full(num_frames - len(arr), pad_val, dtype=np.int32)])

    frame_labels = torch.from_numpy(arr).long()   # [T]

    # Boundary: transition frames (diff != 0), mark the LATER frame
    diff = torch.abs(frame_labels[1:].float() - frame_labels[:-1].float())  # [T-1]
    boundary_labels = torch.zeros(num_frames, dtype=torch.float32)
    boundary_labels[1:] = diff

    return frame_labels, boundary_labels


# ============================================================
# Parse PartialSpoof CM protocol file
# Format: SPEAKER_ID  AUDIO_NAME  -  SYSTEM_ID  KEY
# Returns: (list_IDs, utt_label_dict)
# ============================================================
def parse_ps_protocol(protocol_file, is_eval=False):
    """
    Parse PartialSpoof CM protocol.
    Returns:
        list_IDs : list of audio names
        utt_labels: dict { audio_name: 0(bonafide)/1(spoof) }  (None if is_eval)
    """
    list_IDs = []
    utt_labels = {}
    with open(protocol_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            audio_name = parts[1]   # 2nd column: AUDIO_FILE_NAME
            list_IDs.append(audio_name)
            if not is_eval and len(parts) >= 5:
                key = parts[4]      # 5th column: KEY (bonafide/spoof)
                utt_labels[audio_name] = 0 if key == 'bonafide' else 1
    if is_eval:
        return list_IDs, None
    return list_IDs, utt_labels


# ============================================================
# PartialSpoof Training Dataset
# Returns: (waveform [66800], frame_labels [208] long, boundary_labels [208] float32)
# ============================================================
class Dataset_PartialSpoof_train(Dataset):
    """
    Args:
        list_IDs   : list of audio names (from parse_ps_protocol or .lst file)
        seglab     : dict from load_seglab(), { name: array }
        utt_labels : dict { audio_name: 0(bonafide)/1(spoof) } from parse_ps_protocol
                     Used to supervise the sentence-level loc head with a dedicated loss.
        base_dir   : path to {split}/con_wav/
        args       : argparse namespace (RawBoost params)
        algo       : RawBoost algorithm index
    """
    def __init__(self, list_IDs, seglab, utt_labels, base_dir, args, algo):
        self.list_IDs   = list_IDs
        self.seglab     = seglab
        self.utt_labels = utt_labels  # { audio_name: 0=bonafide / 1=spoof }
        self.base_dir   = base_dir
        self.args       = args
        self.algo       = algo
        self.cut        = CUT

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]

        # Load audio (.wav)
        audio_path = os.path.join(self.base_dir, utt_id + '.wav')
        X = _load_audio(audio_path, sr=SR)

        # RawBoost augmentation
        Y = process_Rawboost_feature(X, SR, self.args, self.algo)
        # Robustness augmentation: simulate MP3/resample/noise (RADAR Challenge conditions)
        Y = _robustness_augment(Y, sr=SR)

        # Pad/truncate to exactly CUT samples
        Y_pad = pad(Y, self.cut)       # [66800]
        waveform = Tensor(Y_pad)       # [66800]

        # Frame-level labels from .npy seglab
        # Use bonafide fallback if utt_id not in seglab
        raw_lab = self.seglab.get(utt_id, np.ones(NUM_FRAMES, dtype=str))
        frame_labels, boundary_labels = seglab_to_frame_labels(
            raw_lab, num_frames=NUM_FRAMES
        )

        # Sentence-level label from CM protocol KEY column (0=bonafide, 1=spoof)
        utt_label = torch.tensor(self.utt_labels.get(utt_id, 0), dtype=torch.long)

        return waveform, frame_labels, boundary_labels, utt_label
        # shapes: [66800]  [208] long  [208] float32  scalar long


# ============================================================
# PartialSpoof Eval Dataset
# Returns: (waveform [66800], frame_labels [208], boundary_labels [208], utt_id)
# ============================================================
class Dataset_PartialSpoof_eval(Dataset):
    def __init__(self, list_IDs, seglab, base_dir):
        self.list_IDs = list_IDs
        self.seglab   = seglab
        self.base_dir = base_dir
        self.cut      = CUT

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        audio_path = os.path.join(self.base_dir, utt_id + '.wav')
        X = _load_audio(audio_path, sr=SR)
        X_pad = pad(X, self.cut)
        waveform = Tensor(X_pad)  # [66800]

        raw_lab = self.seglab.get(utt_id, np.ones(NUM_FRAMES, dtype=str))
        frame_labels, boundary_labels = seglab_to_frame_labels(
            raw_lab, num_frames=NUM_FRAMES
        )
        return waveform, frame_labels, boundary_labels, utt_id


# ============================================================
# Backward-compat: original ASVspoof2019 sentence-level Datasets
# ============================================================
def genSpoof_list(dir_meta, is_train=False, is_eval=False):
    d_meta = {}
    file_list = []
    with open(dir_meta, 'r') as f:
        l_meta = f.readlines()
    if is_train:
        for line in l_meta:
            _, key, _, _, label = line.strip().split()
            file_list.append(key)
            d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta, file_list
    elif is_eval:
        for line in l_meta:
            key = line.strip()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
            _, key, _, _, label = line.strip().split()
            file_list.append(key)
            d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta, file_list


class Dataset_train(Dataset):
    """Original sentence-level dataset (ASVspoof2019 LA)."""
    def __init__(self, args, list_IDs, labels, base_dir, algo):
        self.list_IDs = list_IDs
        self.labels   = labels
        self.base_dir = base_dir
        self.algo     = algo
        self.args     = args
        self.cut      = CUT

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        X, fs = librosa.load(
            os.path.join(self.base_dir, 'flac', utt_id + '.flac'), sr=SR)
        Y = process_Rawboost_feature(X, fs, self.args, self.algo)
        X_pad = pad(Y, self.cut)
        x_inp = Tensor(X_pad)
        target = self.labels[utt_id]
        return x_inp, target


class Dataset_eval(Dataset):
    """Original sentence-level eval dataset (ASVspoof2019 LA)."""
    def __init__(self, list_IDs, base_dir, track):
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut      = CUT
        self.track    = track

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        X, _ = librosa.load(
            os.path.join(self.base_dir, 'flac', utt_id + '.flac'), sr=SR)
        X_pad = pad(X, self.cut)
        return Tensor(X_pad), utt_id


class Dataset_in_the_wild_eval(Dataset):
    def __init__(self, list_IDs, base_dir):
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut      = 64600

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        X, _ = librosa.load(self.base_dir + utt_id, sr=SR)
        X_pad = pad(X, self.cut)
        return Tensor(X_pad), utt_id
