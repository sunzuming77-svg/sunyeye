import argparse
import glob
import os
from types import SimpleNamespace

import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from model import Model


def load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    """Load mono audio as float32 at target_sr."""
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = np.mean(x, axis=1)
    if sr != target_sr:
        x = librosa.resample(x, orig_sr=sr, target_sr=target_sr)
    return x.astype(np.float32)


def crop_or_pad_3tta(x: np.ndarray, cut: int = 66800):
    """
    3-crop test-time augmentation.
    - len(x) > cut: head/middle/tail crops
    - len(x) <= cut: zero-pad once
    """
    n = len(x)
    if n > cut:
        head = x[:cut]
        start_mid = max((n - cut) // 2, 0)
        middle = x[start_mid:start_mid + cut]
        tail = x[-cut:]
        return [head, middle, tail]

    if n < cut:
        out = np.zeros(cut, dtype=np.float32)
        out[:n] = x
        return [out]

    return [x]


@torch.no_grad()
def infer_one(model: torch.nn.Module, device: str, wav: np.ndarray, cut: int = 66800):
    crops = crop_or_pad_3tta(wav, cut=cut)
    loc_scores = []
    dia_scores = []

    for c in crops:
        inp = torch.from_numpy(c).unsqueeze(0).to(device)  # [1, T]
        logits_loc, _, logits_dia, _ = model(inp)

        # Left-branch sentence score (spoof logit)
        score_loc = logits_loc[:, 1].item()

        # Right-branch frame aggregation
        if logits_dia.shape[-1] >= 2:
            score_dia = torch.logsumexp(logits_dia[:, :, 1:], dim=-1).mean(dim=1).item()
        else:
            score_dia = logits_dia[:, :, 0].mean(dim=1).item()

        loc_scores.append(score_loc)
        dia_scores.append(score_dia)

    return float(np.mean(loc_scores)), float(np.mean(dia_scores))


def build_model(emb_size: int, num_encoders: int, num_classes: int, device: str):
    args = SimpleNamespace(
        emb_size=emb_size,
        num_encoders=num_encoders,
        num_classes=num_classes,
    )
    model = Model(args, device).to(device)
    return model


def load_ckpt_strict(model: torch.nn.Module, ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    model.load_state_dict(state, strict=True)
    return model


def main():
    parser = argparse.ArgumentParser("RADAR2026 inference (zero-shot)")
    parser.add_argument("--radar_flac_dir", type=str, default=r"C:\RADAR2026-dev\flac")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="raw_scores.txt")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--emb_size", type=int, default=144)
    parser.add_argument("--num_encoders", type=int, default=12)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--cut", type=int, default=66800)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device: {device}")
    print(f"[INFO] loading checkpoint (strict=True): {args.checkpoint}")

    model = build_model(args.emb_size, args.num_encoders, args.num_classes, device)
    model = load_ckpt_strict(model, args.checkpoint, device)
    model.eval()

    flac_files = sorted(glob.glob(os.path.join(args.radar_flac_dir, "*.flac")))
    if not flac_files:
        raise RuntimeError(f"No .flac files found in: {args.radar_flac_dir}")

    print(f"[INFO] found {len(flac_files)} .flac files")

    with open(args.output, "w", encoding="utf-8") as fw:
        for path in tqdm(flac_files, desc="Infer RADAR"):
            utt_id = os.path.splitext(os.path.basename(path))[0]
            wav = load_audio(path, target_sr=args.sr)
            score_loc, score_dia = infer_one(model, device, wav, cut=args.cut)
            fw.write(f"{utt_id} {score_loc:.10f} {score_dia:.10f}\n")

    print(f"[DONE] raw scores saved to: {args.output}")


if __name__ == "__main__":
    main()
