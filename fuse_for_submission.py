import argparse
import os

import numpy as np
from tqdm import tqdm


def read_raw_scores(path: str):
    utt_ids = []
    loc = []
    dia = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 3:
                continue
            utt_ids.append(p[0])
            loc.append(float(p[1]))
            dia.append(float(p[2]))

    if not utt_ids:
        raise RuntimeError(f"No valid lines in raw score file: {path}")

    return utt_ids, np.asarray(loc, dtype=np.float64), np.asarray(dia, dtype=np.float64)


def zscore(x: np.ndarray, eps: float = 1e-8):
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd < eps:
        sd = eps
    return (x - mu) / sd


def write_submission(path: str, utt_ids, scores: np.ndarray):
    # Strict challenge-like format: "utt_id score" (single space, no header)
    with open(path, "w", encoding="utf-8") as f:
        for u, s in tqdm(zip(utt_ids, scores), total=len(utt_ids), desc=f"Write {os.path.basename(path)}"):
            f.write(f"{u} {s:.10f}\n")


def save_pos_neg(base_name: str, out_dir: str, utt_ids, scores: np.ndarray):
    pos_path = os.path.join(out_dir, f"{base_name}_pos.txt")
    neg_path = os.path.join(out_dir, f"{base_name}_neg.txt")
    write_submission(pos_path, utt_ids, scores)
    write_submission(neg_path, utt_ids, -scores)


def main():
    parser = argparse.ArgumentParser("Fuse raw_scores for RADAR submission")
    parser.add_argument("--raw_scores", type=str, default="raw_scores.txt")
    parser.add_argument("--out_dir", type=str, default="submissions")
    parser.add_argument("--alphas", type=str, default="0.2,0.5,0.8,1.0")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    utt_ids, score_loc, score_dia = read_raw_scores(args.raw_scores)
    print(f"[INFO] loaded {len(utt_ids)} utterances from {args.raw_scores}")

    # 1) branch-only files
    save_pos_neg("submission_loc_only", args.out_dir, utt_ids, score_loc)
    save_pos_neg("submission_dia_only", args.out_dir, utt_ids, score_dia)

    # 2) raw fusion files
    alpha_list = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    for a in alpha_list:
        fused = a * score_loc + (1.0 - a) * score_dia
        tag = f"submission_fuse_raw_alpha{a:.1f}"
        save_pos_neg(tag, args.out_dir, utt_ids, fused)

    # 3) z-score fusion files
    loc_z = zscore(score_loc)
    dia_z = zscore(score_dia)
    for a in alpha_list:
        fused_z = a * loc_z + (1.0 - a) * dia_z
        tag = f"submission_fuse_z_alpha{a:.1f}"
        save_pos_neg(tag, args.out_dir, utt_ids, fused_z)

    print(f"[DONE] all files saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
