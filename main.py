# BAT-Mamba: main.py
# Multi-task progressive training:
#   Phase 1 (ep 0-2):   L = L_loc                        lambda=(1,0,0)
#   Phase 2 (ep 3-12):  L = L_loc + 0.5*L_bound          lambda=(1,0.5,0)
#   Phase 3 (ep 13+):   L = L_loc + L_bound + L_dia      lambda=(1,1,1)

import argparse
import sys
import os
import shutil
import re
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_utils import (
    Dataset_in_the_wild_eval, genSpoof_list,
    Dataset_PartialSpoof_train, Dataset_PartialSpoof_eval,
    load_seglab, parse_ps_protocol,
)
from model import Model, FocalLoss, P2SGradLoss
from utils import reproducibility
import numpy as np

# AMP compatibility: support both new torch.amp and legacy torch.cuda.amp
try:
    AMP_AUTocast = torch.amp.autocast
except AttributeError:
    AMP_AUTocast = torch.cuda.amp.autocast

try:
    _ = torch.amp.GradScaler
    def make_grad_scaler(device):
        return torch.amp.GradScaler(device)
except AttributeError:
    def make_grad_scaler(_device):
        return torch.cuda.amp.GradScaler()


# ============================================================
# Cosine LR scheduler with linear warmup
# Ref: "Bag of Tricks" (He et al.), widely used in INTERSPEECH/ICASSP top systems
# Warmup prevents large gradient steps from LWS weights at epoch 0.
# ============================================================
def get_cosine_schedule_with_warmup(optimizer, num_warmup_epochs, num_training_epochs, min_lr_ratio=0.05):
    """Returns a LambdaLR scheduler with linear warmup + cosine decay.
    min_lr_ratio: final LR = base_lr * min_lr_ratio (default 5% of peak).
    """
    import math
    def lr_lambda(epoch):
        if epoch < num_warmup_epochs:
            return float(epoch + 1) / float(max(1, num_warmup_epochs))
        progress = float(epoch - num_warmup_epochs) / float(
            max(1, num_training_epochs - num_warmup_epochs))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================
# Mixup augmentation (feature-space)
# Ref: Zhang et al. ICLR 2018; adopted by multiple anti-spoofing top systems
# Interpolates pairs of (features, labels) to improve generalisation.
# Applied in feature space (after XLSR) to avoid recomputing SSL features.
# alpha controls interpolation strength: 0 = no mixup, 0.2 = mild.
# ============================================================
def mixup_data(x, y_frame, y_sent, alpha=0.2, device='cuda'):
    """Apply Mixup to a batch.
    x:      [B, T] waveform
    y_frame:[B, T] frame labels (long) — mixed as float then rounded
    y_sent: [B]   sentence labels (long)
    Returns mixed versions + lambda used.
    """
    if alpha > 0:
        import numpy as np
        lam = float(np.random.beta(alpha, alpha))
    else:
        return x, y_frame, y_sent, 1.0
    lam = max(lam, 1 - lam)  # always keep majority component
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=device)
    mixed_x = lam * x + (1 - lam) * x[index]
    # Frame labels: weighted blend — use majority label per frame
    # (lam >= 0.5 guaranteed above, so original label dominates)
    mixed_y_frame = y_frame  # keep original frame labels (majority)
    mixed_y_sent  = y_sent   # keep original sent labels
    return mixed_x, mixed_y_frame, mixed_y_sent, lam


# ============================================================
# Helper: get progressive loss weights by epoch
# ============================================================
def get_loss_weights(epoch):
    """Returns (lambda1, lambda2, lambda3) for the three loss terms.
    Phase 1 (ep 0-2):  L = L_loc                   — build basic acoustic representation
    Phase 2 (ep 3-12): L = L_loc + 0.5*L_bound     — introduce boundary with reduced weight
    Phase 3 (ep 13+):  L = L_loc + L_bound + L_dia — full multi-task
    """
    if epoch < 3:
        return 1.0, 0.0, 0.0   # Phase 1: loc only
    elif epoch < 13:
        return 1.0, 0.5, 0.0   # Phase 2: loc + gentle boundary (0.5 not 1.0)
    else:
        return 1.0, 1.0, 1.0   # Phase 3: full


def get_loss_weights_debug(_epoch):
    """Debug version: activates boundary+dia losses with gentler weights immediately.
    NOTE: Only used when debug_steps > 0 (pipeline smoke-test).
    Do NOT use for real training; use get_loss_weights() instead.
    """
    return 1.0, 0.5, 0.5  # gentler weights even in debug to avoid gradient explosion


# make_frame_labels removed: PartialSpoof Dataset now returns real frame labels directly.


# ============================================================
# Evaluation helpers
# ============================================================
def evaluate_accuracy(dev_loader, model, device, debug_steps=0, epoch=0):
    """Validation loop. dev_loader yields (waveform, frame_labels, boundary_labels, utt_id).
    Computes the same composite loss as training (with phase-appropriate weights),
    so val_loss is directly comparable to training loss.
    debug_steps: if > 0, only run this many batches.
    """
    val_loss = 0.0
    num_total = 0.0
    model.eval()
    criterion_ce      = nn.CrossEntropyLoss()
    criterion_bound   = FocalLoss(alpha=0.25, gamma=2.0)
    criterion_p2sg    = P2SGradLoss(scale=30.0)
    lam1, lam2, lam3 = get_loss_weights(epoch)
    num_batch = len(dev_loader)
    with torch.no_grad():
        for i, batch_data in enumerate(dev_loader):
            if debug_steps > 0 and i >= debug_steps:
                break
            # Dev loader yields 4-tuple: (waveform, frame_labels, boundary_labels, utt_id)
            # utt_id is a string — not a sentence-level int label, so we skip L_sent in val
            if len(batch_data) == 4:
                batch_x, batch_fl, batch_bl, _ = batch_data
            else:
                batch_x, batch_fl, batch_bl = batch_data

            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x  = batch_x.to(device)
            batch_fl = batch_fl.to(device)
            batch_bl = batch_bl.unsqueeze(-1).to(device)

            # Validation also uses AMP to avoid FP32 memory spikes on long runs
            with AMP_AUTocast('cuda'):
                _, p_bound_logits, logits_dia, h_prime = model(batch_x)
            B, T, C = logits_dia.shape

            # Frame-level composite loss (same as training, minus L_sent which needs int label)
            loss_loc = criterion_ce(
                logits_dia.reshape(B * T, C),
                batch_fl.reshape(B * T)
            )
            loss_bound = criterion_bound(p_bound_logits, batch_bl) if lam2 > 0 \
                else torch.tensor(0.0, device=device)
            loss_dia = criterion_p2sg(
                h_prime, batch_fl,
                model.attractor_head.attractor_tokens
            ) if lam3 > 0 else torch.tensor(0.0, device=device)

            loss = lam1 * loss_loc + lam2 * loss_bound + lam3 * loss_dia
            val_loss += loss.item() * batch_size
            print("batch %i/%i (val)" % (i + 1, num_batch), end="\r")

    val_loss /= num_total
    print('Val loss: %.4f' % val_loss)
    return val_loss


def produce_evaluation_file(dataset, model, device, save_path):
    """Evaluation file writer. dataset yields (waveform, frame_labels, boundary_labels, utt_id).
    Score = weighted ensemble of two complementary paths:
      - score_loc : sentence-level head (loc_cls), directly supervised by L_sent
      - score_dia : frame-level attractor logits aggregated by logsumexp over spoof classes
    Both are in logit space; higher = more spoof.
    """
    data_loader = DataLoader(dataset, batch_size=24, shuffle=False, drop_last=False)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, total=len(data_loader)):
            # Support both 4-tuple (PartialSpoof eval) and 2-tuple (legacy eval)
            if len(batch) == 4:
                batch_x, _, _, utt_id = batch
            else:
                batch_x, utt_id = batch
            batch_x = batch_x.to(device)
            logits_loc, _, logits_dia, _ = model(batch_x)

            # Path A: sentence-level head score (directly supervised)
            # logits_loc: [B, 2]  -> spoof logit [B]
            score_loc = logits_loc[:, 1]  # [B]

            # Path B: frame-level attractor aggregation
            # logsumexp over all spoof classes -> mean over time -> [B]
            spoof_logits = logits_dia[:, :, 1:]   # [B, T, num_spoof]
            score_dia = torch.logsumexp(spoof_logits, dim=-1).mean(dim=1)  # [B]

            # Ensemble: equal weight (tune alpha if needed)
            alpha = 0.5
            batch_score = (alpha * score_loc + (1 - alpha) * score_dia).cpu().numpy().ravel()

            ratios = model.compute_spoof_ratio(logits_dia)
            with open(save_path, 'a+') as fh:
                for f, cm, ratio in zip(utt_id, batch_score.tolist(), ratios):
                    ratio_str = ' '.join(
                        ['cls%d=%.1f%%' % (k, v) for k, v in ratio.items()])
                    fh.write('{} {} {}\n'.format(f, cm, ratio_str))
    print('Scores saved to {}'.format(save_path))


# ============================================================
# Training epoch with progressive multi-task loss
# ============================================================
def train_epoch(train_loader, model, optimizer, device, epoch, checkpoint_dir=None, debug_steps=0,
                scheduler=None, scaler=None, best_loss=None, bests=None):
    """PartialSpoof training epoch.
    train_loader yields: (waveform [B,66800], frame_labels [B,208], boundary_labels [B,208], utt_label [B])
    checkpoint_dir: if set, saves model every 1000 steps to prevent data loss.
    debug_steps: if > 0, only run this many batches (quick pipeline test).
    """
    model.train()
    num_total = 0.0
    total_loss = 0.0
    criterion_ce      = nn.CrossEntropyLoss()              # frame-level classification (L_loc)
    criterion_ce_sent = nn.CrossEntropyLoss()              # sentence-level classification (L_sent)
    criterion_bound   = FocalLoss(alpha=0.25, gamma=2.0)   # boundary detection (L_bound)
    criterion_p2sg    = P2SGradLoss(scale=30.0)            # attractor alignment (L_dia)
    lam1, lam2, lam3 = get_loss_weights_debug(epoch) if debug_steps > 0 else get_loss_weights(epoch)
    # L_sent is always active (Phase 1+): gives loc_pool/loc_cls dedicated supervision
    lam_sent = 0.5
    print('Phase weights: lam1=%.1f  lam_sent=%.1f  lam2=%.1f  lam3=%.1f' % (lam1, lam_sent, lam2, lam3))
    # [V3 RESUME] keep a persistent GradScaler across epochs when provided
    scaler = scaler if scaler is not None else make_grad_scaler('cuda')  # AMP scaler
    pbar = tqdm(train_loader, total=len(train_loader))
    for step, (batch_x, batch_fl, batch_bl, batch_ul) in enumerate(pbar):
        if debug_steps > 0 and step >= debug_steps:
            break
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x  = batch_x.to(device)
        batch_fl = batch_fl.to(device)
        batch_bl = batch_bl.unsqueeze(-1).to(device)
        batch_ul = batch_ul.to(device)   # [B] sentence-level 0/1 labels

       # Mixup augmentation (waveform space, alpha=0.2, active from Phase 2+)
        # Disabled in Phase 1 to let model first learn basic representation
        # if lam2 > 0 or lam3 > 0:
        #     batch_x, batch_fl, batch_ul, _lam = mixup_data(
        #         batch_x, batch_fl, batch_ul, alpha=0.2, device=device)

        optimizer.zero_grad()
        with AMP_AUTocast('cuda'):  # AMP: FP16 forward pass
            logits_loc, p_bound_logits, logits_dia, h_prime = model(batch_x)
            B, T, C = logits_dia.shape

            # L_loc: frame-level CE over attractor-enhanced features
            loss_loc = criterion_ce(
                logits_dia.reshape(B * T, C),
                batch_fl.reshape(B * T)
            )
            # L_sent: sentence-level CE on loc_cls head (always active)
            # logits_loc: [B, 2]  batch_ul: [B] with 0=bonafide, 1=spoof
            loss_sent = criterion_ce_sent(logits_loc, batch_ul)
            # L_bound: focal loss on boundary predictions
            loss_bound = criterion_bound(p_bound_logits, batch_bl) if lam2 > 0 \
                else torch.tensor(0.0, device=device)
            # L_dia: P2SGrad loss aligning h_prime to attractor tokens
            loss_dia = criterion_p2sg(
                h_prime, batch_fl,
                model.attractor_head.attractor_tokens
            ) if lam3 > 0 else torch.tensor(0.0, device=device)

            loss = lam1 * loss_loc + lam_sent * loss_sent + lam2 * loss_bound + lam3 * loss_dia

        total_loss += loss.item() * batch_size
        scaler.scale(loss).backward()   # AMP: scaled backward
        scaler.step(optimizer)          # AMP: scaled optimizer step
        scaler.update()                 # AMP: update scaler

        pbar.set_postfix({
            'loss': '%.4f' % (total_loss / num_total),
            'loc':  '%.4f' % loss_loc.item(),
            'sent': '%.4f' % loss_sent.item(),
            'bnd':  '%.4f' % loss_bound.item(),
            'dia':  '%.4f' % loss_dia.item(),
        })

        # [V3 RESUME] Save full-state checkpoint every 1000 steps
        if checkpoint_dir is not None and (step + 1) % 1000 == 0:
            ckpt_path = os.path.join(checkpoint_dir,
                'checkpoint_ep{}_step{}.pth'.format(epoch, step + 1))
            ckpt_payload = {
                'epoch': epoch,
                'step': step + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler is not None else None,
                'scaler': scaler.state_dict() if scaler is not None else None,
                'best_loss': best_loss,
                'bests': bests.tolist() if isinstance(bests, np.ndarray) else bests,
            }
            torch.save(ckpt_payload, ckpt_path)
            print('\nCheckpoint saved: {}'.format(ckpt_path))
            # Delete previous checkpoint for this epoch to save disk space
            prev_step = step + 1 - 1000
            if prev_step > 0:
                prev_ckpt = os.path.join(checkpoint_dir,
                    'checkpoint_ep{}_step{}.pth'.format(epoch, prev_step))
                if os.path.exists(prev_ckpt):
                    os.remove(prev_ckpt)

    sys.stdout.flush()
    return scaler


# ============================================================
# Main entry
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BAT-Mamba')
    parser.add_argument('--database_path', type=str, default='./data/')
    parser.add_argument('--protocols_path', type=str, default='./data/')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--num_epochs', type=int, default=25)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='WCE')
    parser.add_argument('--emb-size', type=int, default=144)
    parser.add_argument('--num_encoders', type=int, default=12)
    parser.add_argument('--num_classes', type=int, default=3,
                        help='0=bonafide,1=TTS,2=VC')
    parser.add_argument('--FT_W2V', default=True,
                        type=lambda x: (str(x).lower() in ['true','yes','1']))
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--comment', type=str, default=None)
    parser.add_argument('--debug_steps', type=int, default=0,
                        help='If > 0, only run this many batches per epoch/eval (quick pipeline test)')
    parser.add_argument('--comment_eval', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None,
                        help='[V3 RESUME] path to checkpoint .pth for seamless resume')
    parser.add_argument('--train', default=True,
                        type=lambda x: (str(x).lower() in ['true','yes','1']))
    parser.add_argument('--n_mejores_loss', type=int, default=5)
    parser.add_argument('--average_model', default=True,
                        type=lambda x: (str(x).lower() in ['true','yes','1']))
    parser.add_argument('--n_average_model', default=5, type=int)
    parser.add_argument('--algo', type=int, default=5)
    parser.add_argument('--N_f', type=int, default=5)
    parser.add_argument('--nBands', type=int, default=5)
    parser.add_argument('--minF', type=int, default=20)
    parser.add_argument('--maxF', type=int, default=8000)
    parser.add_argument('--minBW', type=int, default=100)
    parser.add_argument('--maxBW', type=int, default=1000)
    parser.add_argument('--minCoeff', type=int, default=10)
    parser.add_argument('--maxCoeff', type=int, default=100)
    parser.add_argument('--minG', type=int, default=0)
    parser.add_argument('--maxG', type=int, default=0)
    parser.add_argument('--minBiasLinNonLin', type=int, default=5)
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20)
    parser.add_argument('--P', type=int, default=10)
    parser.add_argument('--g_sd', type=int, default=2)
    parser.add_argument('--SNRmin', type=int, default=10)
    parser.add_argument('--SNRmax', type=int, default=40)

    if not os.path.exists('models'):
        os.mkdir('models')
    args = parser.parse_args()
    args.track = 'LA'
    print(args)
    reproducibility(args.seed, args)

    track = args.track
    n_mejores = args.n_mejores_loss
    assert track in ['LA','DF','In-the-Wild'], 'Invalid track'
    assert args.n_average_model < args.n_mejores_loss + 1

    model_tag = 'BATmamba{}_{}_{}_{}_ES{}_NE{}'.format(
        args.algo, track, args.loss, args.lr, args.emb_size, args.num_encoders)
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join('models', model_tag)
    print('Model tag: ' + model_tag)
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    best_save_path = os.path.join(model_save_path, 'best')
    if not os.path.exists(best_save_path):
        os.mkdir(best_save_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: {}'.format(device))
    model = Model(args, device)
    if not args.FT_W2V:
        for param in model.ssl_model.parameters():
            param.requires_grad = False
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Cosine LR scheduler with 2-epoch linear warmup
    # Warmup protects LWS weights from large updates at epoch 0
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_epochs=2,
        num_training_epochs=min(args.num_epochs + 10, 50),
        min_lr_ratio=0.05
    )

    # [V3 RESUME] persistent GradScaler and seamless state restore
    scaler = make_grad_scaler('cuda')
    resume_next_epoch = 0
    resume_bests = None
    resume_best_loss = None
    if args.resume:
        if not os.path.exists(args.resume):
            print('ERROR: resume checkpoint not found: {}'.format(args.resume))
            sys.exit(1)
        print('[V3 RESUME] Loading checkpoint: {}'.format(args.resume))
        ckpt = torch.load(args.resume, map_location=device)

        # Backward compatibility: old checkpoint may be plain state_dict
        if isinstance(ckpt, dict) and 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
            if ckpt.get('optimizer') is not None:
                optimizer.load_state_dict(ckpt['optimizer'])
            if ckpt.get('scheduler') is not None:
                scheduler.load_state_dict(ckpt['scheduler'])
            if ckpt.get('scaler') is not None:
                scaler.load_state_dict(ckpt['scaler'])
            resume_next_epoch = int(ckpt.get('epoch', -1)) + 1
            resume_best_loss = ckpt.get('best_loss', None)
            resume_bests = ckpt.get('bests', None)
            print('[V3 RESUME] Restored full state. Next epoch: {}'.format(resume_next_epoch))
        else:
            # Fallback for legacy model-only checkpoints (e.g. checkpoint_ep12_step6000.pth old format)
            model.load_state_dict(ckpt)
            m = re.search(r'ep(\d+)', os.path.basename(args.resume))
            if m:
                resume_next_epoch = int(m.group(1)) + 1
            else:
                resume_next_epoch = 0
            print('[V3 RESUME] Legacy model-only checkpoint detected. '
                  'Optimizer/scheduler/scaler reset. Next epoch: {}'.format(resume_next_epoch))

    # ---- In-the-Wild eval only ----
    if args.track == 'In-the-Wild':
        best_save_path = best_save_path.replace(track, 'LA')
        model_save_path = model_save_path.replace(track, 'LA')
        print('######## Eval In-the-Wild ########')
        model.load_state_dict(torch.load(
            os.path.join(best_save_path, 'best_0.pth')))
        sd = model.state_dict()
        for i in range(1, args.n_average_model):
            model.load_state_dict(torch.load(
                os.path.join(best_save_path, 'best_{}.pth'.format(i))))
            sd2 = model.state_dict()
            for key in sd:
                sd[key] = sd[key] + sd2[key]
        for key in sd:
            sd[key] = sd[key] / args.n_average_model
        model.load_state_dict(sd)
        file_eval = genSpoof_list(
            dir_meta=os.path.join(args.protocols_path),
            is_train=False, is_eval=True)
        eval_set = Dataset_in_the_wild_eval(
            list_IDs=file_eval, base_dir=os.path.join(args.database_path))
        produce_evaluation_file(eval_set, model, device,
            'Scores/{}/{}.txt'.format(args.track, model_tag))
        sys.exit(0)

    # ---- PartialSpoof Data loaders ----
    # Paths (based on H:\PS_data layout):
    #   audio:    database_path/{train,dev,eval}/con_wav/*.wav
    #   seglab:   database_path/segment_labels/{split}_seglab_0.02.npy
    #   protocol: protocols_path/PartialSpoof_LA_cm_protocols/PartialSpoof.LA.cm.{split}.trl.txt

    # Load segment labels (.npy) -- '0'=spoof, '1'=bonafide (will be flipped in Dataset)
    seglab_train = load_seglab(
        os.path.join(args.database_path, 'segment_labels', 'train_seglab_0.02.npy'))
    seglab_dev   = load_seglab(
        os.path.join(args.database_path, 'segment_labels', 'dev_seglab_0.02.npy'))

    # Parse CM protocol for audio IDs
    ps_proto_dir = os.path.join(args.protocols_path,
                                'protocols', 'PartialSpoof_LA_cm_protocols')
    files_id_train, utt_labels_train = parse_ps_protocol(
        os.path.join(ps_proto_dir, 'PartialSpoof.LA.cm.train.trl.txt'))
    files_id_dev, _   = parse_ps_protocol(
        os.path.join(ps_proto_dir, 'PartialSpoof.LA.cm.dev.trl.txt'))
    print('no. of training trials', len(files_id_train))
    print('no. of validation trials', len(files_id_dev))

    train_set = Dataset_PartialSpoof_train(
        list_IDs=files_id_train,
        seglab=seglab_train,
        utt_labels=utt_labels_train,
        base_dir=os.path.join(args.database_path, 'train', 'con_wav'),
        args=args,
        algo=args.algo,
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
        num_workers=0, shuffle=True, drop_last=True)
    del train_set

    dev_set = Dataset_PartialSpoof_eval(
        list_IDs=files_id_dev,
        seglab=seglab_dev,
        base_dir=os.path.join(args.database_path, 'dev', 'con_wav'),
    )
    dev_loader = DataLoader(dev_set, batch_size=4, num_workers=0, shuffle=False)
    del dev_set

    # ---- Debug mode: limit batches for quick pipeline test ----
    debug_steps = args.debug_steps  # 0 = disabled, e.g. 5 = only 5 batches
    not_improving = 0
    epoch = resume_next_epoch
    bests = np.ones(n_mejores, dtype=float) * float('inf')
    if resume_bests is not None:
        rb = np.array(resume_bests, dtype=float).reshape(-1)
        bests[:min(len(rb), n_mejores)] = rb[:min(len(rb), n_mejores)]
    best_loss = float(np.min(bests)) if np.isfinite(np.min(bests)) else float('inf')
    if resume_best_loss is not None:
        best_loss = float(resume_best_loss)

    if args.train:
        # NOTE: Do NOT pre-fill best_*.pth with np.savetxt placeholders
        # (that would overwrite real saved models on restart)
        while not_improving < args.num_epochs:
            print('######## Epoch {} ########'.format(epoch))
            scaler = train_epoch(
                train_loader, model, optimizer, device, epoch,
                checkpoint_dir=model_save_path, debug_steps=debug_steps,
                scheduler=scheduler, scaler=scaler, best_loss=best_loss, bests=bests
            )
            val_loss = evaluate_accuracy(dev_loader, model, device,
                                         debug_steps=debug_steps, epoch=epoch)
            # Step LR scheduler (after val, before next epoch)
            # Dynamic set_lr.txt overrides scheduler if present
            scheduler.step()
            if val_loss < best_loss:
                best_loss = val_loss
                # [V3 RESUME] Save full training state for best checkpoint
                torch.save({
                    'epoch': epoch,
                    'step': None,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict() if scaler is not None else None,
                    'best_loss': best_loss,
                    'bests': bests.tolist(),
                }, os.path.join(model_save_path, 'best.pth'))
                print('New best epoch')
                not_improving = 0
            else:
                not_improving += 1
            for i in range(n_mejores):
                if bests[i] > val_loss:
                    for t in range(n_mejores - 1, i, -1):
                        bests[t] = bests[t - 1]
                        src = os.path.join(best_save_path,
                            'best_{}.pth'.format(t - 1))
                        dst = os.path.join(best_save_path,
                            'best_{}.pth'.format(t))
                        if os.path.exists(src):
                            shutil.move(src, dst)
                    bests[i] = val_loss
                    torch.save(model.state_dict(),
                        os.path.join(best_save_path,
                            'best_{}.pth'.format(i)))
                    break
            print('\n{} - val_loss={:.4f}'.format(epoch, val_loss))
            print('n-best losses:', bests)

            # ---- Dynamic LR override via signal file ----
            # To change LR during training without restarting:
            #   Create a file: models/<model_tag>/set_lr.txt
            #   Content: a single float, e.g.  5e-7
            # The new LR will take effect at the START of the next epoch.
            lr_signal_file = os.path.join(model_save_path, 'set_lr.txt')
            if os.path.exists(lr_signal_file):
                try:
                    with open(lr_signal_file, 'r') as f:
                        new_lr = float(f.read().strip())
                    for pg in optimizer.param_groups:
                        pg['lr'] = new_lr
                    print('*** LR changed to {} (from set_lr.txt) ***'.format(new_lr))
                    os.remove(lr_signal_file)  # consume the signal
                except Exception as e:
                    print('WARNING: Failed to parse set_lr.txt: {}'.format(e))
            epoch += 1
            if epoch > 74:
                break
        print('Total epochs: ' + str(epoch))

    # ---- Final evaluation ----
    print('######## Eval ########')
    if args.average_model:
        actual_n = sum(
            1 for i in range(args.n_average_model)
            if os.path.exists(os.path.join(best_save_path,
                'best_{}.pth'.format(i)))
            and os.path.getsize(os.path.join(best_save_path,
                'best_{}.pth'.format(i))) > 1000)
        n_avg = min(args.n_average_model, actual_n)
        print('Averaging {} best models'.format(n_avg))

        if n_avg == 0:
            # No valid best_*.pth — fall back to best.pth or latest checkpoint
            best_single = os.path.join(model_save_path, 'best.pth')
            checkpoints = sorted(
                [f for f in os.listdir(model_save_path)
                 if f.startswith('checkpoint_') and f.endswith('.pth')],
                key=lambda x: os.path.getmtime(os.path.join(model_save_path, x))
            )
            if os.path.exists(best_single) and os.path.getsize(best_single) > 1000:
                print('Loading best.pth')
                model.load_state_dict(torch.load(best_single,
                    map_location=device))
            elif checkpoints:
                latest = os.path.join(model_save_path, checkpoints[-1])
                print('Loading latest checkpoint: {}'.format(latest))
                model.load_state_dict(torch.load(latest,
                    map_location=device))
            else:
                print('ERROR: No valid model found. Please train first.')
                sys.exit(1)
        else:
            model.load_state_dict(torch.load(
                os.path.join(best_save_path, 'best_0.pth'),
                map_location=device))
            sd = model.state_dict()
            for i in range(1, n_avg):
                model.load_state_dict(torch.load(
                    os.path.join(best_save_path, 'best_{}.pth'.format(i)),
                    map_location=device))
                sd2 = model.state_dict()
                for key in sd:
                    sd[key] = sd[key] + sd2[key]
            for key in sd:
                sd[key] = sd[key] / n_avg
            model.load_state_dict(sd)
    else:
        best_single = os.path.join(model_save_path, 'best.pth')
        checkpoints = sorted(
            [f for f in os.listdir(model_save_path)
             if f.startswith('checkpoint_') and f.endswith('.pth')],
            key=lambda x: os.path.getmtime(os.path.join(model_save_path, x))
        )
        if os.path.exists(best_single) and os.path.getsize(best_single) > 1000:
            model.load_state_dict(torch.load(best_single, map_location=device))
        elif checkpoints:
            latest = os.path.join(model_save_path, checkpoints[-1])
            print('Loading latest checkpoint: {}'.format(latest))
            model.load_state_dict(torch.load(latest, map_location=device))
        else:
            print('ERROR: No valid model found. Please train first.')
            sys.exit(1)

    if args.comment_eval:
        model_tag = model_tag + '_{}'.format(args.comment_eval)
    os.makedirs('./Scores/PartialSpoof', exist_ok=True)
    score_path = './Scores/PartialSpoof/{}.txt'.format(model_tag)
    if not os.path.exists(score_path):
        seglab_eval = load_seglab(
            os.path.join(args.database_path, 'segment_labels', 'dev_seglab_0.02.npy'))
        ps_proto_dir = os.path.join(args.protocols_path,
                                    'protocols', 'PartialSpoof_LA_cm_protocols')
        files_id_eval, _ = parse_ps_protocol(
            os.path.join(ps_proto_dir, 'PartialSpoof.LA.cm.dev.trl.txt'),
            is_eval=True)
        print('no. of eval trials (using dev set)', len(files_id_eval))
        eval_set = Dataset_PartialSpoof_eval(
            list_IDs=files_id_eval,
            seglab=seglab_eval,
            base_dir=os.path.join(args.database_path, 'dev', 'con_wav'),
        )
        produce_evaluation_file(eval_set, model, device, score_path)
    else:
        print('Score file already exists')
