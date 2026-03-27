# BAT-Mamba: Boundary-Aware and Token-Attractor Mamba
# Upgraded from XLSR-Mamba baseline.
#
# Data flow:
#   XLSR -> Linear(1024->D) -> BN -> SELU -> MixerModel (unpooled)
#   -> H_{1:T} [B, T, D]
#       |┌─────────────────────────────┐
#        Branch A: BoundaryAwareHead   Branch B: AttractorCrossAttentionHead
#        P_bound [B,T,1]               logits_dia [B,T,num_classes]
#                                      logits_loc  [B,T,2]  (sentence-level via pool)

import torch
import torch.nn as nn
import fairseq
from dataclasses import dataclass, field
import torch.nn.functional as F
from mamba_blocks import MixerModel


# ============================================================
# Loss Functions
# ============================================================
class FocalLoss(nn.Module):
    """Focal Loss for boundary detection (extreme class imbalance).
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        # pred:   [B, T, 1] raw logits
        # target: [B, T, 1] binary float (1=boundary)
        pred = pred.squeeze(-1)    # [B, T]
        target = target.squeeze(-1).float()  # [B, T]
        p = torch.sigmoid(pred)
        ce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')  # [B,T]
        p_t = p * target + (1 - p) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        loss = alpha_t * (1 - p_t) ** self.gamma * ce  # [B,T]
        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()


class P2SGradLoss(nn.Module):
    """Probability-to-Similarity Gradient Loss.
    Operates in L2-normalised (spherical) space to prevent token collapse.
    For each frame, maximise cosine similarity to the correct attractor token
    and minimise similarity to wrong ones.
    Practically: L2-normalise features, then apply CrossEntropy on cosine scores * scale.
    """
    def __init__(self, scale=30.0):
        super().__init__()
        self.scale = scale
        self.ce = nn.CrossEntropyLoss()

    def forward(self, feat, target, weight_matrix):
        """
        feat:          [B, T, D]  frame-level features (will be L2-normalised)
        target:        [B, T]     long  frame-level class labels
        weight_matrix: [num_cls, D]  attractor token embeddings (will be L2-normalised)
        """
        # L2 normalise both
        feat_norm = F.normalize(feat, dim=-1)              # [B, T, D]
        w_norm    = F.normalize(weight_matrix, dim=-1)     # [num_cls, D]
        # Cosine similarity scores: [B, T, num_cls]
        scores = torch.matmul(feat_norm, w_norm.T) * self.scale
        B, T, C = scores.shape
        # Flatten for CE
        return self.ce(scores.reshape(B * T, C), target.reshape(B * T))


# ============================================================
# Branch A: Boundary-Aware Head
# Input:  H [B, T, D]
# Output: P_bound [B, T, 1]  (sigmoid probabilities)
# ============================================================
class BoundaryAwareHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # Conv1d expects [B, C, L], so we transpose in forward
        self.conv1 = nn.Conv1d(d_model, d_model // 2, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(d_model // 2, 1, kernel_size=3, padding=1)
        self.act   = nn.ReLU(inplace=True)

    def forward(self, h):
        # h: [B, T, D]
        x = h.transpose(1, 2)          # [B, D, T]
        x = self.act(self.conv1(x))    # [B, D//2, T]
        x = self.conv2(x)              # [B, 1, T]
        x = x.transpose(1, 2)         # [B, T, 1]  (raw logits)
        return x


# ============================================================
# Branch B: Attractor Cross-Attention Head
# Input:  H [B, T, D]
# Output: logits_dia [B, T, num_classes]
#         H_prime    [B, T, D]  (attention-enhanced features)
# ============================================================
class AttractorCrossAttentionHead(nn.Module):
    def __init__(self, d_model, num_classes=3, num_heads=4):
        """
        num_classes: 3 = {bonafide, TTS, VC}
        Attractor tokens C: [num_classes, D]  (learnable)
        Cross-Attention: Q=H, K=V=C
        """
        super().__init__()
        self.num_classes = num_classes
        # Learnable attractor tokens: [num_classes, D]
        # Orthogonal init: tokens start maximally spread on the unit sphere,
        # preventing early collapse and accelerating P2SGrad convergence.
        # When num_classes <= d_model, nn.init.orthogonal_ guarantees unit-norm
        # and pairwise orthogonality (cosine similarity = 0 at init).
        _tok = torch.empty(num_classes, d_model)
        nn.init.orthogonal_(_tok)
        self.attractor_tokens = nn.Parameter(_tok)
        # batch_first=True: input/output shape [B, seq, D]
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads,
            batch_first=True, dropout=0.1
        )
        self.norm = nn.LayerNorm(d_model)
        # Frame-level multi-class classifier
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, h):
        """
        h: [B, T, D]
        Returns:
            logits_dia: [B, T, num_classes]
            h_prime:    [B, T, D]
        """
        B, T, D = h.shape
        # Expand attractor tokens to batch: [1, num_classes, D] -> [B, num_classes, D]
        tokens = self.attractor_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, num_cls, D]
        # Cross-Attention: Q=h, K=tokens, V=tokens
        # h_prime: [B, T, D]
        h_prime, _ = self.cross_attn(query=h, key=tokens, value=tokens)
        h_prime = self.norm(h + h_prime)   # residual connection
        logits_dia = self.classifier(h_prime)  # [B, T, num_classes]
        return logits_dia, h_prime


# ============================================================
# SSL Front-end (XLSR wav2vec2.0) -- unchanged from baseline
# ============================================================
@dataclass
class MambaConfig:
    d_model: int = 64
    n_layer: int = 6
    vocab_size: int = 50277
    ssm_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8


class SSLModel(nn.Module):
    def __init__(self, device):
        super(SSLModel, self).__init__()
        cp_path = './xlsr2_300m.pt'
        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        self.device = device
        self.out_dim = 1024

    def extract_feat(self, input_data):
        if next(self.model.parameters()).device != input_data.device \
           or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)
            self.model.train()
        if input_data.ndim == 3:
            input_tmp = input_data[:, :, 0]
        else:
            input_tmp = input_data
        # Single forward pass — previously called model() twice, doubling compute cost.
        # Both 'x' and 'layer_results' are extracted from the same output dict.
        out = self.model(input_tmp, mask=False, features_only=True)
        emb          = out['x']            # [B, T, 1024]
        layer_results = out['layer_results']  # list of 24 layer outputs
        return emb, layer_results


# ============================================================
# Module 1: Layer-wise Weighted Sum (LWS)
# Ref: wav2vec2.0 / HuBERT / SUPERB (INTERSPEECH 2021)
#      "Weighted-sum of all SSL layers outperforms last-layer only"
# Uses all 24 XLSR transformer layers instead of just the last one.
# Learnable scalar weights (softmax-normalised) fuse layer representations.
# Proven to improve robustness to channel/codec distortions.
# ============================================================
class LayerWiseWeightedSum(nn.Module):
    def __init__(self, num_layers=24):
        super().__init__()
        # Learnable weight for each transformer layer
        self.weights = nn.Parameter(torch.zeros(num_layers))

    def forward(self, layer_results):
        """
        layer_results: list of num_layers tensors, each [B, T, 1024]
                       (from fairseq layer_results, each entry is (z, _))
        Returns: [B, T, 1024] weighted sum
        """
        # fairseq returns list of (x, _) tuples per layer
        # x shape: [T, B, 1024] — need to transpose
        stacked = torch.stack(
            [lr[0].transpose(0, 1) for lr in layer_results], dim=0
        )  # [num_layers, B, T, 1024]
        w = torch.softmax(self.weights, dim=0)  # [num_layers]
        # Weighted sum: einsum over layer dimension
        out = torch.einsum('l,lbtd->btd', w, stacked)  # [B, T, 1024]
        return out


# ============================================================
# Module 2: Squeeze-and-Excitation (SE) Block
# Ref: AASIST (ICASSP 2022), RawGAT-ST (INTERSPEECH 2021)
#      Channel-wise feature recalibration: suppresses uninformative dims,
#      amplifies discriminative ones. ~2*D parameters, negligible cost.
# ============================================================
class SEBlock(nn.Module):
    def __init__(self, d_model, reduction=8):
        super().__init__()
        mid = max(d_model // reduction, 4)
        self.fc1 = nn.Linear(d_model, mid)
        self.fc2 = nn.Linear(mid, d_model)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        x: [B, T, D]
        Returns: [B, T, D] channel-recalibrated features
        """
        # Global average pooling over time
        s = x.mean(dim=1)              # [B, D]
        s = self.act(self.fc1(s))      # [B, D//reduction]
        s = torch.sigmoid(self.fc2(s)) # [B, D]  — scale in (0,1)
        return x * s.unsqueeze(1)      # [B, T, D] * [B, 1, D]


# ============================================================
# BAT-Mamba Main Model
# ============================================================
class Model(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.num_classes = getattr(args, 'num_classes', 3)  # bonafide / TTS / VC

        # --- Front-end ---
        self.ssl_model = SSLModel(self.device)
        # Layer-wise weighted sum: fuses all 24 XLSR layers instead of last-layer only
        # Ref: wav2vec2 / SUPERB — proven to boost robustness under codec/channel distortion
        self.lws = LayerWiseWeightedSum(num_layers=24)
        self.LL = nn.Linear(1024, args.emb_size)  # [B,T,1024] -> [B,T,D]
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        # SE block after projection: channel-wise recalibration before Mamba encoder
        # Ref: AASIST (ICASSP 2022) — suppresses uninformative feature dims
        self.se = SEBlock(d_model=args.emb_size, reduction=8)

        # --- Unpooled Bi-Mamba encoder ---
        self.config = MambaConfig(
            d_model=args.emb_size,
            n_layer=args.num_encoders // 2
        )
        print('BAT-Mamba: W2V + Unpooled Bi-Mamba + Dual-Head Decoder')
        print(self.config)
        self.mamba = MixerModel(
            d_model=self.config.d_model,
            n_layer=self.config.n_layer,
            ssm_cfg=self.config.ssm_cfg,
            rms_norm=False,          # Triton disabled (Windows)
            residual_in_fp32=self.config.residual_in_fp32,
            fused_add_norm=False,    # Triton disabled (Windows)
        )  # output: [B, T, D]

        # --- Branch A: Boundary-Aware Head ---
        self.boundary_head = BoundaryAwareHead(d_model=args.emb_size)

        # --- Branch B: Attractor Cross-Attention Head ---
        self.attractor_head = AttractorCrossAttentionHead(
            d_model=args.emb_size,
            num_classes=self.num_classes,
            num_heads=4
        )

        # --- Sentence-level binary classifier (loc head) ---
        # Uses attention pooling over H_prime for final bonafide/spoof decision
        self.loc_pool = nn.Linear(args.emb_size, 1)   # attention weights
        self.loc_cls  = nn.Linear(args.emb_size, 2)   # bonafide vs spoof
        self.dropout  = nn.Dropout(p=0.1)

    def forward(self, x):
        """
        Args:
            x: raw waveform [B, 66800] or [B, 66800, 1]
        Returns:
            logits_loc:  [B, 2]         sentence-level bonafide/spoof
            p_bound:     [B, T, 1]      frame-level boundary prob (after sigmoid)
            logits_dia:  [B, T, num_classes]  frame-level spoof diarization
            h_prime:     [B, T, D]      attention-enhanced features
        """
        # ---- Stage 1: SSL feature extraction ----
        emb, layer_results = self.ssl_model.extract_feat(x.squeeze(-1))
        # Layer-wise weighted sum: fuse all 24 XLSR layers
        # Falls back to last-layer emb if layer_results is empty/malformed
        if layer_results and len(layer_results) == 24:
            emb = self.lws(layer_results)  # [B, T, 1024]
        e = self.LL(emb)                   # [B, T, D]
        e = e.unsqueeze(1)                 # [B, 1, T, D]
        e = self.first_bn(e)               # [B, 1, T, D]
        e = self.selu(e)
        e = e.squeeze(1)                   # [B, T, D]
        # SE block: channel-wise recalibration
        e = self.se(e)                     # [B, T, D]

        # ---- Stage 2: Unpooled Bi-Mamba ----
        h = self.mamba(e)                  # [B, T, D]

        # ---- Stage 3: Dual-Head Parallel Decoder ----
        # Branch A: Boundary-Aware Head — returns raw logits (FocalLoss needs raw logit)
        p_bound_logits = self.boundary_head(h)     # [B, T, 1] raw logits

        # Branch B: Attractor Cross-Attention Head
        logits_dia, h_prime = self.attractor_head(h)  # [B,T,num_cls], [B,T,D]

        # Sentence-level loc: attention pool over h_prime -> [B, D] -> [B, 2]
        attn_w = F.softmax(self.loc_pool(h_prime), dim=1)  # [B, T, 1]
        h_pooled = (attn_w * h_prime).sum(dim=1)            # [B, D]
        h_pooled = self.dropout(h_pooled)
        logits_loc = self.loc_cls(h_pooled)                 # [B, 2]

        return logits_loc, p_bound_logits, logits_dia, h_prime

    def compute_spoof_ratio(self, logits_dia):
        """
        Post-inference: compute per-class ratio from frame-level predictions.
        Args:
            logits_dia: [B, T, num_classes]
        Returns:
            ratio_dict: dict {class_idx: percentage}  for each sample in batch
        """
        preds = logits_dia.argmax(dim=-1)  # [B, T]
        B, T = preds.shape
        ratios = []
        for b in range(B):
            r = {}
            for k in range(self.num_classes):
                r[k] = (preds[b] == k).float().mean().item() * 100.0
            ratios.append(r)
        return ratios
