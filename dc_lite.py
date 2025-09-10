# dc_lite.py
# A compact Dynamic-Chunking language model for bytes (vocab=256).

import math, torch, torch.nn as nn, torch.nn.functional as F

# ------------------ tiny transformer blocks ------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3*d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", None, persistent=False)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x)  # [B,T,3C]
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B,h,T,dh]
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B,h,T,T]
        if self.mask is None or self.mask.size(-1) != T:
            self.mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v  # [B,h,T,dh]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        return y

class TransformerBlock(nn.Module):
    def __init__(self, d_model=256, n_heads=4, mlp_mult=2.0, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(mlp_mult*d_model)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(mlp_mult*d_model), d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=8192):
        super().__init__()
        pos = torch.arange(0, max_len).unsqueeze(1)
        i = torch.arange(0, d_model, 2)
        div = torch.exp(-math.log(10000.0) * i / d_model)
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0).to(x.dtype)

# ------------------ dynamic chunking core ------------------
def ste_hard_sigmoid(logits, thresh=0.5):
    """Straight-through estimator: hard step in forward; sigmoid gradient in backward."""
    p = torch.sigmoid(logits)
    b = (p > thresh).float()
    return b + (p - p.detach())  # hard value with sigmoid gradient

def ema_smooth(x, tau=0.6):
    """EMA along the time dimension for smoothing router logits (x: [B,T,1])."""
    B, T, C = x.shape
    out = torch.zeros_like(x)
    out[:,0] = x[:,0]
    for t in range(1, T):
        out[:,t] = tau*out[:,t-1] + (1.0 - tau)*x[:,t]
    return out

def segment_mean(states, seg_ids, num_segs):
    """
    Mean over variable-length segments for each batch independently.
    states: [T,C], seg_ids: [T] in [0..num_segs-1]
    returns [num_segs, C]
    """
    C = states.size(-1)
    sums = states.new_zeros(num_segs, C)
    counts = states.new_zeros(num_segs, 1)
    sums.index_add_(0, seg_ids, states)
    one = torch.ones_like(seg_ids, dtype=states.dtype).unsqueeze(-1)
    counts.index_add_(0, seg_ids, one)
    return sums / counts.clamp_min(1.0)

# === NEW: helper to convert a boundary mask to chunk-length samples ===
def _chunk_lengths_from_mask(bmask: torch.Tensor):
    """
    bmask: Bool tensor [B, T] with True at boundary positions.
    Returns a Python list of chunk lengths collected across the batch.
    """
    lens = []
    bm = bmask.to(torch.bool)
    for row in bm:  # row shape [T]
        run = 0
        for flag in row:
            run += 1
            if bool(flag):
                lens.append(run)
                run = 0
        if run > 0:   # tail if sequence ends without boundary
            lens.append(run)
    return lens

# ------------------ the model ------------------
class DCLiteLM(nn.Module):
    def __init__(
        self,
        vocab_size=256,
        d_model_tok=256,
        d_model_chunk=384,
        n_layers_tok=2,
        n_heads_tok=4,
        n_layers_chunk=4,
        n_heads_chunk=6,
        mlp_mult=2.0,
        dropout=0.1,
        target_chunk_len=64,   # ~ average chunk length
        boundary_rate_weight=0.05,
        smooth_tau=0.6,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.target_chunk_len = target_chunk_len
        self.boundary_rate_weight = boundary_rate_weight
        self.smooth_tau = smooth_tau
        self.boundary_thresh = 0.5            # === NEW: threshold used for masks ===

        # Token embeddings + tiny encoder
        self.tok_embed = nn.Embedding(vocab_size, d_model_tok)
        self.pos_tok = PositionalEncoding(d_model_tok)
        self.enc_blocks = nn.ModuleList([
            TransformerBlock(d_model_tok, n_heads_tok, mlp_mult, dropout) for _ in range(n_layers_tok)
        ])
        self.enc_ln = nn.LayerNorm(d_model_tok)

        # Router (predict boundary logits per position)
        self.router = nn.Sequential(
            nn.Linear(d_model_tok, 128), nn.GELU(),
            nn.Linear(128, 1)  # [B,T,1] logits
        )

        # Chunk main net
        self.chunk_in = nn.Linear(d_model_tok, d_model_chunk)
        self.pos_chunk = PositionalEncoding(d_model_chunk)
        self.chunk_blocks = nn.ModuleList([
            TransformerBlock(d_model_chunk, n_heads_chunk, mlp_mult, dropout) for _ in range(n_layers_chunk)
        ])
        self.chunk_ln = nn.LayerNorm(d_model_chunk)

        # Decoder: fuse token enc + its chunk state
        self.fuse = nn.Linear(d_model_tok + d_model_chunk, d_model_tok)
        self.dec_blocks = nn.ModuleList([
            TransformerBlock(d_model_tok, n_heads_tok, mlp_mult, dropout) for _ in range(1)  # 1 light layer
        ])
        self.dec_ln = nn.LayerNorm(d_model_tok)
        self.head = nn.Linear(d_model_tok, vocab_size, bias=False)

    # === CHANGED: add return_chunks flag ===
    def forward(self, x, return_aux=False, return_chunks=False):
        """
        x: [B, T] byte tokens (0..255)
        returns logits: [B, T, vocab], and optionally aux dict with DC stats/losses.
        If return_chunks=True (and return_aux=True), stats will also include
        "chunk_len_samples": a Python list of chunk lengths (bounded by caller).
        """
        B, T = x.size()
        emb = self.tok_embed(x)
        h = self.pos_tok(emb)
        for blk in self.enc_blocks:
            h = blk(h)
        h = self.enc_ln(h)  # token enc states, [B,T,dt]

        # Router -> smooth -> hard boundaries (STE)
        r_logits = self.router(h)                            # [B,T,1]
        r_logits_s = ema_smooth(r_logits, tau=self.smooth_tau)
        b_hard = ste_hard_sigmoid(r_logits_s, thresh=self.boundary_thresh)  # [B,T,1] in {0,1}
        b_hard = b_hard.squeeze(-1)
        # force a boundary at t=0
        b_hard[:, 0] = 1.0

        # chunk ids = cumsum(boundaries) - 1
        chunk_ids = torch.cumsum(b_hard, dim=1) - 1.0
        chunk_ids = chunk_ids.long().clamp_min(0)           # [B,T]

        # --- build per-chunk representations (mean over tokens in chunk) ---
        chunk_states = []
        chunk_id_per_batch = []
        max_chunks = 0
        for i in range(B):
            ids = chunk_ids[i]
            Li = int(ids.max().item()) + 1
            max_chunks = max(max_chunks, Li)
            tok_states = h[i]  # [T,dt]
            reps = segment_mean(tok_states, ids, Li)  # [Li, dt]
            chunk_states.append(reps)
            chunk_id_per_batch.append(ids)

        # pad to max_chunks for batching
        dt = h.size(-1)
        chunk_tensor = h.new_zeros(B, max_chunks, dt)
        chunk_mask = torch.zeros(B, max_chunks, dtype=torch.bool, device=h.device)
        for i, reps in enumerate(chunk_states):
            Li = reps.size(0)
            chunk_tensor[i, :Li] = reps
            chunk_mask[i, :Li] = True

        # main net on chunks
        c = self.chunk_in(chunk_tensor)           # [B,Lc,dc]
        c = self.pos_chunk(c)
        for blk in self.chunk_blocks:
            c = blk(c)
        c = self.chunk_ln(c)                      # [B,Lc,dc]

        # gather per-token their chunk state
        dc = c
        dc_per_token = torch.zeros(B, T, dc.size(-1), device=h.device, dtype=h.dtype)
        for i in range(B):
            ids = chunk_id_per_batch[i].clamp_max(dc.size(1)-1)
            dc_per_token[i] = dc[i].index_select(0, ids)

        # decoder: fuse and predict next byte
        fused = torch.cat([h, dc_per_token], dim=-1)
        y = self.fuse(fused)
        for blk in self.dec_blocks:
            y = blk(y)
        y = self.dec_ln(y)
        logits = self.head(y)                     # [B,T,256]

        # aux: boundary-rate penalty towards 1/target_chunk_len
        boundaries = b_hard.sum(dim=1)  # [B]
        b_rate = boundaries / T
        target_rate = 1.0 / float(self.target_chunk_len)
        aux_loss = self.boundary_rate_weight * ((b_rate - target_rate) ** 2).mean()

        if return_aux:
            stats = {
                "avg_chunk_len": float(T / (boundaries.mean().item()+1e-8)),
                "avg_boundaries": float(boundaries.mean().item()),
                "boundary_rate": float(b_rate.mean().item()),
                "aux_loss": aux_loss,
            }
            # === NEW: optional chunk-length samples for histogram/ECDF ===
            if return_chunks:
                with torch.no_grad():
                    bmask = (torch.sigmoid(r_logits_s.squeeze(-1)) > self.boundary_thresh)
                    bmask[:, 0] = True
                    stats["chunk_len_samples"] = _chunk_lengths_from_mask(bmask)
            return logits, aux_loss, stats

        return logits
