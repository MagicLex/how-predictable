"""Shared image -> embedding module. THE no-skew keystone.

One frozen backbone, one preprocessing path, used by BOTH the offline embed
pipeline that fills the feature group and the online path that scores an
uploaded photo. Training and serving cannot diverge because they call the same
function on the same weights.

ENCODER is decided by tools/benchmark_encoders.py (pawpularity pairwise CV) and
then pinned here. Candidates share the same call surface; only the winner ships.
Embeddings are L2-normalized float32. Backbones are frozen everywhere: we only
train heads on top, which keeps the whole system CPU-viable.
"""
import numpy as np

# torch imports lazily inside the embed functions: the pandas training env has
# no torch, and it only needs this module's constants (ENCODER/MODEL_ID/dim).

ENCODERS = {
    "clip":   {"model_id": "openai/clip-vit-base-patch32",     "dim": 512},
    "siglip": {"model_id": "google/siglip-base-patch16-224",   "dim": 768},
    "dinov2": {"model_id": "facebook/dinov2-base",             "dim": 768},
}
ENCODER = "siglip"          # pinned by the benchmark; do not change casually
MODEL_ID = ENCODERS[ENCODER]["model_id"]
EMBED_DIM = ENCODERS[ENCODER]["dim"]

_model = None
_proc = None
_loaded_key = None


def _load(key=None):
    global _model, _proc, _loaded_key
    key = key or ENCODER
    if _model is None or _loaded_key != key:
        from transformers import AutoModel, AutoImageProcessor
        mid = ENCODERS[key]["model_id"]
        _model = AutoModel.from_pretrained(mid)
        _model.eval()
        _proc = AutoImageProcessor.from_pretrained(mid)
        _loaded_key = key
    return _model, _proc


def _image_features(model, inp):
    import torch
    if hasattr(model, "get_image_features"):        # CLIP / SigLIP
        f = model.get_image_features(**inp)
        return f if isinstance(f, torch.Tensor) else f.pooler_output
    out = model(**inp)                              # DINOv2: CLS token
    return out.last_hidden_state[:, 0]


def embed_images(pil_images, batch_size=64, encoder=None):
    """List of PIL images -> (n, dim) float32 L2-normalized numpy array."""
    import torch
    model, proc = _load(encoder)
    out = []
    with torch.no_grad():
        for i in range(0, len(pil_images), batch_size):
            inp = proc(images=pil_images[i:i + batch_size], return_tensors="pt")
            f = _image_features(model, inp)
            f = f / f.norm(dim=-1, keepdim=True)
            out.append(f.cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0)


# Zero-shot appeal baseline: the honest bar any trained head must beat.
# Cosine against contrastive prompts; score = appealing - unappealing.
# Zero-shot appeal baseline: always CLIP (BPE tokenizer, no sentencepiece --
# the siglip/T5 tokenizer family crashes on the stock torch env). The baseline
# does not need to live in the winner's space: it is a floor computed once in
# the benchmark, and quoted from there by the training pipeline.
APPEAL_PROMPTS = ("an adorable, sharp, well-lit photo of a cute pet looking at the camera",
                  "a blurry, dark, unappealing photo of a pet")


def zero_shot_appeal(pil_images, batch_size=64):
    """(n,) zero-shot appeal score per image, in CLIP space."""
    import torch
    model, proc = _load("clip")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ENCODERS["clip"]["model_id"])
    with torch.no_grad():
        t = tok(list(APPEAL_PROMPTS), padding=True, return_tensors="pt")
        tf = model.get_text_features(**t)
        if not isinstance(tf, torch.Tensor):
            tf = tf.pooler_output
        tf = tf / tf.norm(dim=-1, keepdim=True)
    img = torch.from_numpy(embed_images(pil_images, batch_size, encoder="clip"))
    s = (img @ tf.T).numpy()
    return s[:, 0] - s[:, 1]
