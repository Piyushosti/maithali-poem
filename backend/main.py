# import os
# import time
# from fastapi import FastAPI
# from fastapi.staticfiles import StaticFiles
# from fastapi.middleware.cors import CORSMiddleware
# from gtts import gTTS
# import torch
# import argparse 
# import torch.nn as nn
# import torch.nn.functional as F
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from tokenizers import ByteLevelBPETokenizer
# from deep_translator import GoogleTranslator
# import unicodedata

# #  Define the app variable
# app = FastAPI()

# # Create a folder for audio files
# os.makedirs("static/audio", exist_ok=True)
# app.mount("/static", StaticFiles(directory="static"), name="static")


# # Security fix for PyTorch 2.6+
# torch.serialization.add_safe_globals([argparse.Namespace])

# # --- 1. MODEL CONFIGURATION ---
# class Config:
#     vocab_size   = 8000
#     block_size   = 512
#     n_layer      = 8
#     n_head       = 8
#     n_embd       = 512
#     ffn_mult     = 2.67
#     dropout      = 0.0
#     rope_base    = 10000

# # --- 2. MODEL ARCHITECTURE (The Skeleton) ---
# class RMSNorm(nn.Module):
#     def __init__(self, dim, eps=1e-6):
#         super().__init__()
#         self.eps = eps
#         self.weight = nn.Parameter(torch.ones(dim))
#     def forward(self, x):
#         norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
#         return (x.float() * norm).type_as(x) * self.weight

# def precompute_freqs_cis(dim, max_seq_len, base=10000):
#     freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
#     t = torch.arange(max_seq_len)
#     freqs = torch.outer(t, freqs)
#     return torch.polar(torch.ones_like(freqs), freqs)

# def apply_rope(xq, xk, freqs_cis):
#     xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
#     xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
#     freqs_cis = freqs_cis[:xq_.shape[1]].unsqueeze(0).unsqueeze(2)
#     xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
#     xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
#     return xq_out.type_as(xq), xk_out.type_as(xk)

# class SwiGLU(nn.Module):
#     def __init__(self, dim, hidden_dim):
#         super().__init__()
#         self.gate = nn.Linear(dim, hidden_dim, bias=False)
#         self.up   = nn.Linear(dim, hidden_dim, bias=False)
#         self.down = nn.Linear(hidden_dim, dim, bias=False)
#     def forward(self, x):
#         return self.down(F.silu(self.gate(x)) * self.up(x))

# class Attention(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         self.n_head = cfg.n_head
#         self.head_dim = cfg.n_embd // cfg.n_head
#         self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
#         self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
#     def forward(self, x, freqs_cis):
#         B, T, C = x.shape
#         q, k, v = self.qkv(x).split(C, dim=2)
#         q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
#         k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
#         v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
#         q, k = apply_rope(q.transpose(1,2), k.transpose(1,2), freqs_cis)
#         q = q.transpose(1,2); k = k.transpose(1,2)
#         y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
#         y = y.transpose(1, 2).contiguous().view(B, T, C)
#         return self.proj(y)

# class Block(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         hidden = int(cfg.ffn_mult * cfg.n_embd)
#         self.norm1 = RMSNorm(cfg.n_embd)
#         self.attn  = Attention(cfg)
#         self.norm2 = RMSNorm(cfg.n_embd)
#         self.ffn   = SwiGLU(cfg.n_embd, hidden)
#     def forward(self, x, freqs_cis):
#         x = x + self.attn(self.norm1(x), freqs_cis)
#         x = x + self.ffn(self.norm2(x))
#         return x

# class MaithiliNano(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         self.cfg = cfg
#         self.embed = nn.Embedding(cfg.vocab_size, cfg.n_embd)
#         self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
#         self.norm   = RMSNorm(cfg.n_embd)
#         self.head   = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
#         self.embed.weight = self.head.weight
#         self.register_buffer("freqs_cis", precompute_freqs_cis(cfg.n_embd // cfg.n_head, cfg.block_size * 2, cfg.rope_base))

#     def forward(self, idx):
#         B, T = idx.shape
#         x = self.embed(idx)
#         freqs = self.freqs_cis[:T]
#         for block in self.blocks:
#             x = block(x, freqs)
#         return self.head(self.norm(x))

#     @torch.no_grad()
#     def generate(self, idx, max_new_tokens=150, temperature=0.8, top_k=50):
#         for _ in range(max_new_tokens):
#             idx_cond = idx[:, -self.cfg.block_size:]
#             logits = self(idx_cond)
#             logits = logits[:, -1, :] / temperature
#             if top_k:
#                 v, _ = torch.topk(logits, top_k)
#                 logits[logits < v[:, [-1]]] = float('-inf')
#             probs = F.softmax(logits, dim=-1)
#             next_tok = torch.multinomial(probs, 1)
#             idx = torch.cat([idx, next_tok], dim=1)
#             if next_tok.item() == 2: # Stop at </s>
#                 break
#         return idx

# # --- 3. INITIALIZATION & LOADING ---
# app = FastAPI()
# app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# device = "cuda" if torch.cuda.is_available() else "cpu"
# cfg = Config()

# # Initialize and Load AI Model
# model = MaithiliNano(cfg)
# model.load_state_dict(torch.load("./model/finetuned_final.pt", map_location=device))
# model.to(device)
# model.eval()

# # Load Tokenizer
# tokenizer = ByteLevelBPETokenizer("./model/vocab.json", "./model/merges.txt")
# tokenizer.add_special_tokens(["<pad>", "<s>", "</s>", "<unk>"])

# # Meaning-based Translator (English/Nepali -> Maithili)
# translator = GoogleTranslator(source='auto', target='maithili')

# class PromptRequest(BaseModel):
#     word: str

# # --- 4. THE API ROUTE ---
# @app.post("/generate")
# async def generate(req: PromptRequest):
#     try:
#         # 1. Translate Meaning (e.g., "house" -> "घर")
#         maithili_word = translator.translate(req.word)
        
#         # 2. Process for Model
#         bos = tokenizer.token_to_id("<s>")
#         clean_input = unicodedata.normalize('NFKC', maithili_word)
#         enc = tokenizer.encode(clean_input)
#         idx = torch.tensor([[bos] + enc.ids], dtype=torch.long, device=device)
        
#         # 3. Generate Poem
#         output_ids = model.generate(idx)
#         poem = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)

#         # Generate Audio
#         tts = gTTS(text=poem, lang='hi') # Using 'hi' for Devanagari support
#         audio_filename = f"{int(time.time())}.mp3"
#         audio_path = f"static/audio/{audio_filename}"
#         tts.save(audio_path)

#         return {
#             "maithili_input": maithili_word,
#             "poem": poem,
#             "audio_url": f"http://127.0.0.1:8000/static/audio/{audio_filename}"
#         }
#     except Exception as e:
#         return {"error": str(e)}

import os
import time
import argparse
import unicodedata

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from gtts import gTTS
from pydantic import BaseModel
from tokenizers import ByteLevelBPETokenizer
from deep_translator import GoogleTranslator

# ─────────────────────────────────────────────
# 1. APP SETUP  (only one FastAPI instance!)
# ─────────────────────────────────────────────
app = FastAPI(title="Maithili Nano Poet API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static/audio", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ─────────────────────────────────────────────
# 2. MODEL CONFIGURATION
# ─────────────────────────────────────────────
class Config:
    vocab_size = 8000
    block_size = 512
    n_layer    = 8
    n_head     = 8
    n_embd     = 512
    ffn_mult   = 2.67
    dropout    = 0.0
    rope_base  = 10000


# ─────────────────────────────────────────────
# 3. MODEL ARCHITECTURE
# ─────────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def precompute_freqs_cis(dim, max_seq_len, base=10000):
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rope(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[: xq_.shape[1]].unsqueeze(0).unsqueeze(2)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up   = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head   = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv  = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x, freqs_cis):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q.transpose(1, 2), k.transpose(1, 2), freqs_cis)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden     = int(cfg.ffn_mult * cfg.n_embd)
        self.norm1 = RMSNorm(cfg.n_embd)
        self.attn  = Attention(cfg)
        self.norm2 = RMSNorm(cfg.n_embd)
        self.ffn   = SwiGLU(cfg.n_embd, hidden)

    def forward(self, x, freqs_cis):
        x = x + self.attn(self.norm1(x), freqs_cis)
        x = x + self.ffn(self.norm2(x))
        return x


class MaithiliNano(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg   = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm  = RMSNorm(cfg.n_embd)
        self.head  = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.embed.weight = self.head.weight
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                cfg.n_embd // cfg.n_head,
                cfg.block_size * 2,
                cfg.rope_base,
            ),
        )

    def forward(self, idx):
        B, T = idx.shape
        x = self.embed(idx)
        freqs = self.freqs_cis[:T]
        for block in self.blocks:
            x = block(x, freqs)
        return self.head(self.norm(x))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=150, temperature=0.8, top_k=50):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits   = self(idx_cond)
            logits   = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            probs    = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            idx      = torch.cat([idx, next_tok], dim=1)
            if next_tok.item() == 2:   # </s> token
                break
        return idx


# ─────────────────────────────────────────────
# 4. LOAD MODEL & TOKENIZER  (once at startup)
# ─────────────────────────────────────────────
torch.serialization.add_safe_globals([argparse.Namespace])
device = "cuda" if torch.cuda.is_available() else "cpu"

cfg   = Config()
model = MaithiliNano(cfg)
model.load_state_dict(
    torch.load("./model/finetuned_final.pt", map_location=device)
)
model.to(device).eval()
print(f"✅ Model loaded on {device}")

tokenizer = ByteLevelBPETokenizer("./model/vocab.json", "./model/merges.txt")
tokenizer.add_special_tokens(["<pad>", "<s>", "</s>", "<unk>"])
print("✅ Tokenizer loaded")

translator = GoogleTranslator(source="auto", target="maithili")


# ─────────────────────────────────────────────
# 5. REQUEST / RESPONSE SCHEMAS
# ─────────────────────────────────────────────
class PromptRequest(BaseModel):
    word: str


# ─────────────────────────────────────────────
# 6. API ROUTES
# ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "device": device}


@app.post("/generate")
async def generate(req: PromptRequest, request: Request):
    if not req.word.strip():
        return JSONResponse({"error": "Input word cannot be empty."}, status_code=400)

    try:
        # Step 1 — Translate input to Maithili / Devanagari
        maithili_word = translator.translate(req.word.strip())

        # Step 2 — Encode and feed to model
        bos         = tokenizer.token_to_id("<s>")
        clean_input = unicodedata.normalize("NFKC", maithili_word)
        enc         = tokenizer.encode(clean_input)
        idx         = torch.tensor([[bos] + enc.ids], dtype=torch.long, device=device)

        # Step 3 — Generate poem
        output_ids = model.generate(idx)
        poem       = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)

        # Step 4 — Text-to-speech  (Hindi TTS covers Devanagari script)
        audio_filename = f"{int(time.time())}.mp3"
        audio_path     = f"static/audio/{audio_filename}"
        tts = gTTS(text=poem, lang="hi")
        tts.save(audio_path)

        # Step 5 — Build absolute audio URL from the incoming request
        base_url  = str(request.base_url).rstrip("/")
        audio_url = f"{base_url}/static/audio/{audio_filename}"

        # Step 6 — Simple cultural note (extend as needed)
        cultural_note = (
            f"The word '{maithili_word}' holds deep roots in Maithili culture, "
            "spoken across the Mithila region of Nepal and Bihar, India — "
            "a language celebrated through the poetry of Vidyapati."
        )

        return {
            "maithili_input": maithili_word,
            "poem":           poem,
            "audio_url":      audio_url,
            "cultural_note":  cultural_note,
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)