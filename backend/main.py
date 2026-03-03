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


#2nd vesion with cultural note and dynamic audio URL

# import os
# import time
# import argparse
# import unicodedata
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from fastapi import FastAPI, Request
# from fastapi.staticfiles import StaticFiles
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from gtts import gTTS
# from pydantic import BaseModel
# from tokenizers import ByteLevelBPETokenizer
# from deep_translator import GoogleTranslator

# # ─────────────────────────────────────────────
# # 1. APP SETUP  (only one FastAPI instance!)
# # ─────────────────────────────────────────────
# app = FastAPI(title="Maithili Nano Poet API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# os.makedirs("static/audio", exist_ok=True)
# app.mount("/static", StaticFiles(directory="static"), name="static")

# # ─────────────────────────────────────────────
# # 2. MODEL CONFIGURATION
# # ─────────────────────────────────────────────
# class Config:
#     vocab_size = 8000
#     block_size = 512
#     n_layer    = 8
#     n_head     = 8
#     n_embd     = 512
#     ffn_mult   = 2.67
#     dropout    = 0.0
#     rope_base  = 10000


# # ─────────────────────────────────────────────
# # 3. MODEL ARCHITECTURE
# # ─────────────────────────────────────────────
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
#     freqs_cis = freqs_cis[: xq_.shape[1]].unsqueeze(0).unsqueeze(2)
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
#         self.n_head   = cfg.n_head
#         self.head_dim = cfg.n_embd // cfg.n_head
#         self.qkv  = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
#         self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

#     def forward(self, x, freqs_cis):
#         B, T, C = x.shape
#         q, k, v = self.qkv(x).split(C, dim=2)
#         q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
#         k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
#         v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
#         q, k = apply_rope(q.transpose(1, 2), k.transpose(1, 2), freqs_cis)
#         q = q.transpose(1, 2)
#         k = k.transpose(1, 2)
#         y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
#         y = y.transpose(1, 2).contiguous().view(B, T, C)
#         return self.proj(y)


# class Block(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         hidden     = int(cfg.ffn_mult * cfg.n_embd)
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
#         self.cfg   = cfg
#         self.embed = nn.Embedding(cfg.vocab_size, cfg.n_embd)
#         self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
#         self.norm  = RMSNorm(cfg.n_embd)
#         self.head  = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
#         self.embed.weight = self.head.weight
#         self.register_buffer(
#             "freqs_cis",
#             precompute_freqs_cis(
#                 cfg.n_embd // cfg.n_head,
#                 cfg.block_size * 2,
#                 cfg.rope_base,
#             ),
#         )

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
#             idx_cond = idx[:, -self.cfg.block_size :]
#             logits   = self(idx_cond)
#             logits   = logits[:, -1, :] / temperature
#             if top_k:
#                 v, _ = torch.topk(logits, top_k)
#                 logits[logits < v[:, [-1]]] = float("-inf")
#             probs    = F.softmax(logits, dim=-1)
#             next_tok = torch.multinomial(probs, 1)
#             idx      = torch.cat([idx, next_tok], dim=1)
#             if next_tok.item() == 2:   # </s> token
#                 break
#         return idx


# # ─────────────────────────────────────────────
# # 4. LOAD MODEL & TOKENIZER  (once at startup)
# # ─────────────────────────────────────────────
# torch.serialization.add_safe_globals([argparse.Namespace])
# device = "cuda" if torch.cuda.is_available() else "cpu"

# cfg   = Config()
# model = MaithiliNano(cfg)
# model.load_state_dict(
#     torch.load("./model/finetuned_final.pt", map_location=device)
# )
# model.to(device).eval()
# print(f"✅ Model loaded on {device}")

# tokenizer = ByteLevelBPETokenizer("./model/vocab.json", "./model/merges.txt")
# tokenizer.add_special_tokens(["<pad>", "<s>", "</s>", "<unk>"])
# print("✅ Tokenizer loaded")

# translator = GoogleTranslator(source="auto", target="maithili")


# # ─────────────────────────────────────────────
# # 5. REQUEST / RESPONSE SCHEMAS
# # ─────────────────────────────────────────────
# class PromptRequest(BaseModel):
#     word: str


# # ─────────────────────────────────────────────
# # 6. API ROUTES
# # ─────────────────────────────────────────────
# @app.get("/health")
# def health():
#     return {"status": "ok", "device": device}


# @app.post("/generate")
# async def generate(req: PromptRequest, request: Request):
#     if not req.word.strip():
#         return JSONResponse({"error": "Input word cannot be empty."}, status_code=400)

#     try:
#         # Step 1 — Translate input to Maithili / Devanagari
#         maithili_word = translator.translate(req.word.strip())

#         # Step 2 — Encode and feed to model
#         bos         = tokenizer.token_to_id("<s>")
#         clean_input = unicodedata.normalize("NFKC", maithili_word)
#         enc         = tokenizer.encode(clean_input)
#         idx         = torch.tensor([[bos] + enc.ids], dtype=torch.long, device=device)

#         # Step 3 — Generate poem
#         output_ids = model.generate(idx)
#         poem       = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)

#         # Step 4 — Text-to-speech  (Hindi TTS covers Devanagari script)
#         audio_filename = f"{int(time.time())}.mp3"
#         audio_path     = f"static/audio/{audio_filename}"
#         tts = gTTS(text=poem, lang="hi")
#         tts.save(audio_path)

#         # Step 5 — Build absolute audio URL from the incoming request
#         base_url  = str(request.base_url).rstrip("/")
#         audio_url = f"{base_url}/static/audio/{audio_filename}"

#         # Step 6 — Simple cultural note (extend as needed)
#         cultural_note = (
#             f"The word '{maithili_word}' holds deep roots in Maithili culture, "
#             "spoken across the Mithila region of Nepal and Bihar, India — "
#             "a language celebrated through the poetry of Vidyapati."
#         )

#         return {
#             "maithili_input": maithili_word,
#             "poem":           poem,
#             "audio_url":      audio_url,
#             "cultural_note":  cultural_note,
#         }

#     except Exception as e:
#         return JSONResponse({"error": str(e)}, status_code=500)



#3rd version
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
# 1. APP SETUP
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
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def precompute_freqs_cis(dim, max_seq_len, base=10000):
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t     = torch.arange(max_seq_len)
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
        self.cfg    = cfg
        self.embed  = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm   = RMSNorm(cfg.n_embd)
        self.head   = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
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
        x     = self.embed(idx)
        freqs = self.freqs_cis[:T]
        for block in self.blocks:
            x = block(x, freqs)
        return self.head(self.norm(x))

    # ── IMPROVED GENERATE: top-k + top-p (nucleus) sampling ──────────────
    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens: int   = 150,
        temperature:    float = 0.75,   # lower = more focused & accurate
        top_k:          int   = 40,     # tightened from 50
        top_p:          float = 0.92,   # nucleus sampling for coherence
    ):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits   = self(idx_cond)
            logits   = logits[:, -1, :] / temperature

            # — top-k filter —
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)

            # — top-p (nucleus) filter —
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cumulative         = torch.cumsum(sorted_probs, dim=-1)
            # drop tokens that push the cumulative probability over top_p
            sorted_probs[cumulative - sorted_probs > top_p] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)   # renormalize
            
            sample     = torch.multinomial(sorted_probs, num_samples=1)  # (1, 1)
            next_tok   = sorted_idx.gather(dim=-1, index=sample)         # (1, 1)  ✅ 2-D


            idx = torch.cat([idx, next_tok], dim=1)
            if next_tok.item() == 2:    # </s> token → stop
                break

        return idx


# ─────────────────────────────────────────────
# 4. SMART TRANSLATION  (fallback chain)
#    Maithili → Nepali → Hindi → original
# ─────────────────────────────────────────────
def translate_to_maithili(text: str) -> tuple[str, str]:
    """
    Returns (translated_word, language_used).
    Google Translate's Maithili support is inconsistent, so we try
    Nepali and Hindi as Devanagari-script fallbacks.
    """
    attempts = [
        ("maithili", "maithili"),
        ("ne",       "nepali"),
        ("hi",       "hindi"),
    ]
    for target_code, label in attempts:
        try:
            result = GoogleTranslator(source="auto", target=target_code).translate(text)
            if result and result.strip() and result.strip() != text:
                print(f"🌐 Translation via {label}: '{text}' → '{result.strip()}'")
                return result.strip(), label
        except Exception as e:
            print(f"⚠️  {label} translation failed: {e}")
            continue

    # Absolute last resort — return original
    print(f"⚠️  All translations failed, using original: '{text}'")
    return text, "original"


# ─────────────────────────────────────────────
# 5. LOAD MODEL & TOKENIZER  (once at startup)
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


# ─────────────────────────────────────────────
# 6. CULTURAL NOTES  (extend this dict freely)
# ─────────────────────────────────────────────
CULTURAL_NOTES: dict[str, str] = {
    "घर":    "In Maithili tradition, the home (घर) is a sacred space — a symbol of family, ancestors, and the warmth of Mithila's artistic heritage, including the famous Madhubani paintings on its walls.",
    "नदी":   "Rivers hold deep significance in Mithila. The Kamala, Koshi, and Bagmati are not just waterways — they are mothers, celebrated in folk songs and seasonal rituals.",
    "फूल":   "Flowers (फूल) are central to Maithili worship, poetry, and the vibrant Madhubani art tradition that depicts nature in bold, intricate patterns.",
    "चाँद":  "The moon (चाँद) is a recurring symbol in Maithili poetry, especially in the works of the legendary poet Vidyapati, where it stands as a metaphor for beauty and longing.",
    "प्रेम": "Love (प्रेम) in Maithili literature is channelled through Vidyapati's devotional verses to Radha and Krishna — poetry that defined an entire era of Indic literary culture.",
    "पानी":  "Water (पानी) is life in Mithila — the rivers, rains, and flooded fields of the Terai are woven into every aspect of its agriculture, festivals, and folk memory.",
    "आकाश": "The sky (आकाश) in Maithili poetry is a canvas for the divine — stars, clouds, and the monsoon are read as messages from the gods in Mithila's rich oral tradition.",
    "माँ":   "The mother (माँ) is the most revered figure in Maithili culture. The goddess Durga is worshipped as the universal mother, and maternal love is a cornerstone of Maithili folk songs.",
}

def get_cultural_note(maithili_word: str, original_word: str, lang_used: str) -> str:
    note = CULTURAL_NOTES.get(maithili_word.strip())
    if note:
        return note

    lang_msg = {
        "maithili": "translated directly into Maithili",
        "nepali":   "rendered in Nepali Devanagari as a close cultural approximation",
        "hindi":    "rendered in Hindi Devanagari as a script-compatible fallback",
        "original": "used as-is since automatic translation was unavailable",
    }.get(lang_used, "translated")

    return (
        f"The concept of '{original_word}' was {lang_msg} as '{maithili_word}'. "
        "Maithili is a language of the Mithila region spanning Nepal and Bihar, India — "
        "kept alive through the timeless poetry of Vidyapati and the vivid colours "
        "of Madhubani art."
    )


# ─────────────────────────────────────────────
# 7. REQUEST SCHEMA
# ─────────────────────────────────────────────
class PromptRequest(BaseModel):
    word:        str
    temperature: float = 0.75   # frontend can override these for experimentation
    top_k:       int   = 40
    top_p:       float = 0.92


# ─────────────────────────────────────────────
# 8. API ROUTES
# ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "device": device}


@app.post("/generate")
async def generate(req: PromptRequest, request: Request):
    if not req.word.strip():
        return JSONResponse({"error": "Input word cannot be empty."}, status_code=400)

    try:
        original_word = req.word.strip()

        # ── Step 1 : Smart translation with fallback chain ────────────────
        maithili_word, lang_used = translate_to_maithili(original_word)

        # ── Step 2 : Structured prompt  ───────────────────────────────────
        # Format:  <s>कविता: <keyword>\n
        # "कविता" = poem in Devanagari.
        # ⚠️  If your training .txt files used a different prefix/header,
        #     update the string below to match that format exactly.
        clean_word        = unicodedata.normalize("NFKC", maithili_word)
        structured_prompt = f"कविता: {clean_word}\n"

        bos = tokenizer.token_to_id("<s>")
        enc = tokenizer.encode(structured_prompt)
        idx = torch.tensor([[bos] + enc.ids], dtype=torch.long, device=device)

        # ── Step 3 : Generate poem ────────────────────────────────────────
        output_ids = model.generate(
            idx,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
        )
        poem = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)

        # Strip the structured prefix so the UI only shows the poem body
        prefix = f"कविता: {clean_word}"
        if poem.startswith(prefix):
            poem = poem[len(prefix):].lstrip("\n").strip()

        # ── Step 4 : Text-to-speech  (Maithili → Hindi fallback) ─────────
        audio_filename = f"{int(time.time())}.mp3"
        audio_path     = f"static/audio/{audio_filename}"

        tts_saved = False
        for lang_code in ["mai", "hi"]:     # try Maithili TTS first
            try:
                gTTS(text=poem, lang=lang_code).save(audio_path)
                tts_saved = True
                print(f"🔊 TTS saved with lang='{lang_code}'")
                break
            except Exception:
                continue

        if not tts_saved:
            open(audio_path, "wb").close()  # empty file so URL doesn't 404

        # ── Step 5 : Build absolute audio URL ────────────────────────────
        base_url  = str(request.base_url).rstrip("/")
        audio_url = f"{base_url}/static/audio/{audio_filename}"

        # ── Step 6 : Cultural note ────────────────────────────────────────
        cultural_note = get_cultural_note(maithili_word, original_word, lang_used)

        return {
            "original_input": original_word,
            "maithili_input": maithili_word,
            "language_used":  lang_used,        # useful for frontend debugging
            "poem":           poem,
            "audio_url":      audio_url,
            "cultural_note":  cultural_note,
        }

    except Exception as e:
        print(f"❌ /generate error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)