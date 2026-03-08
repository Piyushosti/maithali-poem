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
# # 1. APP SETUP
# # ─────────────────────────────────────────────
# app = FastAPI(title="Maithili Nano Poet API - Dual Model (Working)")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# os.makedirs("static/audio", exist_ok=True)
# app.mount("/static", StaticFiles(directory="static"), name="static")


# # ─────────────────────────────────────────────
# # 2. MODEL CONFIGURATION (MaithiliNano)
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
# # 3. MAITHILI NANO ARCHITECTURE
# # ─────────────────────────────────────────────
# class RMSNorm(nn.Module):
#     def __init__(self, dim, eps=1e-6):
#         super().__init__()
#         self.eps    = eps
#         self.weight = nn.Parameter(torch.ones(dim))

#     def forward(self, x):
#         norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
#         return (x.float() * norm).type_as(x) * self.weight


# def precompute_freqs_cis(dim, max_seq_len, base=10000):
#     freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
#     t     = torch.arange(max_seq_len)
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
#         self.cfg    = cfg
#         self.embed  = nn.Embedding(cfg.vocab_size, cfg.n_embd)
#         self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
#         self.norm   = RMSNorm(cfg.n_embd)
#         self.head   = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
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
#         x     = self.embed(idx)
#         freqs = self.freqs_cis[:T]
#         for block in self.blocks:
#             x = block(x, freqs)
#         return self.head(self.norm(x))

#     @torch.no_grad()
#     def generate(
#         self,
#         idx,
#         max_new_tokens: int   = 150,
#         temperature:    float = 0.75,
#         top_k:          int   = 40,
#         top_p:          float = 0.92,
#     ):
#         for _ in range(max_new_tokens):
#             idx_cond = idx[:, -self.cfg.block_size:]
#             logits   = self(idx_cond)
#             logits   = logits[:, -1, :] / temperature

#             if top_k:
#                 v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
#                 logits[logits < v[:, [-1]]] = float("-inf")

#             probs = F.softmax(logits, dim=-1)

#             sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
#             cumulative = torch.cumsum(sorted_probs, dim=-1)
#             sorted_probs[cumulative - sorted_probs > top_p] = 0.0
#             sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            
#             sample     = torch.multinomial(sorted_probs, num_samples=1)
#             next_tok   = sorted_idx.gather(dim=-1, index=sample)

#             idx = torch.cat([idx, next_tok], dim=1)
#             if next_tok.item() == 2:
#                 break

#         return idx


# # ─────────────────────────────────────────────
# # 4. SMART TRANSLATION (fallback chain)
# # ─────────────────────────────────────────────
# def translate_to_maithili(text: str) -> tuple[str, str]:
#     """
#     Returns (translated_word, language_used).
#     """
#     attempts = [
#         ("maithili", "maithili"),
#         ("ne",       "nepali"),
#         ("hi",       "hindi"),
#     ]
#     for target_code, label in attempts:
#         try:
#             result = GoogleTranslator(source="auto", target=target_code).translate(text)
#             if result and result.strip() and result.strip() != text:
#                 print(f"🌐 Translation via {label}: '{text}' → '{result.strip()}'")
#                 return result.strip(), label
#         except Exception as e:
#             print(f"⚠️  {label} translation failed: {e}")
#             continue

#     print(f"⚠️  All translations failed, using original: '{text}'")
#     return text, "original"


# # ─────────────────────────────────────────────
# # 5. LOAD MODELS AT STARTUP
# # ─────────────────────────────────────────────
# torch.serialization.add_safe_globals([argparse.Namespace])

# # Determine device
# if torch.cuda.is_available():
#     device = "cuda"
#     device_name = "NVIDIA GPU"
# elif torch.backends.mps.is_available():
#     device = "mps"
#     device_name = "Apple Silicon (MPS)"
# else:
#     device = "cpu"
#     device_name = "CPU"

# print(f"🖥️  Using device: {device} ({device_name})")

# # Load MaithiliNano
# cfg   = Config()
# model_nano = MaithiliNano(cfg)
# model_nano.load_state_dict(
#     torch.load("./model/finetuned_final.pt", map_location=device)
# )
# model_nano.to(device).eval()
# print(f"✅ MaithiliNano loaded on {device}")

# tokenizer_nano = ByteLevelBPETokenizer("./model/vocab.json", "./model/merges.txt")
# tokenizer_nano.add_special_tokens(["<pad>", "<s>", "</s>", "<unk>"])
# print("✅ MaithiliNano tokenizer loaded")

# # Load Gemma 2B with LoRA adapter - SIMPLIFIED VERSION
# model_gemma = None
# tokenizer_gemma = None

# try:
#     print("\n🔄 Loading Gemma 2B base model with LoRA adapter...")
    
#     from transformers import AutoTokenizer, AutoModelForCausalLM
#     from peft import PeftModel
    
#     # Try loading the base model with compatibility settings
#     print("  → Loading base model (google/gemma-2-2b-it)...")
#     try:
#         base_model = AutoModelForCausalLM.from_pretrained(
#             "google/gemma-2-2b-it",
#             device_map=device,
#             torch_dtype=torch.float32,
#             low_cpu_mem_usage=True,
#             trust_remote_code=True,  # Added for compatibility
#             attn_implementation="eager"  # Use eager attention instead of flash_attn
#         )
#     except TypeError:
#         # Fallback if attn_implementation not supported
#         print("  → (Retrying without attn_implementation...)")
#         base_model = AutoModelForCausalLM.from_pretrained(
#             "google/gemma-2-2b-it",
#             device_map=device,
#             torch_dtype=torch.float32,
#             low_cpu_mem_usage=True,
#             trust_remote_code=True
#         )
    
#     print("  → Base model loaded")
    
#     # Load tokenizer
#     print("  → Loading tokenizer...")
#     tokenizer_gemma = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
#     print("  → Tokenizer loaded")
    
#     # Load LoRA adapter
#     print("  → Loading LoRA adapter from ./model/gemma_2b...")
#     model_gemma = PeftModel.from_pretrained(
#         base_model,
#         "./model/gemma_2b"
#     )
#     print("  → LoRA adapter loaded")
    
#     # Merge adapter into base model for inference
#     print("  → Merging adapter into base model...")
#     model_gemma = model_gemma.merge_and_unload()
#     model_gemma.to(device).eval()
#     print("✅ Gemma 2B with LoRA adapter loaded successfully!")
    
# except ImportError as e:
#     print(f"⚠️  Import error: {e}")
#     print("⚠️  Make sure you have installed: pip install peft")
#     model_gemma = None
#     tokenizer_gemma = None
# except Exception as e:
#     print(f"⚠️  Gemma 2B failed to load: {type(e).__name__}: {e}")
#     print("⚠️  Only MaithiliNano will be available")
#     model_gemma = None
#     tokenizer_gemma = None


# # ─────────────────────────────────────────────
# # 6. GENERATION FUNCTIONS
# # ─────────────────────────────────────────────
# def generate_with_nano(
#     maithili_word: str,
#     tokenizer,
#     model,
#     device
# ) -> str:
#     """Generate poem using MaithiliNano model."""
#     try:
#         clean_word = unicodedata.normalize("NFKC", maithili_word)
#         structured_prompt = f"कविता: {clean_word}\n"
        
#         bos = tokenizer.token_to_id("<s>")
#         enc = tokenizer.encode(structured_prompt)
#         idx = torch.tensor([[bos] + enc.ids], dtype=torch.long, device=device)
        
#         output_ids = model.generate(
#             idx,
#             temperature=0.75,
#             top_k=40,
#             top_p=0.92,
#         )
#         poem = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
        
#         # Strip prefix
#         prefix = f"कविता: {clean_word}"
#         if poem.startswith(prefix):
#             poem = poem[len(prefix):].lstrip("\n").strip()
        
#         return poem
#     except Exception as e:
#         print(f"❌ MaithiliNano generation error: {e}")
#         raise


# def generate_with_gemma(
#     word: str,
#     tokenizer,
#     model,
#     device
# ) -> str:
#     """Generate poem using fine-tuned Gemma 2B model with LoRA."""
#     try:
#         prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

# ### Instruction:
# Write a Maithili poem based on the following title or theme.

# ### Input:
# {word}

# ### Response:
# """
        
#         inputs = tokenizer([prompt], return_tensors="pt").to(device)
        
#         with torch.no_grad():
#             outputs = model.generate(
#                 **inputs,
#                 max_new_tokens=200,
#                 temperature=0.5,
#                 repetition_penalty=1.2,
#                 top_k=40,
#                 top_p=0.9,
#                 do_sample=True
#             )
        
#         result = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
#         poem = result.split("### Response:")[-1].strip()
#         return poem
#     except Exception as e:
#         print(f"❌ Gemma generation error: {e}")
#         raise


# # ─────────────────────────────────────────────
# # 7. CULTURAL NOTES
# # ─────────────────────────────────────────────
# CULTURAL_NOTES: dict[str, str] = {
#     "घर":    "In Maithili tradition, the home (घर) is a sacred space — a symbol of family, ancestors, and the warmth of Mithila's artistic heritage, including the famous Madhubani paintings on its walls.",
#     "नदी":   "Rivers hold deep significance in Mithila. The Kamala, Koshi, and Bagmati are not just waterways — they are mothers, celebrated in folk songs and seasonal rituals.",
#     "फूल":   "Flowers (फूल) are central to Maithili worship, poetry, and the vibrant Madhubani art tradition that depicts nature in bold, intricate patterns.",
#     "चाँद":  "The moon (चाँद) is a recurring symbol in Maithili poetry, especially in the works of the legendary poet Vidyapati, where it stands as a metaphor for beauty and longing.",
#     "प्रेम": "Love (प्रेम) in Maithili literature is channelled through Vidyapati's devotional verses to Radha and Krishna — poetry that defined an entire era of Indic literary culture.",
#     "पानी":  "Water (पानी) is life in Mithila — the rivers, rains, and flooded fields of the Terai are woven into every aspect of its agriculture, festivals, and folk memory.",
#     "आकाश": "The sky (आकाश) in Maithili poetry is a canvas for the divine — stars, clouds, and the monsoon are read as messages from the gods in Mithila's rich oral tradition.",
#     "माँ":   "The mother (माँ) is the most revered figure in Maithili culture. The goddess Durga is worshipped as the universal mother, and maternal love is a cornerstone of Maithili folk songs.",
# }

# def get_cultural_note(maithili_word: str, original_word: str, lang_used: str) -> str:
#     note = CULTURAL_NOTES.get(maithili_word.strip())
#     if note:
#         return note

#     lang_msg = {
#         "maithili": "translated directly into Maithili",
#         "nepali":   "rendered in Nepali Devanagari as a close cultural approximation",
#         "hindi":    "rendered in Hindi Devanagari as a script-compatible fallback",
#         "original": "used as-is since automatic translation was unavailable",
#     }.get(lang_used, "translated")

#     return (
#         f"The concept of '{original_word}' was {lang_msg} as '{maithili_word}'. "
#         "Maithili is a language of the Mithila region spanning Nepal and Bihar, India — "
#         "kept alive through the timeless poetry of Vidyapati and the vivid colours "
#         "of Madhubani art."
#     )


# # ─────────────────────────────────────────────
# # 8. REQUEST SCHEMA
# # ─────────────────────────────────────────────
# class PromptRequest(BaseModel):
#     word: str
#     models: list[str] = ["nano", "gemma"]
#     temperature: float = 0.75
#     top_k: int = 40
#     top_p: float = 0.92


# # ─────────────────────────────────────────────
# # 9. API ROUTES
# # ─────────────────────────────────────────────
# @app.get("/health")
# def health():
#     return {
#         "status": "ok",
#         "device": device,
#         "models_available": {
#             "nano": True,
#             "gemma": model_gemma is not None
#         }
#     }


# @app.post("/generate")
# async def generate(req: PromptRequest, request: Request):
#     """Generate poems from one or both models."""
#     if not req.word.strip():
#         return JSONResponse({"error": "Input word cannot be empty."}, status_code=400)

#     try:
#         original_word = req.word.strip()

#         # Translate once (shared for both models)
#         maithili_word, lang_used = translate_to_maithili(original_word)
#         clean_word = unicodedata.normalize("NFKC", maithili_word)
        
#         responses = {}
        
#         # Generate with MaithiliNano if requested
#         if "nano" in req.models and model_nano is not None:
#             try:
#                 print(f"\n🎨 Generating with MaithiliNano...")
#                 poem_nano = generate_with_nano(
#                     maithili_word, 
#                     tokenizer_nano, 
#                     model_nano, 
#                     device
#                 )
#                 responses["nano"] = poem_nano
#                 print(f"✅ MaithiliNano poem generated ({len(poem_nano)} chars)")
#             except Exception as e:
#                 print(f"⚠️  MaithiliNano generation failed: {e}")
#                 responses["nano_error"] = str(e)
        
#         # Generate with Gemma if requested
#         if "gemma" in req.models and model_gemma is not None:
#             try:
#                 print(f"\n🤖 Generating with Gemma 2B...")
#                 poem_gemma = generate_with_gemma(
#                     original_word, 
#                     tokenizer_gemma, 
#                     model_gemma, 
#                     device
#                 )
#                 responses["gemma"] = poem_gemma
#                 print(f"✅ Gemma 2B poem generated ({len(poem_gemma)} chars)")
#             except Exception as e:
#                 print(f"⚠️  Gemma generation failed: {e}")
#                 responses["gemma_error"] = str(e)
        
#         # Check if we have at least one valid response
#         if not any(k in responses for k in ["nano", "gemma"]):
#             return JSONResponse(
#                 {"error": "Both models failed to generate poems"},
#                 status_code=500
#             )

#         # Generate audio from first available poem
#         poem_for_audio = responses.get("nano") or responses.get("gemma", "")
#         audio_filename = f"{int(time.time())}.mp3"
#         audio_path = f"static/audio/{audio_filename}"

#         print(f"\n🔊 Generating audio...")
#         tts_saved = False
#         for lang_code in ["mai", "hi"]:
#             try:
#                 gTTS(text=poem_for_audio, lang=lang_code).save(audio_path)
#                 tts_saved = True
#                 print(f"✅ TTS saved with lang='{lang_code}'")
#                 break
#             except Exception as e:
#                 print(f"⚠️  TTS with {lang_code} failed: {e}")
#                 continue

#         if not tts_saved:
#             open(audio_path, "wb").close()
#             print("⚠️  Created empty audio file")

#         base_url = str(request.base_url).rstrip("/")
#         audio_url = f"{base_url}/static/audio/{audio_filename}"

#         cultural_note = get_cultural_note(maithili_word, original_word, lang_used)

#         return {
#             "original_input": original_word,
#             "maithili_input": maithili_word,
#             "language_used": lang_used,
#             "audio_url": audio_url,
#             "cultural_note": cultural_note,
#             "responses": responses,
#             "selected_model": None,
#         }

#     except Exception as e:
#         print(f"❌ /generate error: {e}")
#         return JSONResponse({"error": str(e)}, status_code=500)


# # ─────────────────────────────────────────────
# # 10. STARTUP MESSAGE
# # ─────────────────────────────────────────────
# @app.on_event("startup")
# async def startup_event():
#     print("\n" + "="*70)
#     print("🚀 Maithili Nano Poet API - Dual Model Edition (WORKING)")
#     print("="*70)
#     print(f"Device: {device} ({device_name})")
#     print(f"MaithiliNano: ✅ Ready")
#     print(f"Gemma 2B with LoRA: {'✅ Ready' if model_gemma else '⚠️  Not loaded'}")
#     print("="*70 + "\n")




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
app = FastAPI(title="Maithili Nano Poet API - Dual Model (CPU Fixed)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static/audio", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ─────────────────────────────────────────────
# 2. MODEL CONFIGURATION (MaithiliNano)
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
# 3. MAITHILI NANO ARCHITECTURE
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

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens: int   = 150,
        temperature:    float = 0.75,
        top_k:          int   = 40,
        top_p:          float = 0.92,
    ):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits   = self(idx_cond)
            logits   = logits[:, -1, :] / temperature

            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)

            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            sorted_probs[cumulative - sorted_probs > top_p] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            
            sample     = torch.multinomial(sorted_probs, num_samples=1)
            next_tok   = sorted_idx.gather(dim=-1, index=sample)

            idx = torch.cat([idx, next_tok], dim=1)
            if next_tok.item() == 2:
                break

        return idx


# ─────────────────────────────────────────────
# 4. SMART TRANSLATION (fallback chain)
# ─────────────────────────────────────────────
def translate_to_maithili(text: str) -> tuple[str, str]:
    """
    Returns (translated_word, language_used).
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

    print(f"⚠️  All translations failed, using original: '{text}'")
    return text, "original"


# ─────────────────────────────────────────────
# 5. LOAD MODELS AT STARTUP
# ─────────────────────────────────────────────
torch.serialization.add_safe_globals([argparse.Namespace])

# FORCE CPU DEVICE - Apple Silicon (MPS) runs out of VRAM
device = "cpu"
device_name = "CPU (Apple Silicon compatible)"

print(f"🖥️  Using device: {device} ({device_name})")
print("⚠️  Note: Using CPU for better VRAM management on Apple Silicon")

# Load MaithiliNano
cfg   = Config()
model_nano = MaithiliNano(cfg)
model_nano.load_state_dict(
    torch.load("./model/finetuned_final.pt", map_location=device)
)
model_nano.to(device).eval()
print(f"✅ MaithiliNano loaded on {device}")

tokenizer_nano = ByteLevelBPETokenizer("./model/vocab.json", "./model/merges.txt")
tokenizer_nano.add_special_tokens(["<pad>", "<s>", "</s>", "<unk>"])
print("✅ MaithiliNano tokenizer loaded")

# Load Gemma 2B with LoRA adapter - FIXED FOR APPLE SILICON
model_gemma = None
tokenizer_gemma = None

try:
    print("\n🔄 Loading Gemma 2B base model with LoRA adapter...")
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    
    # Load base Gemma 2B model ON CPU with int8 quantization
    print("  → Loading base model (google/gemma-2-2b-it)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b-it",
        device_map="cpu",  # FORCE CPU
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager"
    )
    print("  → Base model loaded")
    
    # Load tokenizer
    print("  → Loading tokenizer...")
    tokenizer_gemma = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    print("  → Tokenizer loaded")
    
    # Load LoRA adapter
    print("  → Loading LoRA adapter from ./model/gemma_2b...")
    model_gemma = PeftModel.from_pretrained(
        base_model,
        "./model/gemma_2b"
    )
    print("  → LoRA adapter loaded")
    
    # Merge adapter into base model for inference
    print("  → Merging adapter into base model...")
    model_gemma = model_gemma.merge_and_unload()
    model_gemma.to(device).eval()
    print("✅ Gemma 2B with LoRA adapter loaded successfully!")
    
except Exception as e:
    print(f"⚠️  Gemma 2B failed to load: {e}")
    print("⚠️  Only MaithiliNano will be available")
    model_gemma = None
    tokenizer_gemma = None


# ─────────────────────────────────────────────
# 6. GENERATION FUNCTIONS
# ─────────────────────────────────────────────
def generate_with_nano(
    maithili_word: str,
    tokenizer,
    model,
    device
) -> str:
    """Generate poem using MaithiliNano model."""
    try:
        clean_word = unicodedata.normalize("NFKC", maithili_word)
        structured_prompt = f"कविता: {clean_word}\n"
        
        bos = tokenizer.token_to_id("<s>")
        enc = tokenizer.encode(structured_prompt)
        idx = torch.tensor([[bos] + enc.ids], dtype=torch.long, device=device)
        
        output_ids = model.generate(
            idx,
            temperature=0.75,
            top_k=40,
            top_p=0.92,
        )
        poem = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
        
        # Strip prefix
        prefix = f"कविता: {clean_word}"
        if poem.startswith(prefix):
            poem = poem[len(prefix):].lstrip("\n").strip()
        
        return poem
    except Exception as e:
        print(f"❌ MaithiliNano generation error: {e}")
        raise


def generate_with_gemma(
    word: str,
    tokenizer,
    model,
    device
) -> str:
    """Generate poem using fine-tuned Gemma 2B model with LoRA."""
    try:
        prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Write a Maithili poem based on the following title or theme.

### Input:
{word}

### Response:
"""
        
        inputs = tokenizer([prompt], return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.5,
                repetition_penalty=1.2,
                top_k=40,
                top_p=0.9,
                do_sample=True
            )
        
        result = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        poem = result.split("### Response:")[-1].strip()
        return poem
    except Exception as e:
        print(f"❌ Gemma generation error: {e}")
        raise


# ─────────────────────────────────────────────
# 7. CULTURAL NOTES
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
# 8. REQUEST SCHEMA
# ─────────────────────────────────────────────
class PromptRequest(BaseModel):
    word: str
    models: list[str] = ["nano", "gemma"]
    temperature: float = 0.75
    top_k: int = 40
    top_p: float = 0.92


# ─────────────────────────────────────────────
# 9. API ROUTES
# ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": device,
        "models_available": {
            "nano": True,
            "gemma": model_gemma is not None
        }
    }


@app.post("/generate")
async def generate(req: PromptRequest, request: Request):
    """Generate poems from one or both models."""
    if not req.word.strip():
        return JSONResponse({"error": "Input word cannot be empty."}, status_code=400)

    try:
        original_word = req.word.strip()

        # Translate once (shared for both models)
        maithili_word, lang_used = translate_to_maithili(original_word)
        clean_word = unicodedata.normalize("NFKC", maithili_word)
        
        responses = {}
        
        # Generate with MaithiliNano if requested
        if "nano" in req.models and model_nano is not None:
            try:
                print(f"\n🎨 Generating with MaithiliNano...")
                poem_nano = generate_with_nano(
                    maithili_word, 
                    tokenizer_nano, 
                    model_nano, 
                    device
                )
                responses["nano"] = poem_nano
                print(f"✅ MaithiliNano poem generated ({len(poem_nano)} chars)")
            except Exception as e:
                print(f"⚠️  MaithiliNano generation failed: {e}")
                responses["nano_error"] = str(e)
        
        # Generate with Gemma if requested
        if "gemma" in req.models and model_gemma is not None:
            try:
                print(f"\n🤖 Generating with Gemma 2B...")
                poem_gemma = generate_with_gemma(
                    original_word, 
                    tokenizer_gemma, 
                    model_gemma, 
                    device
                )
                responses["gemma"] = poem_gemma
                print(f"✅ Gemma 2B poem generated ({len(poem_gemma)} chars)")
            except Exception as e:
                print(f"⚠️  Gemma generation failed: {e}")
                responses["gemma_error"] = str(e)
        
        # Check if we have at least one valid response
        if not any(k in responses for k in ["nano", "gemma"]):
            return JSONResponse(
                {"error": "Both models failed to generate poems"},
                status_code=500
            )

        # Generate audio from first available poem
        poem_for_audio = responses.get("nano") or responses.get("gemma", "")
        audio_filename = f"{int(time.time())}.mp3"
        audio_path = f"static/audio/{audio_filename}"

        print(f"\n🔊 Generating audio...")
        tts_saved = False
        for lang_code in ["mai", "hi"]:
            try:
                gTTS(text=poem_for_audio, lang=lang_code).save(audio_path)
                tts_saved = True
                print(f"✅ TTS saved with lang='{lang_code}'")
                break
            except Exception as e:
                print(f"⚠️  TTS with {lang_code} failed: {e}")
                continue

        if not tts_saved:
            open(audio_path, "wb").close()
            print("⚠️  Created empty audio file")

        base_url = str(request.base_url).rstrip("/")
        audio_url = f"{base_url}/static/audio/{audio_filename}"

        cultural_note = get_cultural_note(maithili_word, original_word, lang_used)

        return {
            "original_input": original_word,
            "maithili_input": maithili_word,
            "language_used": lang_used,
            "audio_url": audio_url,
            "cultural_note": cultural_note,
            "responses": responses,
            "selected_model": None,
        }

    except Exception as e:
        print(f"❌ /generate error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# ─────────────────────────────────────────────
# 10. STARTUP MESSAGE
# ─────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    print("\n" + "="*70)
    print("🚀 Maithili Nano Poet API - Dual Model Edition (CPU Fixed)")
    print("="*70)
    print(f"Device: {device} ({device_name})")
    print(f"MaithiliNano: ✅ Ready")
    print(f"Gemma 2B with LoRA: {'✅ Ready' if model_gemma else '⚠️  Not loaded'}")
    print("="*70 + "\n")