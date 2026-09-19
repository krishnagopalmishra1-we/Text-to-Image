import gradio as gr
import torch
import time
from diffusers import (
    AutoPipelineForText2Image,
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    UniPCMultistepScheduler,
    LCMScheduler,
)
from compel import Compel, ReturnedEmbeddingsType
import os
import gc

# ---------------------------------------------------------------------------
# Environment & Auth
# ---------------------------------------------------------------------------
AUTH_USER = os.environ.get("GRADIO_USERNAME")
AUTH_PASSWORD = os.environ.get("GRADIO_PASSWORD")
MAX_PROMPT_CHARS = 1200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_PIXELS = 1024 * 1024  # T4 16GB safe limit (~1.05 megapixels)

# ---------------------------------------------------------------------------
# Model Registry — Verified SDXL models (public, no token required)
#
# Each entry contains the HuggingFace repo, optimal generation defaults
# tuned for T4 16GB, a model-specific negative prompt, the recommended
# scheduler, and a type tag used for UI hints.
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "Juggernaut XL V9  (Cinematic / Photoreal)": {
        "repo": "RunDiffusion/Juggernaut-XL-v9",
        "default_steps": 30,
        "default_guidance": 6.0,
        "default_width": 1024,
        "default_height": 1024,
        "default_negative": (
            "drawing, painting, anime, cartoon, 3d render, illustration, "
            "sketch, bad anatomy, bad hands, deformed face, extra fingers, "
            "missing fingers, watermark, blurry, noise, low quality, "
            "worst quality, mutated"
        ),
        "default_scheduler": "DPM++ 2M Karras",
        "type": "photoreal",
    },
    "RealVisXL V4  (Best Realism)": {
        "repo": "SG161222/RealVisXL_V4.0",
        "default_steps": 30,
        "default_guidance": 7.0,
        "default_width": 1024,
        "default_height": 1024,
        "default_negative": (
            "drawing, painting, anime, cartoon, 3d render, illustration, "
            "text, logo, watermark, low quality, bad anatomy, deformed eyes, "
            "extra fingers, blurry, noise, skin spots"
        ),
        "default_scheduler": "DPM++ 2M Karras",
        "type": "photoreal",
    },
    "Fluently XL V4  (Uncensored Photoreal & Art)": {
        "repo": "fluently/Fluently-XL-v4",
        "default_steps": 28,
        "default_guidance": 6.5,
        "default_width": 1024,
        "default_height": 1024,
        "default_negative": (
            "worst quality, low quality, blurry, deformed face, bad anatomy, "
            "extra fingers, missing fingers, watermark, text, lowres, artifacts"
        ),
        "default_scheduler": "DPM++ 2M Karras",
        "type": "hybrid",
    },
    "Animagine XL 3.1  (Uncensored Anime / 2D)": {
        "repo": "cagliostrolab/animagine-xl-3.1",
        "default_steps": 28,
        "default_guidance": 7.0,
        "default_width": 832,
        "default_height": 1216,
        "default_negative": (
            "lowres, bad anatomy, bad hands, text, error, missing fingers, "
            "extra digit, fewer digits, cropped, worst quality, low quality, "
            "normal quality, jpeg artifacts, signature, watermark, username, "
            "blurry, photo, photorealistic, 3d render, real human"
        ),
        "default_scheduler": "Euler Ancestral",
        "type": "anime",
    },
    "DreamShaper XL Turbo  (Fast 4-Step)": {
        "repo": "Lykon/dreamshaper-xl-v2-turbo",
        "default_steps": 6,
        "default_guidance": 2.0,
        "default_width": 1024,
        "default_height": 1024,
        "default_negative": "ugly, blurry, low quality, distorted, bad anatomy",
        "default_scheduler": "DPM++ 2M Karras",
        "type": "turbo",
    },
    "SDXL Base 1.0  (Standard)": {
        "repo": "stabilityai/stable-diffusion-xl-base-1.0",
        "default_steps": 30,
        "default_guidance": 7.5,
        "default_width": 1024,
        "default_height": 1024,
        "default_negative": (
            "worst quality, low quality, blurry, out of focus, noise, "
            "jpeg artifacts, distorted, bad anatomy, extra fingers, "
            "watermark, text"
        ),
        "default_scheduler": "DPM++ 2M Karras",
        "type": "standard",
    },
}

MODELS = {k: v["repo"] for k, v in MODEL_CONFIGS.items()}

# ---------------------------------------------------------------------------
# LoRA Registry — VERIFIED on HuggingFace (web-validated, repos confirmed)
#
#   nerijs/pixel-art-xl          ✅ exists (note: "nerijs" not "nerjs")
#   CiroN2022/toy-face           ✅ exists
#   latent-consistency/lcm-lora-sdxl  ✅ exists (acceleration LoRA)
#
# Previously listed repos that were REMOVED because they 404:
#   greggh/sdxl-lora-vector-art  ❌ does not exist
#   Lingxiao/line-art-xl         ❌ does not exist
# ---------------------------------------------------------------------------
POPULAR_LORAS = {
    "None": {
        "repo": "", "weight_name": "", "trigger": "", "is_lcm": False,
    },
    "🔥 Sensual & Body Detailer": {
        "repo": "ntc-ai/SDXL-LoRA-slider.sexy",
        "weight_name": "sexy.safetensors",
        "trigger": "sexy, highly detailed, realistic skin",
        "is_lcm": False,
    },
    "🌸 Anime Uncensored Detailer": {
        "repo": "Linaqruf/anime-detailer-xl-lora",
        "weight_name": "anime-detailer-xl.safetensors",
        "trigger": "masterpiece, best quality, highly detailed anime",
        "is_lcm": False,
    },
    "✨ Ultra Detail Enhancer": {
        "repo": "ntc-ai/SDXL-LoRA-slider.extremely-detailed",
        "weight_name": "extremely detailed.safetensors",
        "trigger": "extremely detailed, high resolution",
        "is_lcm": False,
    },
    "🎬 Cinematic Lighting Slider": {
        "repo": "ntc-ai/SDXL-LoRA-slider.cinematic-lighting",
        "weight_name": "cinematic lighting.safetensors",
        "trigger": "cinematic lighting, dramatic atmosphere",
        "is_lcm": False,
    },
    "Pixel Art XL": {
        "repo": "nerijs/pixel-art-xl",
        "weight_name": "pixel-art-xl.safetensors",
        "trigger": "pixel art",
        "is_lcm": False,
    },
    "Toy Face 3D": {
        "repo": "CiroN2022/toy-face",
        "weight_name": "toy_face_sdxl.safetensors",
        "trigger": "toy_face",
        "is_lcm": False,
    },
    "LCM Accelerator (4-step speed)": {
        "repo": "latent-consistency/lcm-lora-sdxl",
        "weight_name": "pytorch_lora_weights.safetensors",
        "trigger": "",
        "is_lcm": True,
    },
    "Custom HuggingFace Repo": {
        "repo": "custom", "weight_name": "", "trigger": "", "is_lcm": False,
    },
}

# ---------------------------------------------------------------------------
# Schedulers — applied per-generation so model weights stay clean
# ---------------------------------------------------------------------------
SCHEDULER_MAP = {
    "DPM++ 2M Karras": lambda cfg: DPMSolverMultistepScheduler.from_config(
        cfg, use_karras_sigmas=True
    ),
    "Euler Ancestral": lambda cfg: EulerAncestralDiscreteScheduler.from_config(cfg),
    "Euler": lambda cfg: EulerDiscreteScheduler.from_config(cfg),
    "UniPC (Fast)": lambda cfg: UniPCMultistepScheduler.from_config(cfg),
}

# ---------------------------------------------------------------------------
# SDXL Native Aspect Ratios (trained resolutions, all ≤ 1,048,576 px)
# ---------------------------------------------------------------------------
ASPECT_RATIOS = {
    "1:1 Square (1024×1024)": (1024, 1024),
    "2:3 Portrait (832×1216)": (832, 1216),
    "3:2 Landscape (1216×832)": (1216, 832),
    "9:16 Tall (768×1344)": (768, 1344),
    "16:9 Wide (1344×768)": (1344, 768),
    "4:5 Social (896×1152)": (896, 1152),
    "5:4 Landscape (1152×896)": (1152, 896),
}

# ---------------------------------------------------------------------------
# Style Presets
#
# Emoji prefixes hint at the intended model family:
#   🌸/🎬  = best with Anime models
#   📸/🎥/👤 = best with Photoreal models
#   🎨/🌆/⚔️/📦 = universal
#
# apply_preset() REPLACES the negative prompt (does not append).
# ---------------------------------------------------------------------------
STYLE_PRESETS = {
    "None (Use Model Default)": ("", ""),
    "🌸 Anime Masterpiece": (
        "masterpiece, best quality, highly detailed, anime style, "
        "vibrant colors, crisp lines",
        "lowres, bad anatomy, bad hands, text, error, missing fingers, "
        "extra digit, cropped, worst quality, low quality, jpeg artifacts, "
        "signature, watermark, photo, 3d render",
    ),
    "🎬 Anime Cinematic": (
        "masterpiece, anime aesthetic, dramatic lighting, cinematic "
        "composition, breathtaking background, 8k",
        "photorealistic, real photo, 3d render, low quality, blurry, "
        "noise, watermark",
    ),
    "🎨 Digital Illustration": (
        "masterpiece, digital painting, fine art illustration, smooth "
        "shading, concept art, highly detailed",
        "photo, 3d render, low quality, blurry, watermark",
    ),
    "📸 Photorealistic RAW": (
        "RAW photo, photorealistic, 8k uhd, DSLR, soft lighting, "
        "high quality, film grain, Fujifilm XT3",
        "drawing, painting, cartoon, anime, 3d render, illustration, "
        "text, logo, watermark, low quality",
    ),
    "🎥 Cinematic Film": (
        "cinematic photo, dramatic lighting, film still, anamorphic lens, "
        "shallow depth of field, 4k, movie quality",
        "amateur, flat lighting, low resolution, blurry, noise, watermark, "
        "cartoon, anime",
    ),
    "👤 Portrait Studio": (
        "professional portrait photography, studio lighting, sharp eyes, "
        "85mm lens, beautiful bokeh, skin detail",
        "bad anatomy, bad hands, extra fingers, deformed eyes, blurry, "
        "low quality, watermark, cartoon",
    ),
    "🌆 Cyberpunk": (
        "cyberpunk city, neon lights, rain-soaked streets, volumetric fog, "
        "ultra-detailed, cinematic, 8k",
        "daylight, countryside, low quality, blurry, cartoon, "
        "simple background",
    ),
    "⚔️ Fantasy Art": (
        "epic fantasy art, highly detailed, magical atmosphere, dramatic "
        "composition, trending on artstation, 8k",
        "photo, realistic, ugly, bad anatomy, watermark, blurry, simple",
    ),
    "📦 Product Photography": (
        "product photo, clean white studio background, professional "
        "lighting, sharp focus, commercial quality",
        "dirty, cluttered background, low quality, blurry, "
        "shadow artifacts, text",
    ),
}

# ---------------------------------------------------------------------------
# Pipeline State (module-level singletons)
# ---------------------------------------------------------------------------
current_model_key = None
pipe = None
original_scheduler_config = None


def load_model(model_key: str):
    """Load an SDXL model, unloading any previous model to free VRAM.

    Memory optimizations applied for T4 16GB:
    - enable_vae_slicing: zero-cost safety for batch decode
    - enable_vae_tiling: caps VAE decode VRAM at ~0.5GB regardless of res
    - NO enable_attention_slicing: it's SLOWER than PyTorch 2.x SDPA
    """
    global pipe, current_model_key, original_scheduler_config

    if current_model_key == model_key and pipe is not None:
        return pipe

    # Unload previous model to free VRAM
    if pipe is not None:
        print(f"[Pipeline] Unloading {current_model_key}...")
        try:
            if hasattr(pipe, "unload_lora_weights"):
                pipe.unload_lora_weights()
        except Exception:
            pass
        del pipe
        pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    model_id = MODELS[model_key]
    print(f"[Pipeline] Loading {model_id} on {DEVICE}...")
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32

    try:
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_id, torch_dtype=dtype, use_safetensors=True, variant="fp16"
        )
    except Exception:
        # Some repos don't publish an fp16 variant — fall back to default
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_id, torch_dtype=dtype, use_safetensors=True
        )

    pipe = pipe.to(DEVICE)

    if DEVICE == "cuda":
        # vae_slicing: zero-cost safety for potential batch decode
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        # vae_tiling: caps VAE decode VRAM spike at ~0.5GB (critical for T4)
        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()

    # Store the original scheduler config for creating new scheduler instances
    original_scheduler_config = pipe.scheduler.config

    # Apply model's recommended default scheduler
    cfg = MODEL_CONFIGS.get(model_key, {})
    default_sched = cfg.get("default_scheduler", "DPM++ 2M Karras")
    if default_sched in SCHEDULER_MAP:
        pipe.scheduler = SCHEDULER_MAP[default_sched](pipe.scheduler.config)

    current_model_key = model_key
    print(f"[Pipeline] Ready: {model_key}")
    return pipe


# ---------------------------------------------------------------------------
# UI Callbacks
# ---------------------------------------------------------------------------
def update_model_defaults(model_key):
    """When user switches model, auto-populate optimal settings."""
    cfg = MODEL_CONFIGS.get(model_key, MODEL_CONFIGS[list(MODEL_CONFIGS.keys())[0]])
    model_type = cfg.get("type", "hybrid")
    
    # Filter presets by model type
    filtered_presets = ["None (Use Model Default)"]
    for preset in STYLE_PRESETS.keys():
        if preset == "None (Use Model Default)":
            continue
        # Anime models get Anime and Universal presets
        if model_type == "anime":
            if any(preset.startswith(e) for e in ["🌸", "🎬", "🎨", "🌆", "⚔️"]):
                filtered_presets.append(preset)
        # Photoreal/Standard get Photoreal and Universal presets
        else:
            if any(preset.startswith(e) for e in ["📸", "🎥", "👤", "🎨", "🌆", "⚔️", "📦"]):
                filtered_presets.append(preset)

    return (
        cfg["default_negative"],
        cfg["default_steps"],
        cfg["default_guidance"],
        cfg["default_width"],
        cfg["default_height"],
        cfg.get("default_scheduler", "DPM++ 2M Karras"),
        gr.update(choices=filtered_presets, value="None (Use Model Default)")
    )


def update_aspect_ratio(ratio_key):
    """When user picks an aspect ratio preset, update width/height sliders."""
    w, h = ASPECT_RATIOS.get(ratio_key, (1024, 1024))
    return w, h


def apply_preset(preset_name, cur_prompt, model_key):
    """Apply a style preset.

    Positive tokens are appended to the current prompt.
    Negative prompt is REPLACED (not appended) to prevent the additive
    duplication bug where clicking Apply multiple times creates garbage.
    """
    pos_add, neg_override = STYLE_PRESETS.get(preset_name, ("", ""))
    # Append positive style tokens to existing prompt without duplicating
    cur_prompt = (cur_prompt or "").strip()
    if pos_add and pos_add not in cur_prompt:
        new_prompt = f"{cur_prompt}, {pos_add}".strip(", ") if cur_prompt else pos_add
    else:
        new_prompt = cur_prompt
    # REPLACE negative with preset's negative (or fall back to model default)
    if neg_override:
        new_neg = neg_override
    else:
        cfg = MODEL_CONFIGS.get(model_key, {})
        new_neg = cfg.get("default_negative", "")
    return new_prompt, new_neg


# ---------------------------------------------------------------------------
# Core Generation
# ---------------------------------------------------------------------------
def generate_image(
    generation_mode, init_image, mask_image, denoising_strength,
    model_key, prompt, negative_prompt,
    num_inference_steps, guidance_scale, width, height,
    scheduler_name, use_freeu,
    lora_select, lora_custom_repo, lora_scale,
    seed,
    history,
):
    """Generate an image with full error handling, seed control, LoRA, and
    scheduler support.  Returns (gallery, gallery_state, status_text)."""

    prompt_str = str(prompt or "")
    print(f"[Generate] [{model_key}] [{generation_mode}] '{prompt_str[:80]}...'")
    try:
        prompt = prompt_str.strip()
        negative_prompt = (negative_prompt or "").strip()

        if not prompt:
            return history, history, "⚠️ Prompt is required."
        if len(prompt) > MAX_PROMPT_CHARS or len(negative_prompt) > MAX_PROMPT_CHARS:
            return (
                history, history,
                f"⚠️ Prompt exceeds {MAX_PROMPT_CHARS} character limit.",
            )

        width, height = int(width), int(height)
        total_px = width * height

        if total_px > MAX_PIXELS:
            return (
                history, history,
                f"⚠️ Resolution {width}×{height} = {total_px:,} pixels "
                f"exceeds {MAX_PIXELS:,} pixel limit. Use a preset aspect "
                f"ratio or reduce dimensions.",
            )
        if width % 8 != 0 or height % 8 != 0:
            return history, history, "⚠️ Width and height must be multiples of 8."

        # --- Load model --------------------------------------------------
        active_pipe = load_model(model_key)

        # --- Cast Pipeline based on generation_mode -----------------------
        if generation_mode == "Img2Img":
            active_pipe = AutoPipelineForImage2Image.from_pipe(active_pipe)
        elif generation_mode == "Inpainting":
            active_pipe = AutoPipelineForInpainting.from_pipe(active_pipe)

        # --- Apply user-selected scheduler --------------------------------
        if scheduler_name in SCHEDULER_MAP:
            active_pipe.scheduler = SCHEDULER_MAP[scheduler_name](
                original_scheduler_config
            )

        # --- Resolve LoRA -------------------------------------------------
        loras_to_load = []
        is_lcm = False
        
        lora_selections = lora_select if isinstance(lora_select, list) else [lora_select]
        
        for lora_item in lora_selections:
            if not lora_item or lora_item == "None":
                continue
            if lora_item == "Custom HuggingFace Repo":
                repo = (lora_custom_repo or "").strip()
                if not repo:
                    return (
                        history, history,
                        "⚠️ Please specify a Custom HuggingFace LoRA repository name.",
                    )
                loras_to_load.append({"repo": repo, "weight_name": None, "name": "custom"})
            else:
                cfg = POPULAR_LORAS.get(lora_item, {})
                if cfg:
                    if cfg.get("is_lcm", False):
                        is_lcm = True
                    loras_to_load.append({
                        "repo": cfg.get("repo", ""),
                        "weight_name": cfg.get("weight_name", "") or None,
                        "name": lora_item
                    })

        lora_loaded = len(loras_to_load) > 0

        # --- Seed ---------------------------------------------------------
        seed_val = int(seed)
        if seed_val < 0:
            seed_val = torch.randint(0, 2**32 - 1, (1,)).item()
        generator = torch.Generator(device=DEVICE).manual_seed(seed_val)

        # --- Build generation kwargs --------------------------------------
        actual_steps = int(num_inference_steps)
        actual_guidance = float(guidance_scale)

        try:
            # Purge any lingering LoRAs from previous runs before loading
            try:
                active_pipe.unload_lora_weights()
            except Exception:
                pass

            if lora_loaded:
                loaded_names = []
                for linfo in loras_to_load:
                    print(f"[LoRA] Loading: {linfo['repo']} as {linfo['name']}")
                    if linfo['weight_name']:
                        active_pipe.load_lora_weights(
                            linfo['repo'], weight_name=linfo['weight_name'], adapter_name=linfo['name']
                        )
                    else:
                        active_pipe.load_lora_weights(
                            linfo['repo'], adapter_name=linfo['name']
                        )
                    loaded_names.append(linfo['name'])
                active_pipe.set_adapters(loaded_names)
                print(f"[LoRA] Set adapters: {loaded_names}")

            # LCM Accelerator LoRA overrides: scheduler, steps, and guidance
            if is_lcm and lora_loaded:
                active_pipe.scheduler = LCMScheduler.from_config(
                    original_scheduler_config
                )
                actual_steps = min(actual_steps, 8)
                actual_guidance = 1.5
                print(
                    f"[LCM] Auto-override: steps={actual_steps}, "
                    f"guidance={actual_guidance}, scheduler=LCMScheduler"
                )

            if use_freeu:
                print("[FreeU] Enabling FreeU")
                active_pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.2, b2=1.4)

            # Compel prompt weighting
            compel = Compel(
                tokenizer=[active_pipe.tokenizer, active_pipe.tokenizer_2], 
                text_encoder=[active_pipe.text_encoder, active_pipe.text_encoder_2], 
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, 
                requires_pooled=[False, True]
            )
            prompt_embeds, pooled_prompt_embeds = compel(prompt)
            negative_prompt_embeds, negative_pooled_prompt_embeds = compel(negative_prompt)

            gen_kwargs = dict(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                num_inference_steps=actual_steps,
                guidance_scale=actual_guidance,
                generator=generator,
            )

            if generation_mode == "Text2Img":
                gen_kwargs["width"] = width
                gen_kwargs["height"] = height
            elif generation_mode == "Img2Img":
                if init_image is None:
                    return history, history, "⚠️ Init image is required for Img2Img."
                gen_kwargs["image"] = init_image
                gen_kwargs["strength"] = denoising_strength
            elif generation_mode == "Inpainting":
                if init_image is None or mask_image is None:
                    return history, history, "⚠️ Init image and mask are required for Inpainting."
                gen_kwargs["image"] = init_image
                gen_kwargs["mask_image"] = mask_image
                gen_kwargs["strength"] = denoising_strength
            # Apply LoRA weight scale (not needed for LCM which runs at full weight)
            if lora_loaded and not is_lcm:
                gen_kwargs["cross_attention_kwargs"] = {"scale": float(lora_scale)}

            # --- Generate -----------------------------------------------------
            t0 = time.time()
            with torch.inference_mode():
                result = active_pipe(**gen_kwargs)
            elapsed = time.time() - t0

        finally:
            if use_freeu:
                try:
                    active_pipe.disable_freeu()
                except Exception:
                    pass
            # --- Cleanup LoRA ALWAYS (even on error/cancellation) --------------
            if lora_loaded:
                try:
                    active_pipe.unload_lora_weights()
                except Exception:
                    pass
                # Restore user's scheduler after LCM override
                if is_lcm and scheduler_name in SCHEDULER_MAP:
                    active_pipe.scheduler = SCHEDULER_MAP[scheduler_name](
                        original_scheduler_config
                    )

        # --- Build status message -----------------------------------------
        parts = [f"✅ {model_key}"]
        if use_freeu:
            parts.append("FreeU")
        if lora_loaded:
            lora_names_str = ", ".join(lora_selections) if isinstance(lora_selections, list) else str(lora_select)
            parts.append(f"LoRAs: {lora_names_str}")
        if is_lcm and lora_loaded:
            parts.append(f"LCM: steps={actual_steps} cfg={actual_guidance}")
        parts.append(f"Seed: {seed_val}")
        parts.append(f"{elapsed:.1f}s")
        parts.append(f"{width}×{height}")
        status = " | ".join(parts)

        # --- Update gallery -----------------------------------------------
        caption_parts = [
            f"Mode: {generation_mode}",
            f"Model: {model_key.split('(')[0].strip()}",
            f"Seed: {seed_val}",
            f"Time: {elapsed:.1f}s"
        ]
        if lora_loaded:
            lora_names_str = ", ".join(lora_selections) if isinstance(lora_selections, list) else str(lora_select)
            caption_parts.append(f"LoRAs: {lora_names_str}")
        if use_freeu:
            caption_parts.append("FreeU: On")
        
        caption = " | ".join(caption_parts)
        updated_history = history + [(result.images[0], caption)]

        return updated_history, updated_history, status

    except torch.cuda.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return (
            history, history,
            "❌ GPU out of memory. Lower resolution or step count, "
            "or try a Turbo model.",
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        err_msg = str(e)[:300]
        return history, history, f"❌ Generation failed: {err_msg}"


# ---------------------------------------------------------------------------
# UI Construction
# ---------------------------------------------------------------------------
CSS = """
/* App background and fonts */
.gradio-container { 
    max-width: 1400px !important; 
    margin: auto; 
    font-family: 'Inter', sans-serif;
}

/* Glassmorphism containers */
.gr-box, .gr-panel, .gr-block { 
    background: rgba(255, 255, 255, 0.05) !important; 
    backdrop-filter: blur(10px); 
    border: 1px solid rgba(255, 255, 255, 0.1) !important; 
    border-radius: 16px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.1);
}

/* Button styling */
#gen-btn { 
    font-size: 1.15em; 
    font-weight: 700; 
    background: linear-gradient(135deg, #6366f1, #3b82f6) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
}

#gen-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 20px -10px rgba(99, 102, 241, 0.8) !important;
}

button.primary {
    border-radius: 8px !important;
}

/* Input fields */
input, textarea, select {
    border-radius: 8px !important;
    transition: all 0.2s ease;
}

input:focus, textarea:focus, select:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
}

/* Gallery and Image styling */
.gr-gallery {
    border-radius: 16px !important;
    overflow: hidden;
}

/* Tabs styling */
.tab-nav {
    border-bottom: none !important;
    background: rgba(0, 0, 0, 0.05) !important;
    border-radius: 12px 12px 0 0 !important;
    padding: 8px 8px 0 8px !important;
}

.tab-nav button {
    border-radius: 8px 8px 0 0 !important;
    border: none !important;
    margin-right: 4px !important;
    transition: background 0.2s !important;
}

.tab-nav button.selected {
    background: #6366f1 !important;
    color: white !important;
}
"""

default_model = list(MODEL_CONFIGS.keys())[0]
default_cfg = MODEL_CONFIGS[default_model]

with gr.Blocks(title="AI Image Studio", theme=gr.themes.Slate(primary_hue="indigo", secondary_hue="blue"), css=CSS) as demo:
    gr.Markdown("# 🎨 AI Image Studio")
    gr.Markdown(
        "Multi-model SDXL · Auto-Optimized Settings · Seed Control · "
        "Scheduler Selection · Verified LoRAs · LCM Accelerator"
    )

    gallery_state = gr.State([])

    def on_generate_wrapper(
        mode, i2i_img, i2i_denoise, inpaint_img, inpaint_mask, inpaint_denoise,
        *args
    ):
        if mode == "Img2Img":
            init_img = i2i_img
            mask_img = None
            denoise = i2i_denoise
        elif mode == "Inpainting":
            init_img = inpaint_img
            mask_img = inpaint_mask
            denoise = inpaint_denoise
        else:
            init_img = None
            mask_img = None
            denoise = 0.0
            
        return generate_image(mode, init_img, mask_img, denoise, *args)

    with gr.Row():
        # ---- Left panel ---------------------------------------------------
        with gr.Column(scale=1):
            model_selector = gr.Dropdown(
                choices=list(MODEL_CONFIGS.keys()),
                value=default_model,
                label="🤖 Model",
            )

            with gr.Tabs() as mode_tabs:
                with gr.TabItem("Text2Img", id="Text2Img") as tab_t2i:
                    pass
                with gr.TabItem("Img2Img", id="Img2Img") as tab_i2i:
                    init_image_i2i = gr.Image(type="pil", label="Initial Image")
                    denoise_i2i = gr.Slider(minimum=0.0, maximum=1.0, value=0.75, step=0.05, label="Denoising Strength")
                with gr.TabItem("Inpainting", id="Inpainting") as tab_inpaint:
                    init_image_inpaint = gr.Image(type="pil", label="Image to Inpaint")
                    mask_image_inpaint = gr.Image(type="pil", label="Mask Image")
                    denoise_inpaint = gr.Slider(minimum=0.0, maximum=1.0, value=0.75, step=0.05, label="Denoising Strength")

            generation_mode = gr.State("Text2Img")
            tab_t2i.select(lambda: "Text2Img", outputs=generation_mode)
            tab_i2i.select(lambda: "Img2Img", outputs=generation_mode)
            tab_inpaint.select(lambda: "Inpainting", outputs=generation_mode)

            prompt = gr.Textbox(
                label="✏️ Prompt",
                placeholder=(
                    "A cinematic photo of a futuristic city at night, "
                    "neon lights, rain-soaked streets..."
                ),
                lines=4,
            )
            negative_prompt = gr.Textbox(
                label="🚫 Negative Prompt (Auto-Tuned per Model)",
                value=default_cfg["default_negative"],
                lines=3,
            )

            with gr.Row():
                style_preset = gr.Dropdown(
                    choices=list(STYLE_PRESETS.keys()),
                    value="None (Use Model Default)",
                    label="🎨 Style Preset",
                    scale=3,
                )
                preset_btn = gr.Button("Apply", scale=1)

            with gr.Accordion(
                "⚙️ Generation Settings (Auto-Optimized per Model)", open=True
            ):
                with gr.Row():
                    steps = gr.Slider(
                        label="Steps",
                        minimum=4,
                        maximum=60,
                        value=default_cfg["default_steps"],
                        step=1,
                    )
                    guidance = gr.Slider(
                        label="Guidance Scale (CFG)",
                        minimum=1.0,
                        maximum=15.0,
                        value=default_cfg["default_guidance"],
                        step=0.5,
                    )
                with gr.Row():
                    aspect_ratio = gr.Dropdown(
                        choices=list(ASPECT_RATIOS.keys()),
                        value="1:1 Square (1024×1024)",
                        label="📐 Aspect Ratio",
                        scale=2,
                    )
                    scheduler_select = gr.Dropdown(
                        choices=list(SCHEDULER_MAP.keys()),
                        value=default_cfg.get(
                            "default_scheduler", "DPM++ 2M Karras"
                        ),
                        label="🔄 Scheduler",
                        scale=2,
                    )
                    use_freeu_checkbox = gr.Checkbox(
                        label="🌟 Enable FreeU", value=False, scale=1
                    )
                with gr.Row():
                    width_slider = gr.Slider(
                        label="Width",
                        minimum=512,
                        maximum=1344,
                        value=default_cfg["default_width"],
                        step=64,
                    )
                    height_slider = gr.Slider(
                        label="Height",
                        minimum=512,
                        maximum=1344,
                        value=default_cfg["default_height"],
                        step=64,
                    )
                seed_input = gr.Number(
                    label="🎲 Seed (-1 = Random)", value=-1, precision=0
                )

            with gr.Accordion("🧩 LoRA Style Enhancer", open=True):
                lora_select = gr.Dropdown(
                    choices=list(POPULAR_LORAS.keys()),
                    value=[],
                    multiselect=True,
                    label="Select LoRA(s)",
                )
                lora_custom_repo = gr.Textbox(
                    label="Custom HuggingFace LoRA Repo",
                    placeholder="e.g. username/my-sdxl-lora",
                    visible=True,
                )
                lora_scale = gr.Slider(
                    label="LoRA Weight",
                    minimum=0.1,
                    maximum=1.5,
                    value=0.8,
                    step=0.05,
                )

            generate_btn = gr.Button(
                "🚀 Generate", variant="primary", elem_id="gen-btn"
            )

        # ---- Right panel --------------------------------------------------
        with gr.Column(scale=1):
            gallery = gr.Gallery(
                label="Generated Images",
                columns=2,
                height=640,
            )
            status_box = gr.Textbox(label="Status", interactive=False)

    # ---- Event Callbacks --------------------------------------------------
    model_selector.change(
        fn=update_model_defaults,
        inputs=[model_selector],
        outputs=[
            negative_prompt, steps, guidance,
            width_slider, height_slider, scheduler_select,
            style_preset,
        ],
    )

    aspect_ratio.change(
        fn=update_aspect_ratio,
        inputs=[aspect_ratio],
        outputs=[width_slider, height_slider],
    )

    preset_btn.click(
        fn=apply_preset,
        inputs=[style_preset, prompt, model_selector],
        outputs=[prompt, negative_prompt],
    )

    generate_btn.click(
        fn=on_generate_wrapper,
        inputs=[
            generation_mode,
            init_image_i2i, denoise_i2i,
            init_image_inpaint, mask_image_inpaint, denoise_inpaint,
            model_selector, prompt, negative_prompt,
            steps, guidance, width_slider, height_slider,
            scheduler_select, use_freeu_checkbox,
            lora_select, lora_custom_repo, lora_scale,
            seed_input,
            gallery_state,
        ],
        outputs=[gallery, gallery_state, status_box],
    )

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    share = os.environ.get("GRADIO_SHARE", "false").lower() == "true"
    auth = None
    if AUTH_USER and AUTH_PASSWORD:
        auth = [(AUTH_USER, AUTH_PASSWORD)]
    demo.queue()
    demo.launch(server_name=server_name, server_port=port, share=share, auth=auth)
