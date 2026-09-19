import gradio as gr
import torch
from diffusers import StableDiffusionXLPipeline
import os
import gc

AUTH_USER = os.environ.get("GRADIO_USERNAME")
AUTH_PASSWORD = os.environ.get("GRADIO_PASSWORD")
LORA_ALLOW_REMOTE = os.environ.get("LORA_ALLOW_REMOTE", "true").lower() == "true"
LORA_ALLOWED_REPOS = {
    repo.strip() for repo in os.environ.get("LORA_ALLOWED_REPOS", "").split(",") if repo.strip()
}
MAX_PROMPT_CHARS = 1200

# ---------------------------------------------------------------------------
# Model Registry with Model-Specific Negative Prompts & Optimal Settings
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "Juggernaut XL V9  (Cinematic / Photoreal)": {
        "repo": "RunDiffusion/Juggernaut-XL-v9",
        "default_steps": 30,
        "default_guidance": 6.0,
        "default_width": 1024,
        "default_height": 1024,
        "default_negative": "drawing, painting, anime, cartoon, 3d render, illustration, sketch, bad anatomy, bad hands, deformed face, extra fingers, missing fingers, watermark, blurry, noise, low quality, worst quality, mutated",
        "type": "photoreal"
    },
    "RealVisXL V4  (Best Realism)": {
        "repo": "SG161222/RealVisXL_V4.0",
        "default_steps": 30,
        "default_guidance": 7.0,
        "default_width": 1024,
        "default_height": 1024,
        "default_negative": "drawing, painting, anime, cartoon, 3d render, illustration, text, logo, watermark, low quality, bad anatomy, deformed eyes, extra fingers, blurry, noise, skin spots",
        "type": "photoreal"
    },
    "Fluently XL V4  (Uncensored Photoreal & Art)": {
        "repo": "fluently/Fluently-XL-v4",
        "default_steps": 28,
        "default_guidance": 6.5,
        "default_width": 1024,
        "default_height": 1024,
        "default_negative": "worst quality, low quality, blurry, deformed face, bad anatomy, extra fingers, missing fingers, watermark, text, lowres, artifacts",
        "type": "hybrid"
    },
    "Animagine XL 3.1  (Uncensored Anime / 2D)": {
        "repo": "cagliostrolab/animagine-xl-3.1",
        "default_steps": 28,
        "default_guidance": 7.0,
        "default_width": 832,
        "default_height": 1216,
        "default_negative": "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, photo, photorealistic, 3d render, real human",
        "type": "anime"
    },
    "DreamShaper XL Turbo  (Fast 4-Step)": {
        "repo": "Lykon/dreamshaper-xl-v2-turbo",
        "default_steps": 6,
        "default_guidance": 2.0,
        "default_width": 1024,
        "default_height": 1024,
        "default_negative": "ugly, blurry, low quality, distorted, bad anatomy",
        "type": "turbo"
    },
    "SDXL Base 1.0  (Standard)": {
        "repo": "stabilityai/stable-diffusion-xl-base-1.0",
        "default_steps": 30,
        "default_guidance": 7.5,
        "default_width": 1024,
        "default_height": 1024,
        "default_negative": "worst quality, low quality, blurry, out of focus, noise, jpeg artifacts, distorted, bad anatomy, extra fingers, watermark, text",
        "type": "standard"
    },
}

MODELS = {k: v["repo"] for k, v in MODEL_CONFIGS.items()}

# ---------------------------------------------------------------------------
# Popular Working SDXL LoRAs
# ---------------------------------------------------------------------------
POPULAR_LORAS = {
    "None": "",
    "Pixel Art XL": "nerjs/pixel-art-xl",
    "Vector Art XL": "greggh/sdxl-lora-vector-art",
    "Line Art XL": "Lingxiao/line-art-xl",
    "Custom HuggingFace Repo": "custom",
}

# ---------------------------------------------------------------------------
# Model & Style Presets
# ---------------------------------------------------------------------------
STYLE_PRESETS = {
    "None": ("", ""),
    "Anime Masterpiece (for Animagine)": (
        "masterpiece, best quality, highly detailed, anime style, vibrant colors, crisp lines",
        "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, cropped, worst quality, low quality, jpeg artifacts, signature, watermark, photo, 3d render",
    ),
    "Anime Cinematic (Makoto Shinkai)": (
        "masterpiece, anime aesthetic, dramatic lighting, cinematic composition, Makoto Shinkai style, breathtaking background, 8k",
        "photorealistic, real photo, 3d render, low quality, blurry, noise, watermark",
    ),
    "Digital Illustration (2D Concept)": (
        "masterpiece, digital painting, fine art illustration, smooth shading, concept art, pixiv, highly detailed",
        "photo, 3d render, low quality, blurry, watermark",
    ),
    "Photorealistic (for Juggernaut/RealVis)": (
        "RAW photo, photorealistic, 8k uhd, DSLR, soft lighting, high quality, film grain, Fujifilm XT3",
        "drawing, painting, cartoon, anime, 3d render, illustration, text, logo, watermark, low quality",
    ),
    "Cinematic Photo": (
        "cinematic photo, dramatic lighting, film still, anamorphic lens, shallow depth of field, 4k, movie quality",
        "amateur, flat lighting, low resolution, blurry, noise, watermark",
    ),
    "Portrait Studio": (
        "professional portrait photography, studio lighting, sharp eyes, 85mm lens, beautiful bokeh, skin detail",
        "bad anatomy, bad hands, extra fingers, deformed eyes, blurry, low quality, watermark",
    ),
    "Cyberpunk": (
        "cyberpunk city, neon lights, rain-soaked streets, volumetric fog, ultra-detailed, cinematic, 8k",
        "daylight, countryside, low quality, blurry, cartoon",
    ),
    "Fantasy Art": (
        "epic fantasy art, highly detailed, magical atmosphere, dramatic composition, trending on artstation, 8k",
        "photo, realistic, ugly, bad anatomy, watermark, blurry",
    ),
    "Product Photography": (
        "product photo, clean white studio background, professional lighting, sharp focus, commercial quality",
        "dirty, cluttered background, low quality, blurry, shadow artifacts",
    ),
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_PIXELS = 1024 * 1024

current_model_key = None
pipe = None


def load_model(model_key: str):
    global pipe, current_model_key

    if current_model_key == model_key and pipe is not None:
        return pipe

    if pipe is not None:
        print(f"Unloading {current_model_key}...")
        del pipe
        pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    model_id = MODELS[model_key]
    print(f"Loading {model_id} on {DEVICE}...")
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32

    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id, torch_dtype=dtype, use_safetensors=True, variant="fp16"
        )
    except Exception:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id, torch_dtype=dtype, use_safetensors=True
        )

    pipe = pipe.to(DEVICE)
    if DEVICE == "cuda":
        pipe.enable_attention_slicing()
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()

    current_model_key = model_key
    print(f"Ready: {model_key}")
    return pipe


def update_model_defaults(model_key):
    cfg = MODEL_CONFIGS.get(model_key, MODEL_CONFIGS[list(MODEL_CONFIGS.keys())[0]])
    return (
        cfg["default_negative"],
        cfg["default_steps"],
        cfg["default_guidance"],
        cfg["default_width"],
        cfg["default_height"],
    )


def apply_preset(preset_name, cur_prompt, cur_neg):
    pos_add, neg_add = STYLE_PRESETS.get(preset_name, ("", ""))
    new_prompt = f"{cur_prompt}, {pos_add}".strip(", ") if pos_add else cur_prompt
    new_neg = f"{cur_neg}, {neg_add}".strip(", ") if neg_add else cur_neg
    return new_prompt, new_neg


def generate_image(
    model_key, prompt, negative_prompt,
    num_inference_steps, guidance_scale, width, height,
    lora_select, lora_custom_repo, lora_scale
):
    print(f"[{model_key}] '{prompt}'")
    try:
        prompt = (prompt or "").strip()
        negative_prompt = (negative_prompt or "").strip()
        
        # Determine LoRA repository
        lora_repo = ""
        if lora_select == "Custom HuggingFace Repo":
            lora_repo = (lora_custom_repo or "").strip()
        elif lora_select and lora_select != "None":
            lora_repo = POPULAR_LORAS.get(lora_select, "").strip()

        if not prompt:
            return None, "Prompt is required."
        if len(prompt) > MAX_PROMPT_CHARS or len(negative_prompt) > MAX_PROMPT_CHARS:
            return None, "Prompt or negative prompt is too long."

        width, height = int(width), int(height)

        if (width * height) > MAX_PIXELS:
            return None, f"Resolution {width}x{height} exceeds 1024x1024 limit. Reduce size."
        if width % 8 != 0 or height % 8 != 0:
            return None, "Width and height must both be multiples of 8."

        active_pipe = load_model(model_key)

        lora_loaded = False
        if lora_repo:
            try:
                print(f"Loading LoRA weights: {lora_repo}")
                active_pipe.load_lora_weights(lora_repo)
                lora_loaded = True
            except Exception as lora_err:
                print(f"LoRA load failed (skipping): {lora_err}")

        gen_kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            width=width,
            height=height,
        )
        if lora_loaded:
            gen_kwargs["cross_attention_kwargs"] = {"scale": float(lora_scale)}

        with torch.inference_mode():
            result = active_pipe(**gen_kwargs)

        if lora_loaded:
            try:
                active_pipe.unload_lora_weights()
            except Exception:
                pass

        return result.images[0], f"Done — {model_key}"

    except torch.cuda.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None, "GPU out of memory. Lower resolution or step count."
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, "Generation failed. Check server logs for details."


# ---------------------------------------------------------------------------
# UI Construction
# ---------------------------------------------------------------------------
CSS = """
.gradio-container { max-width: 1280px !important; margin: auto; }
#gen-btn { font-size: 1.1em; }
"""

default_model = list(MODEL_CONFIGS.keys())[0]
default_cfg = MODEL_CONFIGS[default_model]

with gr.Blocks(title="AI Image Studio", theme=gr.themes.Soft(), css=CSS) as demo:
    gr.Markdown("# AI Image Studio")
    gr.Markdown("Multi-model SDXL · Auto-Optimized Settings · Tailored Negative Prompts · LoRA Support")

    with gr.Row():
        # ---- Left panel -----------------------------------------------
        with gr.Column(scale=1):
            model_selector = gr.Dropdown(
                choices=list(MODEL_CONFIGS.keys()),
                value=default_model,
                label="Model",
            )

            prompt = gr.Textbox(
                label="Prompt",
                placeholder="A cinematic photo of a futuristic city at night, neon lights, rain-soaked streets...",
                lines=4,
            )
            negative_prompt = gr.Textbox(
                label="Negative Prompt (Auto-Tuned per Model)",
                value=default_cfg["default_negative"],
                lines=3,
            )

            with gr.Row():
                style_preset = gr.Dropdown(
                    choices=list(STYLE_PRESETS.keys()),
                    value="None",
                    label="Style Preset",
                    scale=3,
                )
                preset_btn = gr.Button("Apply Preset", scale=1)

            with gr.Accordion("Generation Settings (Auto-Optimized)", open=True):
                with gr.Row():
                    steps = gr.Slider(
                        label="Steps", minimum=4, maximum=60,
                        value=default_cfg["default_steps"], step=1
                    )
                    guidance = gr.Slider(
                        label="Guidance Scale", minimum=1.0, maximum=15.0,
                        value=default_cfg["default_guidance"], step=0.5
                    )
                with gr.Row():
                    width = gr.Slider(
                        label="Width", minimum=512, maximum=1280,
                        value=default_cfg["default_width"], step=64
                    )
                    height = gr.Slider(
                        label="Height", minimum=512, maximum=1280,
                        value=default_cfg["default_height"], step=64
                    )

            with gr.Accordion("LoRA Enhancer (Working & Testable)", open=False):
                lora_select = gr.Dropdown(
                    choices=list(POPULAR_LORAS.keys()),
                    value="None",
                    label="Select LoRA Style",
                )
                lora_custom_repo = gr.Textbox(
                    label="Or Enter Custom HuggingFace LoRA Repo",
                    placeholder="e.g. ostris/super-cereal-sdxl-lora",
                    visible=True,
                )
                lora_scale = gr.Slider(
                    label="LoRA Weight", minimum=0.1, maximum=1.5, value=0.8, step=0.05
                )

            generate_btn = gr.Button("Generate", variant="primary", elem_id="gen-btn")

        # ---- Right panel ----------------------------------------------
        with gr.Column(scale=1):
            output_image = gr.Image(label="Output", height=640)
            status_box = gr.Textbox(label="Status", interactive=False)

    # ---- Event Callbacks -------------------------------------------
    model_selector.change(
        fn=update_model_defaults,
        inputs=[model_selector],
        outputs=[negative_prompt, steps, guidance, width, height],
    )

    preset_btn.click(
        fn=apply_preset,
        inputs=[style_preset, prompt, negative_prompt],
        outputs=[prompt, negative_prompt],
    )

    generate_btn.click(
        fn=generate_image,
        inputs=[
            model_selector, prompt, negative_prompt,
            steps, guidance, width, height,
            lora_select, lora_custom_repo, lora_scale,
        ],
        outputs=[output_image, status_box],
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    share = os.environ.get("GRADIO_SHARE", "false").lower() == "true"
    demo.queue()
    demo.launch(server_name=server_name, server_port=port, share=share)
