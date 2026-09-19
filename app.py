import gradio as gr
import torch
from diffusers import StableDiffusionXLPipeline
import os
import gc

AUTH_USER = os.environ.get("GRADIO_USERNAME")
AUTH_PASSWORD = os.environ.get("GRADIO_PASSWORD")
LORA_ALLOW_REMOTE = os.environ.get("LORA_ALLOW_REMOTE", "false").lower() == "true"
LORA_ALLOWED_REPOS = {
    repo.strip() for repo in os.environ.get("LORA_ALLOWED_REPOS", "").split(",") if repo.strip()
}
MAX_PROMPT_CHARS = 1200

# ---------------------------------------------------------------------------
# Model registry — all SDXL, no token required
# ---------------------------------------------------------------------------
MODELS = {
    "Juggernaut XL V9  (Cinematic / Photoreal)": "RunDiffusion/Juggernaut-XL-v9",
    "RealVisXL V4  (Best Realism)": "SG161222/RealVisXL_V4.0",
    "Fluently XL V4  (Uncensored Photoreal & Art)": "fluently/Fluently-XL-v4",
    "Animagine XL 3.1  (Uncensored Anime / 2D)": "cagliostrolab/animagine-xl-3.1",
    "DreamShaper XL Turbo  (Fast 4-Step)": "Lykon/dreamshaper-xl-v2-turbo",
    "SDXL Base 1.0  (Standard)": "stabilityai/stable-diffusion-xl-base-1.0",
}

# ---------------------------------------------------------------------------
# Style presets
# ---------------------------------------------------------------------------
STYLE_PRESETS = {
    "None": ("", ""),
    "Photorealistic": (
        "RAW photo, photorealistic, 8k uhd, DSLR, soft lighting, high quality, film grain, Fujifilm XT3",
        "drawing, painting, cartoon, anime, 3d render, illustration, text, logo, watermark, low quality",
    ),
    "Cinematic": (
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

DEFAULT_NEGATIVE = (
    "worst quality, low quality, normal quality, lowres, blurry, out of focus, "
    "noise, jpeg artifacts, overexposed, underexposed, distorted, bad anatomy, "
    "bad hands, extra fingers, missing fingers, deformed face, duplicate, "
    "watermark, text, logo, signature, frame, cropped, mutated"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_PIXELS = 1024 * 1024  # 1024x1024 max for SDXL on T4

current_model_key = None
pipe = None


def load_model(model_key: str):
    global pipe, current_model_key

    if current_model_key == model_key and pipe is not None:
        return pipe

    # Unload previous model
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
        # Some repos don't have fp16 variant files — fall back to default
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


def apply_preset(preset_name, cur_prompt, cur_neg):
    pos_add, neg_add = STYLE_PRESETS.get(preset_name, ("", ""))
    new_prompt = f"{cur_prompt}, {pos_add}".strip(", ") if pos_add else cur_prompt
    new_neg = f"{cur_neg}, {neg_add}".strip(", ") if neg_add else cur_neg
    return new_prompt, new_neg


def generate_image(
    model_key, prompt, negative_prompt,
    num_inference_steps, guidance_scale, width, height,
    lora_repo, lora_scale
):
    print(f"[{model_key}] '{prompt}'")
    try:
        prompt = (prompt or "").strip()
        negative_prompt = (negative_prompt or "").strip()
        lora_repo = (lora_repo or "").strip()

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
            if not LORA_ALLOW_REMOTE and lora_repo not in LORA_ALLOWED_REPOS:
                return None, "LoRA loading is disabled unless the repo is explicitly allowed by server config."
            try:
                print(f"Loading LoRA: {lora_repo}")
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
            active_pipe.unload_lora_weights()

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
# UI
# ---------------------------------------------------------------------------
CSS = """
.gradio-container { max-width: 1280px !important; margin: auto; }
#gen-btn { font-size: 1.1em; }
"""

with gr.Blocks(title="AI Image Studio", theme=gr.themes.Soft(), css=CSS) as demo:
    gr.Markdown("# AI Image Studio")
    gr.Markdown("Multi-model SDXL · LoRA support · Style presets")

    with gr.Row():
        # ---- Left panel -----------------------------------------------
        with gr.Column(scale=1):
            model_selector = gr.Dropdown(
                choices=list(MODELS.keys()),
                value=list(MODELS.keys())[0],
                label="Model",
            )

            prompt = gr.Textbox(
                label="Prompt",
                placeholder="A cinematic photo of a futuristic city at night, neon lights, rain-soaked streets...",
                lines=4,
            )
            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                value=DEFAULT_NEGATIVE,
                lines=3,
            )

            with gr.Row():
                style_preset = gr.Dropdown(
                    choices=list(STYLE_PRESETS.keys()),
                    value="None",
                    label="Style Preset",
                    scale=3,
                )
                preset_btn = gr.Button("Apply", scale=1)

            with gr.Accordion("Generation Settings", open=True):
                with gr.Row():
                    steps = gr.Slider(label="Steps", minimum=10, maximum=60, value=30, step=1)
                    guidance = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=15.0, value=7.5, step=0.5)
                with gr.Row():
                    width = gr.Slider(label="Width", minimum=512, maximum=1280, value=1024, step=64)
                    height = gr.Slider(label="Height", minimum=512, maximum=1280, value=1024, step=64)

            with gr.Accordion("LoRA (Optional)", open=False):
                lora_repo = gr.Textbox(
                    label="LoRA HuggingFace Repo",
                    placeholder="Allowed server-side repo only",
                )
                lora_scale = gr.Slider(
                    label="LoRA Weight", minimum=0.1, maximum=1.5, value=0.8, step=0.05
                )

            generate_btn = gr.Button("Generate", variant="primary", elem_id="gen-btn")

        # ---- Right panel ----------------------------------------------
        with gr.Column(scale=1):
            output_image = gr.Image(label="Output", height=640)
            status_box = gr.Textbox(label="Status", interactive=False)

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
            lora_repo, lora_scale,
        ],
        outputs=[output_image, status_box],
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    share = os.environ.get("GRADIO_SHARE", "false").lower() == "true"
    auth = None
    if AUTH_USER and AUTH_PASSWORD:
        auth = [(AUTH_USER, AUTH_PASSWORD)]
    try:
        demo.queue(default_concurrency_limit=1)
    except TypeError:
        demo.queue(concurrency_count=1)
    demo.launch(server_name=server_name, server_port=port, share=share, auth=auth)
