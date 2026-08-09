import os
import gradio as gr
from horde_sdk.ai_horde_api.ai_horde_clients import AIHordeAPISimpleClient
from horde_sdk.ai_horde_api.apimodels import (
    ImageGenerateAsyncRequest,
    ImageGenerationInputPayload,
)
from horde_sdk.generation_parameters.image.consts import KNOWN_IMAGE_SAMPLERS

# Read API key from environment variable (safer)
DEFAULT_API_KEY = os.environ.get("HORDE_API_KEY", "fallback")

MODELS = [
    "stable_diffusion",
    "AlbedoBase XL (SDXL)",
    "Juggernaut XL",
    "AlbedoBase XL 3.1",
    "Flux.1-Schnell fp8 (Compact)",
]

client = AIHordeAPISimpleClient()

def generate_image(prompt, negative_prompt, model, width, height, steps, cfg_scale, nsfw):
    if not prompt.strip():
        return None, "Please enter a prompt."

    full_prompt = f"{prompt} ### {negative_prompt}" if negative_prompt.strip() else prompt

    if "Flux" in model:
        steps = min(steps, 8)

    try:
        request = ImageGenerateAsyncRequest(
            apikey=DEFAULT_API_KEY,
            prompt=full_prompt,
            models=[model],
            params=ImageGenerationInputPayload(
                width=width,
                height=height,
                steps=steps,
                cfg_scale=cfg_scale,
                sampler_name=KNOWN_IMAGE_SAMPLERS.k_euler_a,
                n=1,
            ),
            nsfw=nsfw,
            censor_nsfw=False,
            r2=True,
        )

        status, gen_id = client.image_generate_request(request)

        if not status.generations:
            return None, "No image returned. Try another model or lower settings."

        gen = status.generations[0]
        img = client.download_image_from_generation(gen)

        info = f"✅ {model} | {width}×{height} | Steps: {steps}"
        return img, info

    except Exception as e:
        return None, f"❌ Error: {str(e)}"

with gr.Blocks(title="AI Horde Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🎨 AI Horde Image Generator")

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", lines=4)
            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                value="blurry, low quality, deformed, ugly, bad anatomy, extra limbs, poorly drawn face, mutation, watermark, text",
                lines=2
            )
            model = gr.Dropdown(choices=MODELS, value="stable_diffusion", label="Model")
            with gr.Row():
                width = gr.Slider(512, 1024, value=512, step=64, label="Width")
                height = gr.Slider(512, 1024, value=512, step=64, label="Height")
            steps = gr.Slider(10, 40, value=18, step=1, label="Steps")
            cfg_scale = gr.Slider(4, 12, value=6.5, step=0.5, label="CFG Scale")
            nsfw = gr.Checkbox(label="Allow NSFW", value=True)

            btn = gr.Button("🚀 Generate", variant="primary")

        with gr.Column():
            output_image = gr.Image(label="Generated Image", type="pil", height=500)
            status = gr.Textbox(label="Status", lines=3)

    btn.click(
        fn=generate_image,
        inputs=[prompt, negative_prompt, model, width, height, steps, cfg_scale, nsfw],
        outputs=[output_image, status]
    )

# Important for Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)