import os
import gradio as gr
from horde_sdk.ai_horde_api.ai_horde_clients import AIHordeAPISimpleClient
from horde_sdk.ai_horde_api.apimodels import (
    ImageGenerateAsyncRequest,
    ImageGenerationInputPayload,
)
from horde_sdk.generation_parameters.image.consts import KNOWN_IMAGE_SAMPLERS

DEFAULT_API_KEY = "hNFM7V-1fY6LBNjGVPgKtQ"

MODELS = [
    "Juggernaut XL",
    "AlbedoBase XL (SDXL)",
    "AlbedoBase XL 3.1",
    "stable_diffusion",
]

client = AIHordeAPISimpleClient()

def generate_image(prompt, negative_prompt, api_key, model, width, height, steps, cfg_scale, nsfw):
    if not prompt.strip():
        return None, "Please enter a prompt."

    key = api_key.strip() if api_key.strip() else DEFAULT_API_KEY
    full_prompt = f"{prompt} ### {negative_prompt}" if negative_prompt.strip() else prompt

    try:
        request = ImageGenerateAsyncRequest(
            apikey=key,
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
            return None, "No image returned. Try another model."

        gen = status.generations[0]
        img = client.download_image_from_generation(gen)

        info = f"Done | Model: {model} | Size: {width}x{height} | Steps: {steps}"
        return img, info

    except Exception as e:
        return None, f"Error: {str(e)}"

with gr.Blocks(title="AI Horde Generator") as demo:
    gr.Markdown("## AI Horde Image Generator\nRecommended: Juggernaut XL + 512x512 + 18 steps")

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", lines=4)
            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                value="blurry, low quality, deformed, ugly, bad anatomy, extra limbs, poorly drawn face, mutation, plastic skin, doll-like, artificial, over-smooth, cartoon, anime, illustration, painting, watermark, text",
                lines=3
            )

            with gr.Accordion("Settings", open=True):
                api_key = gr.Textbox(label="API Key", type="password", value="")
                model = gr.Dropdown(choices=MODELS, value="Juggernaut XL", label="Model")
                with gr.Row():
                    width = gr.Slider(512, 768, value=512, step=64, label="Width")
                    height = gr.Slider(512, 768, value=512, step=64, label="Height")
                steps = gr.Slider(12, 28, value=18, step=1, label="Steps")
                cfg_scale = gr.Slider(4.0, 8.0, value=6.0, step=0.5, label="CFG Scale")
                nsfw = gr.Checkbox(label="Allow NSFW", value=True)

            btn = gr.Button("Generate Image", variant="primary")

        with gr.Column():
            output_image = gr.Image(label="Generated Image", type="pil", height=480)
            status = gr.Textbox(label="Status", lines=3)

    btn.click(
        fn=generate_image,
        inputs=[prompt, negative_prompt, api_key, model, width, height, steps, cfg_scale, nsfw],
        outputs=[output_image, status]
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
