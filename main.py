import os
import gradio as gr
from horde_sdk.ai_horde_api.ai_horde_clients import AIHordeAPISimpleClient
from horde_sdk.ai_horde_api.apimodels import (
    ImageGenerateAsyncRequest,
    ImageGenerationInputPayload,
)
from horde_sdk.generation_parameters.image.consts import KNOWN_IMAGE_SAMPLERS

MODELS = [
    "stable_diffusion",
    "AlbedoBase XL (SDXL)",
    "Juggernaut XL",
    "AlbedoBase XL 3.1",
    "Flux.1-Schnell fp8 (Compact)",
]

client = AIHordeAPISimpleClient()

def generate_image(prompt, negative_prompt, model, width, height, steps, cfg_scale, nsfw, api_key):
    if not prompt.strip():
        return None, "Please enter a prompt."

    full_prompt = f"{prompt} ### {negative_prompt}" if negative_prompt.strip() else prompt

    # Dynamic Optimization for Quality
    sampler = KNOWN_IMAGE_SAMPLERS.k_dpmpp_2m 
    
    if "Flux" in model:
        steps = min(steps, 8)
        cfg_scale = 1.0 
        sampler = KNOWN_IMAGE_SAMPLERS.k_euler 

    # Handle API Key for Queue Priority
    active_key = api_key.strip() if api_key.strip() else "0000000000"

    try:
        request = ImageGenerateAsyncRequest(
            apikey=active_key,
            prompt=full_prompt,
            models=[model],
            params=ImageGenerationInputPayload(
                width=width,
                height=height,
                steps=steps,
                cfg_scale=cfg_scale,
                sampler_name=sampler,
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

        info = f"✅ {model} | {width}×{height} | Steps: {steps} | Sampler: {sampler.value}"
        return img, info

    except Exception as e:
        return None, f"❌ Error: {str(e)}"

def update_sliders(model_name):
    # Dynamically match native resolutions to prevent bad anatomy
    if "XL" in model_name or "Flux" in model_name:
        return gr.update(value=1024), gr.update(value=1024)
    else:
        return gr.update(value=512), gr.update(value=512)

# CSS injection for brutalist, high-contrast serif styling
custom_css = """
.gradio-container {
    font-family: 'Times New Roman', serif !important;
    background-color: #FFFFFF !important;
}
button {
    border: 2px solid #000000 !important;
    border-radius: 0px !important;
    text-transform: uppercase !important;
    font-weight: bold !important;
}
input, textarea, .dropdown {
    border: 1px solid #000000 !important;
    border-radius: 0px !important;
}
"""

with gr.Blocks(title="AI Horde Generator", theme=gr.themes.Monochrome(), css=custom_css) as demo:
    gr.Markdown("# AI Horde Image Generator")
    
    with gr.Row():
        with gr.Column():
            api_key = gr.Textbox(
                label="AI Horde API Key (Required for High Speed)",
                placeholder="Enter your API key from stablehorde.net to bypass the anonymous slow queue...",
                type="password"
            )
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
            
            steps = gr.Slider(10, 40, value=25, step=1, label="Steps")
            cfg_scale = gr.Slider(1, 12, value=6.5, step=0.5, label="CFG Scale")
            nsfw = gr.Checkbox(label="Allow NSFW", value=True)

            btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            output_image = gr.Image(label="Generated Image", type="pil", height=500)
            status = gr.Textbox(label="Status", lines=3)

    model.change(
        fn=update_sliders,
        inputs=[model],
        outputs=[width, height]
    )

    btn.click(
        fn=generate_image,
        inputs=[prompt, negative_prompt, model, width, height, steps, cfg_scale, nsfw, api_key],
        outputs=[output_image, status]
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
