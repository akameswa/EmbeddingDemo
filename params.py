import torch
from gradio.networking import setup_tunnel
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    LCMScheduler,
    StableDiffusionPipeline,
)

torch_device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "segmind/small-sd"
num_inference_steps = 8
guidance_scale = 8

tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder").to(
    torch_device
)
scheduler = LCMScheduler.from_pretrained(model_path, subfolder="scheduler")
unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet").to(
    torch_device
)
vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae").to(torch_device)

pipe = StableDiffusionPipeline(
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    unet=unet,
    scheduler=scheduler,
    vae=vae,
    safety_checker=None,
    feature_extractor=None,
    requires_safety_checker=False,
).to(torch_device)
