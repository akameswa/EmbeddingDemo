import torch
import secrets
from gradio.networking import setup_tunnel
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    LCMScheduler,
    EulerDiscreteScheduler,
    StableDiffusionPipeline,
)

torch_device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "segmind/small-sd"
num_inference_steps = 8
seed = 69420
guidance_scale = 8
imageHeight, imageWidth = 512, 512

tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder").to(torch_device)
scheduler = LCMScheduler.from_pretrained(model_path, subfolder="scheduler")
unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet").to(torch_device)
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

dash_tunnel = setup_tunnel("0.0.0.0", 8000, secrets.token_urlsafe(32), None)

__all__ = [
    "num_inference_steps",
    "seed",
    "tokenizer",
    "text_encoder",
    "scheduler",
    "unet",
    "vae",
    "torch_device",
    "imageHeight",
    "imageWidth",
    "guidance_scale",
    "model_path",
    "dash_tunnel",
    "pipe",
]
