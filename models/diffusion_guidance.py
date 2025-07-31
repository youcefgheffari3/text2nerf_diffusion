import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image

class StableDiffusionSDS:
    def __init__(self, prompt, guidance_scale=100.0, device="cuda"):
        self.device = device
        self.guidance_scale = guidance_scale
        self.prompt = prompt

        # Load Stable Diffusion
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None
        ).to(self.device)
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

        # Only use the UNet + text encoder
        self.unet = self.pipe.unet.eval()
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder.eval()
        self.vae = self.pipe.vae.eval()

        for m in [self.unet, self.text_encoder, self.vae]:
            for p in m.parameters():
                p.requires_grad = False

        # Tokenize prompt
        text_input = self.tokenizer(prompt, padding="max_length", max_length=77, return_tensors="pt")
        self.embeds = self.text_encoder(text_input.input_ids.to(self.device))[0]

    def get_sds_loss(self, rgb_render: torch.Tensor):
        """
        Compute SDS loss from rendered image using Stable Diffusion.
        rgb_render: (1, 3, H, W), values in [0, 1]
        """
        image = (rgb_render * 2 - 1).clamp(-1, 1)  # to [-1, 1] for VAE
        latents = self.vae.encode(image).latent_dist.sample() * 0.18215  # scale factor from SD docs

        noise = torch.randn_like(latents)
        timesteps = torch.randint(50, 200, (1,), dtype=torch.long).to(self.device)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timesteps)

        # Predict noise with classifier-free guidance
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states=self.embeds).sample

        loss = torch.mean((noise_pred - noise) ** 2)
        return loss
