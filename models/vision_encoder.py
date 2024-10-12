from typing import List, Tuple

import PIL
import torch
from torch import nn
from torchvision import transforms
from transformers import CLIPImageProcessor, CLIPVisionModel
from diffusers import AutoencoderKL

from models.img_utils import resize_and_pad_image


class CLIPVisionEncoder(nn.Module):
    def __init__(self, pretrained_path):
        super().__init__()

        self.image_processor = CLIPImageProcessor.from_pretrained(pretrained_path)
        self.vision_tower = CLIPVisionModel.from_pretrained(pretrained_path)
        self.hidden_size = self.vision_tower.config.hidden_size
        self.vision_tower.requires_grad_(False)

    def feature_select(self, image_forward_out):
        image_features = image_forward_out.hidden_states[-2]  # by default use the second last layer features
        image_features = image_features[:, 1:]
        return image_features
    
    @torch.no_grad()
    def forward(self, images):
        images = self.image_processor.preprocess(images, return_tensors="pt")['pixel_values']
        if isinstance(images, List):
            images = torch.stack(images)
        images = images.to(self.device).to(self.dtype)

        image_features = []
        for image in images:
            image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
            image_feature = self.feature_select(image_forward_out).to(image.dtype)[0]
            image_features.append(image_feature)

        return torch.stack(image_features)
    
    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device


class VAEVisionEncoder(nn.Module):
    def __init__(self, pretrained_path, patch_size=2, resolution=256):
        super().__init__()
        self.patch_size = patch_size
        self.image_transform = transforms.Compose([
            transforms.Lambda(lambda pil_img: resize_and_pad_image(pil_img, resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        self.vision_tower = AutoencoderKL.from_pretrained(pretrained_path)
        self.hidden_size = self.vision_tower.config.latent_channels * patch_size * patch_size
        self.vision_tower.requires_grad_(False)

    def image_processor(self, images):
        if isinstance(images, PIL.Image.Image):
            images = [images]
        
        output_images = []
        for image in images:
            image = self.image_transform(image)
            output_images.append(image)

        return torch.stack(output_images)
    
    @torch.no_grad()
    def forward(self, images):
        images = self.image_processor(images).to(self.device).to(self.dtype)
        image_features = []
        image_forward_out = self.vision_tower.encode(images).latent_dist.sample().mul_(0.18215).to(self.device)
        for image in image_forward_out:
            image_features.append(image)
        
        image_features = self.patchify_and_embed(image_features)
        
        return image_features
    
    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    def patchify_and_embed(self, x: List[torch.Tensor]) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        if len(x) == 0:
            return []

        pH = pW = self.patch_size
        x_embed = []

        for img in x:
            C, H, W = img.size()
            img = img.view(C, H // pH, pH, W // pW, pW).permute(1, 3, 0, 2, 4).flatten(2)
            img = img.flatten(0, 1)
            x_embed.append(img)
        
        x_embed = torch.stack(x_embed, dim=0)
        
        return x_embed


def build_vision_encoder(vision_encoder, pretrained_path, **kwargs):
    if vision_encoder == "clip":
        return CLIPVisionEncoder(pretrained_path)
    elif vision_encoder == "vae":
        return VAEVisionEncoder(pretrained_path, **kwargs)
    else:
        raise NotImplementedError(f"{vision_encoder} not implemented")
