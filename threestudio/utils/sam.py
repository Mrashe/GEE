import argparse
import json
from pathlib import Path
from PIL import Image
import torch
from einops import rearrange
from torchvision.transforms import ToPILImage, ToTensor

from lang_sam import LangSAM

# from threestudio.utils.typing import *


class LangSAMTextSegmentor(torch.nn.Module):
    def __init__(self, sam_type="vit_h"):
        super().__init__()
        self.model = LangSAM(sam_type)
        xs=34
        self.to_pil_image = ToPILImage(mode="RGB")
        self.to_tensor = ToTensor()

    def forward(self, images, prompt: str):
        images = rearrange(images, "b h w c -> b c h w")
        masks = []
        boxes = []
        for image in images:
            # breakpoint()
            image = self.to_pil_image(image.clamp(0.0, 1.0))
            mask,box, _, _ = self.model.predict(image, prompt)
            # breakpoint()
            if mask.ndim == 3:
                masks.append(mask[0:1].to(torch.float32))
                # print(box.shape)
                boxes.append(box[0].unsqueeze(0))
            else:
                print(f"None {prompt} Detected")
                # masks.append(torch.zeros_like(images[0, 0:1]).cpu())
                masks.append(torch.ones_like(images[0, 0:1]).cpu())
                boxes.append(torch.tensor([[0, 0, 0, 0]]).cpu())
                #save image
                image.save(f"{prompt}.jpg")
        
        M = torch.stack(masks, dim=0)
        B = torch.stack(boxes, dim=0)
        # print(boxes[0].device,boxes[1].device,masks[0].device,mask.device)
        return torch.stack(masks, dim=0),torch.stack(boxes, dim=0)

if __name__ == "__main__":
    model = LangSAMTextSegmentor()

    image = Image.open("load/lego_bulldozer.jpg")
    prompt = "a lego bulldozer"

    image = ToTensor()(image)

    image = image.unsqueeze(0)

    mask = model(image, prompt)

    breakpoint()
