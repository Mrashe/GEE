from dataclasses import dataclass, field

from tqdm import tqdm

from datetime import datetime



import torch
import threestudio
import os
from torchvision.utils import save_image
from threestudio.utils.clip_metrics import ClipSimilarity

from threestudio.systems.GassuianEditor import GaussianEditor
import numpy as np

import cv2
import torch.nn.functional as F

def scale(tensor, scale_factor=1.5):
    # 提取左下顶点和右上顶点坐标
    y0, x0, y1, x1 = tensor.view(-1)

    # 计算矩阵的中心点坐标
    center_y = (y0 + y1) / 2.0
    center_x = (x0 + x1) / 2.0

    # 将矩阵的中心移到原点
    shifted_y0 = y0 - center_y
    shifted_x0 = x0 - center_x
    shifted_y1 = y1 - center_y
    shifted_x1 = x1 - center_x

    # 将矩阵的左下顶点和右上顶点坐标分别乘以 scale_factor
    scaled_y0 = shifted_y0 * scale_factor*1.2
    scaled_x0 = shifted_x0 * scale_factor*1.6
    scaled_y1 = shifted_y1 * scale_factor*1.4
    scaled_x1 = shifted_x1 * scale_factor

    # 将矩阵中心、左下顶点和右上顶点移回原来的位置
    # scaled_center_y = center_y * scale_factor
    # scaled_center_x = center_x * scale_factor
    scaled_y0 += center_y
    scaled_x0 += center_x
    scaled_y1 += center_y
    scaled_x1 += center_x

    return torch.tensor([scaled_y0, scaled_x0, scaled_y1, scaled_x1])
def warping(original_frames, edited_frames):
    # 确保输入张量在 CPU 上，以便转换为 NumPy 数组
    original_frames = original_frames.detach().cpu()
    edited_frames = edited_frames.detach().cpu()

    B, H, W, C = original_frames.shape  # 批次大小、高度、宽度、通道数
    warped_frames = torch.zeros_like(edited_frames)  # 初始化扭曲帧张量

    for i in range(B):
        # 提取单个帧并转换为 BGR 格式，因为 OpenCV 使用 BGR
        frame1 = original_frames[i].numpy()
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR)
        frame2 = original_frames[i + 1].numpy() if i < B - 1 else original_frames[i].numpy()
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2BGR)

        # 计算两帧之间的光流
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # 构造扭曲的网格
        flow = torch.from_numpy(flow).permute(2, 0, 1)  # 转换光流为 [2, H, W] 格式
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=-1)  # 形状 [H, W, 2]
        flow = flow / torch.tensor([W / 2, H / 2]).view(2, 1, 1)  # 归一化光流值
        displacement = grid + flow.permute(1, 2, 0)
        displacement = displacement.clamp(-1, 1)  # 限制扭曲网格在正常范围内

        # 使用 grid_sample 对 edited_frames 进行扭曲
        edited_frame = edited_frames[i].permute(2, 0, 1)  # 调整为 [C, H, W] 用于 grid_sample
        warped_frame = F.grid_sample(edited_frame.unsqueeze(0), displacement.unsqueeze(0), mode='bilinear', padding_mode='zeros', align_corners=True)
        warped_frames[i] = warped_frame.squeeze(0).permute(1, 2, 0)  # 转换回 [H, W, C] 格式

    return warped_frames.to('cuda')  # 返回扭曲帧，转回原始设备
@threestudio.register("gsedit-system-edit")
class GaussianEditor_Edit(GaussianEditor):
    @dataclass
    class Config(GaussianEditor.Config):
        local_edit: bool = False

        seg_prompt: str = "grass"

        second_guidance_type: str = "dds"
        second_guidance: dict = field(default_factory=dict)
        dds_target_prompt_processor: dict = field(default_factory=dict)
        dds_source_prompt_processor: dict = field(default_factory=dict)

        clip_prompt_origin: str = ""
        clip_prompt_target: str = ""  # only for metrics

    cfg: Config

    def configure(self) -> None:
        super().configure()
        if len(self.cfg.cache_dir) > 0:
            self.cache_dir = os.path.join("edit_cache", self.cfg.cache_dir)
        else:
            self.cache_dir = os.path.join("edit_cache", self.cfg.gs_source.replace("/", "-"))
        self.flag = True
        
        current_time = datetime.now()


        folder_name = current_time.strftime("%Y%m%d%H%M%S")
        output_folder = os.path.join("outputs", "model", folder_name)
        print(output_folder)
        os.makedirs(output_folder)
        self.modelfolder = output_folder


    def on_fit_start(self) -> None:
        super().on_fit_start()
        self.render_all_view(cache_name="origin_render")

        # if len(self.cfg.seg_prompt) > 0:
        #     self.update_mask()

        if len(self.cfg.prompt_processor) > 0:
            self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
                self.cfg.prompt_processor
            )
        if len(self.cfg.dds_target_prompt_processor) > 0:
            self.dds_target_prompt_processor = threestudio.find(
                self.cfg.prompt_processor_type
            )(self.cfg.dds_target_prompt_processor)
        if len(self.cfg.dds_source_prompt_processor) > 0:
            self.dds_source_prompt_processor = threestudio.find(
                self.cfg.prompt_processor_type
            )(self.cfg.dds_source_prompt_processor)
        if self.cfg.loss.lambda_l1 > 0 or self.cfg.loss.lambda_p > 0:
            self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        if self.cfg.loss.lambda_dds > 0:
            self.second_guidance = threestudio.find(self.cfg.second_guidance_type)(
                self.cfg.second_guidance
            )
            
    def training_step(self, batch, batch_idx):
        self.gaussian.update_learning_rate(self.true_global_step)

        batch_index = batch["index"]
        if isinstance(batch_index, int):
            batch_index = [batch_index]
        # print(self.gaussian.mask.shape,self.gaussian._xyz.shape,"shaaape")
        out = self(batch, local=self.cfg.local_edit,mask=self.gaussian.mask,deform = True)
   
        # out = self(batch, local=self.cfg.local_edit)
        images = out["comp_rgb"]
        tensor = images.permute(0, 3, 1, 2)
        masks,boxes = self.text_segmentor(images, self.cfg.seg_prompt)
        # print(masks[0].shape, "mask")
        self.update_mask(masks[0],0)
# # Save each image in the batch
        for i in range(tensor.shape[0]):
            save_image(tensor[i], f'img_{i}.png')
        raise Exception("stop")
        
        
        loss = 0.0
        if self.global_step % 200 == 0:
            self.gaussian.save_deformation(self.modelfolder,str(self.global_step))
            self.gaussian.save_ply(f"{self.modelfolder}/{str(self.global_step)}.ply")
        # nerf2nerf loss
        if self.cfg.loss.lambda_l1 > 0 or self.cfg.loss.lambda_p > 0:
            prompt_utils = self.prompt_processor()
            gt_images = []
           
            for img_index, cur_index in enumerate(batch_index):
                if cur_index not in self.edit_frames or (
                        self.cfg.per_editing_step > 0
                        and self.cfg.edit_begin_step
                        < self.global_step
                        < self.cfg.edit_until_step
                        and self.global_step % self.cfg.per_editing_step == 0
                ):
                    
                    edit_image = images[img_index]
                    if self.cfg.crop:
                        mask = masks[img_index][None]
                        box = scale(boxes[img_index][0])
                        if(box[1]==box[3]): 
                            print(img_index)
                            continue
                        croped = edit_image
                
                        croped = croped[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                    

                        original_croped = self.origin_frames[cur_index].squeeze()[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                   
                        result = self.guidance(
                        croped[None],#images[img_index][None],# croped,
                        original_croped[None],#self.origin_frames[cur_index]
                        prompt_utils,
                        )
                   
                        downsample = downsample_image(result["edit_images"].permute(0, 3, 1, 2), 1).squeeze().permute(1, 2, 0)
                        edit_image[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = downsample
                        self.edit_frames[cur_index] = edit_image[None].detach().clone()
                    # Create an image from the numpy array
                    else:
                        result = self.guidance(
                        images[img_index][None],# croped,
                        self.origin_frames[cur_index],
                        prompt_utils,
                    )
                        self.edit_frames[cur_index] = result["edit_images"].detach().clone()
                    
                
                gt_images.append(self.edit_frames[cur_index])

            if(len(gt_images) > 0):
                gt_dif = []
                dif = []
                for i in range(len(gt_images)-1):
                    gt_dif.append(torch.nn.functional.l1_loss(gt_images[i], gt_images[(i+1)%len(gt_images)]))
                    dif.append(torch.nn.functional.l1_loss(images[i], images[(i+1)%len(images)]))
                
                gt_images = torch.concatenate(gt_images, dim=0)
                # print( gt_images.shape, "gt_images")
                # warping_images = warping(gt_images, images)
                # print(gt_images[0].shape, type(gt_images[0]),"gt_images")
                # raise ValueError("stop")
                gt_dif = torch.tensor(gt_dif)
                dif = torch.tensor(dif)

                guidance_out = {
                    "loss_l1": torch.nn.functional.l1_loss(images, gt_images),
                    "loss_p": self.perceptual_loss(
                        images.permute(0, 3, 1, 2).contiguous(),
                        gt_images.permute(0, 3, 1, 2).contiguous(),
                ).sum(),
                    # "loss_dif": torch.nn.functional.l1_loss(warping_images, images),
                    "loss_dif": torch.nn.functional.l1_loss(dif, gt_dif),
                }
                for name, value in guidance_out.items():
                    self.log(f"train/{name}", value)
                    if name.startswith("loss_"):
                        loss += value * self.C(
                            self.cfg.loss[name.replace("loss_", "lambda_")]
                    )
                # raise ValueError("stop")

              

        # dds loss
        if self.cfg.loss.lambda_dds > 0:
            dds_target_prompt_utils = self.dds_target_prompt_processor()
            dds_source_prompt_utils = self.dds_source_prompt_processor()

            second_guidance_out = self.second_guidance(
                out["comp_rgb"],
                torch.concatenate(
                    [self.origin_frames[idx] for idx in batch_index], dim=0
                ),
                dds_target_prompt_utils,
                dds_source_prompt_utils,
            )
            for name, value in second_guidance_out.items():
                self.log(f"train/{name}", value)
                if name.startswith("loss_"):
                    loss += value * self.C(
                        self.cfg.loss[name.replace("loss_", "lambda_")]
                    )

        if not(
                self.cfg.loss.lambda_anchor_color > 0
                or self.cfg.loss.lambda_anchor_geo > 0
                or self.cfg.loss.lambda_anchor_scale > 0
                or self.cfg.loss.lambda_anchor_opacity > 0
        ):
            anchor_out = self.gaussian.anchor_loss()
            for name, value in anchor_out.items():
                self.log(f"train/{name}", value)
                if name.startswith("loss_"):
                    loss += value * self.C(
                        self.cfg.loss[name.replace("loss_", "lambda_")]
                    )

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))
        return {"loss": loss}

    def on_validation_epoch_end(self):
        if len(self.cfg.clip_prompt_target) > 0:
            self.compute_clip()

    def compute_clip(self):
        clip_metrics = ClipSimilarity().to(self.gaussian.get_xyz.device)
        total_cos = 0
        with torch.no_grad():
            for id in tqdm(self.view_list):
                cur_cam = self.trainer.datamodule.train_dataset.scene.cameras[id]
                cur_batch = {
                    "index": id,
                    "camera": [cur_cam],
                    "height": self.trainer.datamodule.train_dataset.height,
                    "width": self.trainer.datamodule.train_dataset.width,
                }
                out = self(cur_batch)["comp_rgb"]
                _, _, cos_sim, _ = clip_metrics(self.origin_frames[id].permute(0, 3, 1, 2), out.permute(0, 3, 1, 2),
                                                self.cfg.clip_prompt_origin, self.cfg.clip_prompt_target)
                total_cos += abs(cos_sim.item())
        print(self.cfg.clip_prompt_origin, self.cfg.clip_prompt_target, total_cos / len(self.view_list))
        self.log("train/clip_sim", total_cos / len(self.view_list))

import torch
from PIL import Image


def save_tensor_as_image(tensor, file_path):
    
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    tensor = (tensor - tensor_min) / (tensor_max - tensor_min) * 255.0
    tensor = tensor.type(torch.uint8)
    image_array = tensor.cpu().detach().numpy()

    image = Image.fromarray(image_array.astype('uint8'), 'RGB')

    image.save(file_path)
import cv2

import torch.nn.functional as F

def upsample_image(image, scale_factor):
    upscaled_image = F.interpolate(image, scale_factor=scale_factor, mode='bicubic')
    return upscaled_image
def downsample_image(image, scale_factor):
    if scale_factor == 1:
        return image
    downscaled_image = F.interpolate(image, scale_factor=1/scale_factor, mode='bicubic', align_corners=False)
    return downscaled_image


