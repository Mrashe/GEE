#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from gaussiansplatting.scene.dataset_readers import sceneLoadTypeCallbacks
from gaussiansplatting.utils.camera_utils import cameraList_load
from gaussiansplatting.scene.dataset import FourDGSdataset


class CamScene:
    def __init__(self, source_path, h=512, w=512, aspect=-1):
        """b
        :param path: Path to colmap scene main folder.
        """

        if aspect != -1:
            h = 512
            w = 512 * aspect
        if os.path.exists(os.path.join(source_path, "sparse")):
            if h == -1 or w == -1:
                scene_info = sceneLoadTypeCallbacks["Colmap"](source_path, None, False)
                h = scene_info.train_cameras[0].height
                w = scene_info.train_cameras[0].width
                if w > 1920:
                    scale = w / 1920
                    h /= scale
                    w /= scale
            else:
                scene_info = sceneLoadTypeCallbacks["Colmap_hw"](source_path, h, w, None, False)
            self.cameras_extent = scene_info.nerf_normalization["radius"]
            self.cameras = cameraList_load(scene_info.train_cameras, h, w)
        elif os.path.exists(os.path.join(source_path, "scene_info1.pth")):
            scene_info = sceneLoadTypeCallbacks["readyinfo"](source_path)
            self.cameras_extent = scene_info.nerf_normalization["radius"]
            self.cameras = FourDGSdataset(scene_info.train_cameras, h)
        elif os.path.exists(os.path.join(source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](source_path, True, True)
            self.cameras_extent = scene_info.nerf_normalization["radius"]
            self.cameras = FourDGSdataset(scene_info.train_cameras, h)
        else:
            assert False, "Could not recognize scene type!"

        # self.cameras_extent = scene_info.nerf_normalization["radius"]
        # self.cameras = cameraList_load(scene_info.train_cameras, h, w)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
