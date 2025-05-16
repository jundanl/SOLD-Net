# By Jundan Luo

import os
import glob
import json
import warnings

import cv2
import torch
from torch.utils.data import Dataset
import numpy as np


def read_image(path: str, type: str, inf_v=None, nan_v=None, preserve_alpha=False):
    """ Read image from path """
    MAX_8bit = 255.0
    MAX_16bit = 65535.0
    # Read image
    assert os.path.exists(path), f"image {path} not exists."
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED|cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH)
    assert img is not None, f"read image {path} failed."
    # Convert to float32 [0, 1]
    if img.dtype == np.uint16:
        img = img.astype(np.float32) / MAX_16bit
    elif img.dtype == np.uint8:
        img = img.astype(np.float32) / MAX_8bit
    elif img.dtype == np.float32:
        pass
    else:
        assert False, f"Not supported dtype {img.dtype} for image {path}."
    # check infinities
    if np.any(np.isinf(img)):
        if inf_v is None:
            warnings.warn(f"image {path} contains Inf values.")
        else:
            img[np.isinf(img)] = inf_v  # set inf to a specific value
    # check NaNs
    if np.any(np.isnan(img)):
        if nan_v is None:
            assert False, f"image {path} contains NaN values."
        else:
            img[np.isnan(img)] = nan_v  # set nan to a specific value
    # Check image shape and convert to RGB
    assert img.ndim < 4, f"Image should be 2D or 3D, but got {img.ndim}."
    if img.ndim == 3:
        if img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img.shape[-1] == 4:
            if not preserve_alpha:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        assert img.shape[-1] == 3 or img.shape[-1] == 1 or (img.shape[-1] == 4 and preserve_alpha), \
            f"image {path} should be gray or RGB or RGBA({preserve_alpha}), but got shape {img.shape}."
    elif img.ndim == 2:
        img = img[:, :, np.newaxis]  # HW -> HWC
    else:
        assert False, f"image {path} has wrong dimension {img.ndim}."
    # Convert to specified type
    if type == "numpy":
        pass
    elif type == "tensor":
        img = torch.from_numpy(img.copy()).to(torch.float32).permute(2, 0, 1)  # HWC -> CHW
    else:
        raise NotImplementedError(f"Type {type} is not implemented.")
    return img


def numpy_to_tensor(img):
    assert isinstance(img, np.ndarray), "img should be np.ndarray"
    if img.ndim == 3:  # HWC
        img = np.transpose(img, (2, 0, 1))  # CHW
    elif img.ndim == 4:  # BHWC
        img = np.transpose(img, (0, 3, 1, 2))  # BCHW
    else:
        assert False
    return torch.from_numpy(img).contiguous().to(torch.float32)


class ImagePostfix:
    input = "input_srgb.png"
    depth = "depth.exr"
    normal = "normal.exr"
    albedo = "albedo.exr"
    shading = "shading.exr"


BACKGROUND_DIR: str = "background"
CAMERA_FILE: str = "camera_params.json"
SCENE_FILE: str = "scene_information.json"
POINT_CLOUD_POSTFIX: str = "pc.npy"


class LightSORDataset(Dataset):
    GAMMA = 1. / 2.2

    ImagePostfix = ImagePostfix
    image_postfix = ImagePostfix()

    BACKGROUND_DIR = BACKGROUND_DIR
    CAMERA_FILE = CAMERA_FILE
    SCENE_FILE = SCENE_FILE
    POINT_CLOUD_POSTFIX = POINT_CLOUD_POSTFIX
    VALID_DEPTH_RANGE = [0.1, 50.0]

    def __init__(self, data_root, only_sunlight_scene,
                 load_gt_images=[],
                 load_information=[]):
        """
        :param data_root:
        :param only_sunlight_scene:
        :param load_gt_images: "depth", "normal", "albedo", "shading"
        :param load_information: "camera", "scene", "object_gt_images", "point_cloud"
        """

        self.data_root = data_root
        self.only_sunlight_scene = only_sunlight_scene
        self.load_gt_images = [] if load_gt_images is None else load_gt_images
        self.load_information = [] if load_information is None else load_information
        if "object_gt_images" in self.load_information:
            if "scene" not in self.load_information:  # object_gt_images information is stored in scene information
                self.load_information.append("scene")

        # Get the scene list
        scene_list = os.listdir(self.data_root)
        scene_list = [scene for scene in scene_list if os.path.isdir(os.path.join(self.data_root, scene))]
        scene_list.sort()

        # Image path and output_name
        self.bk_img_list = []
        self.scene_list = []  # exclude not needed scenes
        for scene in scene_list:
            bk_img_dir = os.path.join(self.data_root, scene, self.BACKGROUND_DIR)
            input_srgb_path = glob.glob(os.path.join(bk_img_dir, f"*{self.image_postfix.input}"))
            assert len(input_srgb_path) == 1, \
                f"Found {len(input_srgb_path)} images in {bk_img_dir}: {input_srgb_path}, expected 1"
            input_srgb_path = input_srgb_path[0]
            assert os.path.exists(input_srgb_path), f"Image not found: {input_srgb_path}"

            if self.only_sunlight_scene:
                scene_info_path = os.path.join(bk_img_dir, self.SCENE_FILE)
                assert os.path.exists(scene_info_path), f"Scene information not found: {scene_info_path}."
                scene_info = json.load(open(scene_info_path, "r"))
                if "primary_light_params" in scene_info:
                    primary_light_info = scene_info["primary_light_params"]
                else:
                    primary_light_info = None
                if "type" not in primary_light_info:
                    print(f"Primary light type not found in {scene_info_path}. Skip this scene.")
                    continue
                if primary_light_info["type"].lower() != "sun":
                    print(f"Primary light type is not sun in {scene_info_path}. Skip this scene.")
                    continue
            self.bk_img_list.append(input_srgb_path)
            self.scene_list.append(os.path.join(self.data_root, scene))
        assert len(self.bk_img_list) == len(self.scene_list), "bk_img_list and scene_list should have the same length."
        print(f"Found {len(self.scene_list)} scenes in {self.data_root}.")

    def __len__(self):
        return len(self.bk_img_list)

    def get_idx_by_scene_name(self, scene_name):
        for idx, bk_srgb_image_path in enumerate(self.bk_img_list):
            _, _, curr_scene_name = self.split_bk_srgb_img_path(bk_srgb_image_path)
            if curr_scene_name == scene_name:
                return idx
        return None

    def split_bk_srgb_img_path(self, bk_srgb_image_path):
        bk_dir = os.path.dirname(bk_srgb_image_path)
        scene_dir = os.path.dirname(bk_dir)
        scene_name = os.path.basename(scene_dir)
        return bk_dir, scene_dir, scene_name

    def __getitem__(self, idx):
        # Read the input (background) srgb image
        bk_srgb_image_path = self.bk_img_list[idx]
        bk_srgb_image = read_image(bk_srgb_image_path, "tensor", inf_v=0.0, nan_v=0.0,
                                              preserve_alpha=True)[:3]  # C x H x W

        # Paths
        bk_dir, scene_dir, scene_name = self.split_bk_srgb_img_path(bk_srgb_image_path)
        # check scene_dir and self.scene_list[idx] are the same, using absolute path
        assert os.path.abspath(scene_dir) == os.path.abspath(self.scene_list[idx]), \
            f"scene_dir and self.scene_list[idx] are not the same: {scene_dir}, {self.scene_list[idx]}"
        img_name = scene_name  # img_name is the same as scene_name

        # Output dict - 1
        out_dict = {
            "index": idx,
            "scene_path": scene_dir,
            "img_name": img_name,
            "scene_name": scene_name,
            "bk_dir_path": bk_dir,
            "bk_srgb_image_path": bk_srgb_image_path,
            "bk_srgb_image": bk_srgb_image,
        }

        # Load camera information
        if "camera" in self.load_information:
            camera_info_path = os.path.join(bk_dir, self.CAMERA_FILE)
            assert os.path.exists(camera_info_path), f"Camera information not found: {camera_info_path}."
            camera_info = json.load(open(camera_info_path, "r"))
        else:
            camera_info_path = None
            camera_info = None

        # Load scene information
        if "scene" in self.load_information:
            scene_info_path = os.path.join(bk_dir, self.SCENE_FILE)
            assert os.path.exists(scene_info_path), f"Scene information not found: {scene_info_path}."
            scene_info = json.load(open(scene_info_path, "r"))
            if "object_params" in scene_info:
                object_info = scene_info["object_params"]
                object_list = object_info.keys()
                if "object_gt_images" in self.load_information:
                    object_gt_paths = {}
                    object_mask_paths = {}
                    for obj_name in object_list:
                        gt_obj_img_path = os.path.join(self.data_root, scene_name, f"image_{obj_name}", "Image0000_input_srgb.png")
                        assert os.path.exists(gt_obj_img_path), f"Object image not found: {gt_obj_img_path}"
                        object_gt_paths[obj_name] = gt_obj_img_path
                        mask_obj_img_path = os.path.join(self.data_root, scene_name, f"image_{obj_name}", "Image0000_object_alpha.png")
                        assert os.path.exists(mask_obj_img_path), f"Object mask image not found: {mask_obj_img_path}"
                        object_mask_paths[obj_name] = mask_obj_img_path
                else:
                    object_gt_paths = None
                    object_mask_paths = None
            else:
                assert "object_gt_images" not in self.load_information, \
                    f"scene_info does not have object_params. Cannot load object_gt_images."
                object_info = None
                object_list = []
                object_gt_paths = None
                object_mask_paths = None
            if "primary_light_params" in scene_info:
                primary_light_info = scene_info["primary_light_params"]
            else:
                primary_light_info = None
        else:
            scene_info_path = None
            scene_info = None
            object_info = None
            object_list = []
            object_gt_paths = None
            object_mask_paths = None
            primary_light_info = None

        # Load point cloud
        if "point_cloud" in self.load_information:
            point_cloud_path = bk_srgb_image_path.replace(self.image_postfix.input, self.POINT_CLOUD_POSTFIX)
            assert os.path.exists(point_cloud_path), f"Point cloud not found: {point_cloud_path}."
            point_cloud_map = np.load(point_cloud_path)
            point_cloud_map = numpy_to_tensor(point_cloud_map)
        else:
            point_cloud_path = None
            point_cloud_map = None

        # Output dict - 2
        out_dict.update({
            "camera_info_path": camera_info_path,
            "camera_info": camera_info,
            "scene_info_path": scene_info_path,
            "scene_info": scene_info,
            "object_info": object_info,
            "object_list": object_list,
            "object_gt_paths": object_gt_paths,
            "object_mask_paths": object_mask_paths,
            "primary_light_info": primary_light_info,
            "point_cloud_path": point_cloud_path,
            "point_cloud_map": point_cloud_map,
        })

        # Image path dict
        img_path_dict = {
            "albedo": bk_srgb_image_path.replace(self.image_postfix.input, self.image_postfix.albedo),
            "shading": bk_srgb_image_path.replace(self.image_postfix.input, self.image_postfix.shading),
            "depth": bk_srgb_image_path.replace(self.image_postfix.input, self.image_postfix.depth),
            "normal": bk_srgb_image_path.replace(self.image_postfix.input, self.image_postfix.normal),
        }

        # Load images
        for type in self.load_gt_images:
            type = type.lower()
            assert type in img_path_dict, f"Invalid image type: {type}."
            image = read_image(img_path_dict[type], "tensor", inf_v=0.0, nan_v=0.0,
                                          preserve_alpha=True)
            if type in ["depth"]:
                image = image[:1]  # 1 x H x W
                image = image * (image >= self.VALID_DEPTH_RANGE[0]) * (image <= self.VALID_DEPTH_RANGE[1])
            else:
                image = image[:3]  # 3 x H x W
            if type == "normal":
                valid_normal_mask = (image ** 2).sum(dim=0, keepdim=True) > 1e-5
                image = image * 2.0 - 1.0  # from [0, 1] to [-1, 1]
                image = image / (image ** 2).sum(dim=0, keepdim=True).sqrt().clamp(min=1e-6)  # normalize
                image = image * valid_normal_mask  # remove invalid normal
            out_dict[f"{type}_image"] = image
        return out_dict