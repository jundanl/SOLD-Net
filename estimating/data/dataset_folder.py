import os
import numpy as np
import imageio
import glob
import torch
from torch.utils.data import Dataset
import cv2
import warnings


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


class FolderDataset(Dataset):
    IMG_POSTFIX = ["*.png", "*.jpg", "*.jpeg", "*.JPEG", "*.JPG", "*.PNG"]

    def __init__(self, opt, dataroot):
        self.opt = opt
        self.dataroot = dataroot
        self.items = []

        for img_postfix in self.IMG_POSTFIX:
            img_list = glob.glob(os.path.join(self.dataroot, img_postfix))
            self.items += img_list
        self.items = sorted(self.items)
        print(f"{len(self.items)} images in folder: {self.dataroot}")

    def get_item(self, idx):
        persp_ldr_path = self.items[idx]
        persp_ldr_filename = os.path.basename(persp_ldr_path).split(".")[0]
        img_idx = persp_ldr_filename
        persp = read_image(persp_ldr_path, type="numpy", inf_v=0.0, nan_v=0.0,
                           preserve_alpha=True)[:, :, :3]  # H * W * C
        persp = np.transpose(np.asfarray(persp, dtype=np.float32), (2, 0, 1))  # C * H * W
        local_pos_path = persp_ldr_path.replace("persp.JPG", "local.npy")
        assert os.path.exists(local_pos_path), f"Not found local position file: {local_pos_path}"
        local_pos = np.load(local_pos_path)
        return_dict = {
            'color': persp,
            'local_pos': local_pos,
            'meta': img_idx
        }
        return return_dict

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.get_item(idx)
