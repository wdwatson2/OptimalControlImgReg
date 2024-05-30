import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint

class VelocityFieldNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, t):
        # t is a scalar, so we need to expand it to match the batch size of x
        t_expOutsanded = t.expand_as(x[:, :1])
        x = torch.cat([x, t_expOutsanded], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def bicubic_interpolation(image, coords, device):
    """
    Perform bicubic interpolation on the image at the given coordinates.
    :param image: Tensor of shape (C, H, W)
    :param coords: Tensor of shape (N, 2) where N is the number of coordinates
    :return: Interpolated values at the coordinates

    Assumes image domain is on [-1,1]
    """
    grid = coords.unsqueeze(0).unsqueeze(0).to(device)
    interpolated = nn.functional.grid_sample(image.unsqueeze(0).to(device), grid, mode='bicubic', align_corners=True)
    return interpolated.squeeze()

def bilinear_interpolation(image, coords, device):
    """
    Perform bicubic interpolation on the image at the given coordinates.
    :param image: Tensor of shape (C, H, W)
    :param coords: Tensor of shape (N, 2) where N is the number of coordinates
    :return: Interpolated values at the coordinates

    Assumes image domain is on [-1,1]
    """
    grid = coords.unsqueeze(0).unsqueeze(0).to(device)
    interpolated = nn.functional.grid_sample(image.unsqueeze(0).to(device), grid, mode='bilinear', align_corners=True)
    return interpolated.squeeze()

def loss_function(v_net, x, CR, CT, device):
    """
    In the forward pass: 
    1. Approximate the terminal solution to the ODE to get transformation
    2. Use that solution to approximate the inverse transformation
    3. Evaluate the transformed template image using inverse sampling
    4. Compute loss
    """
    gridsize = int(x.shape[0] ** (1/2))
    def odefunc(t, z):
        return v_net(z, t)
    t = torch.tensor([0.0, 1.0], device=device)
    z_T = odeint(odefunc, x, t)[-1]
    def inv_odefunc(t, z):
        return -v_net(z, t)
    z_inv_T = odeint(inv_odefunc, z_T, torch.flip(t, dims=[0]))[0]
    CT_z_inv_T = bilinear_interpolation(CT, z_inv_T, device).view(1, gridsize, gridsize).transpose(1, 2)
    CR_x = bilinear_interpolation(CR, x, device).view(1, gridsize, gridsize).transpose(1, 2)
    loss = 0.5 * torch.mean((CT_z_inv_T - CR_x) ** 2)
    return loss

import matplotlib.pyplot as plt
import torch
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

# Taken from torchvision documentation

def plot(imgs, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            boxes = None
            masks = None
            if isinstance(img, tuple):
                img, target = img
                if isinstance(target, dict):
                    boxes = target.get("boxes")
                    masks = target.get("masks")
                elif isinstance(target, tv_tensors.BoundingBoxes):
                    boxes = target
                else:
                    raise ValueError(f"Unexpected target type: {type(target)}")
            img = F.to_image(img)
            if img.dtype.is_floating_point and img.min() < 0:
                # Poor man's re-normalization for the colors to be OK-ish. This
                # is useful for images coming out of Normalize()
                img -= img.min()
                img /= img.max()

            img = F.to_dtype(img, torch.uint8, scale=True)
            if boxes is not None:
                img = draw_bounding_boxes(img, boxes, colors="yellow", width=3)
            if masks is not None:
                img = draw_segmentation_masks(img, masks.to(torch.bool), colors=["green"] * masks.shape[0], alpha=.65)

            ax = axs[row_idx, col_idx]
            ax.imshow(img.permute(1, 2, 0).numpy(), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()