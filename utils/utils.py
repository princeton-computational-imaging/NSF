import matplotlib.pyplot as plt
import numpy as np
import re
import torch
import torch.nn.functional as F

@torch.no_grad()
def raw_to_rgb(bundle):
    """ Convert RAW mosaic into three-channel RGB volume
        by only in-filling empty pixels.
        Returns volume of shape: (T, C, H, W)
    """ 

    raw_frames = torch.tensor(np.array([bundle[f'raw_{i}']['raw'] for i in range(bundle['num_raw_frames'])]).astype(np.int32), dtype=torch.float32)[None]  # C,T,H,W
    raw_frames = raw_frames.permute(1,0,2,3)  # T,C,H,W
    color_correction_gains = bundle['raw_0']['android']['colorCorrection.gains']
    color_correction_gains = np.array([float(el) for el in re.sub(r'[^0-9.,]', '', color_correction_gains).split(',')]) # RGGB gains
    color_filter_arrangement = bundle['characteristics']['color_filter_arrangement']
    blacklevel = torch.tensor(np.array([bundle[f'raw_{i}']['blacklevel'] for i in range(bundle['num_raw_frames'])]))[:,:,None,None]
    whitelevel = torch.tensor(np.array([bundle[f'raw_{i}']['whitelevel'] for i in range(bundle['num_raw_frames'])]))[:,None,None,None]
    shade_maps = torch.tensor(np.array([bundle[f'raw_{i}']['shade_map'] for i in range(bundle['num_raw_frames'])])).permute(0,3,1,2) # T,C,H,W
    # interpolate to size of image
    shade_maps = F.interpolate(shade_maps, size=(raw_frames.shape[-2]//2, raw_frames.shape[-1]//2), mode='bilinear', align_corners=False)

    top_left = raw_frames[:,:,0::2,0::2]
    top_right = raw_frames[:,:,0::2,1::2]
    bottom_left = raw_frames[:,:,1::2,0::2]
    bottom_right = raw_frames[:,:,1::2,1::2]

    # figure out color channels
    if color_filter_arrangement == 0: # RGGB
        R, G1, G2, B = top_left, top_right, bottom_left, bottom_right
    elif color_filter_arrangement == 1: # GRBG
        G1, R, B, G2 = top_left, top_right, bottom_left, bottom_right
    elif color_filter_arrangement == 2: # GBRG
        G1, B, R, G2 = top_left, top_right, bottom_left, bottom_right
    elif color_filter_arrangement == 3: # BGGR
        B, G1, G2, R = top_left, top_right, bottom_left, bottom_right

    # apply color correction gains, flip to portrait
    R = ((R - blacklevel[:,0:1]) / (whitelevel - blacklevel[:,0:1]) * color_correction_gains[0]) 
    R *= shade_maps[:,0:1]
    G1 = ((G1 - blacklevel[:,1:2]) / (whitelevel - blacklevel[:,1:2]) * color_correction_gains[1])
    G1 *= shade_maps[:,1:2] 
    G2 = ((G2 - blacklevel[:,2:3]) / (whitelevel - blacklevel[:,2:3]) * color_correction_gains[2]) 
    G2 *= shade_maps[:,2:3]  
    B = ((B - blacklevel[:,3:4]) / (whitelevel - blacklevel[:,3:4]) * color_correction_gains[3]) 
    B *= shade_maps[:,3:4] 

    rgb_volume = torch.zeros(raw_frames.shape[0], 3, raw_frames.shape[-2], raw_frames.shape[-1], dtype=torch.float32)

    # Fill gaps in blue channel
    rgb_volume[:, 2, 0::2, 0::2] = B.squeeze(1)
    rgb_volume[:, 2, 0::2, 1::2] = (B + torch.roll(B, -1, dims=3)).squeeze(1) / 2
    rgb_volume[:, 2, 1::2, 0::2] = (B + torch.roll(B, -1, dims=2)).squeeze(1) / 2
    rgb_volume[:, 2, 1::2, 1::2] = (B + torch.roll(B, -1, dims=2) + torch.roll(B, -1, dims=3) + torch.roll(B, [-1, -1], dims=[2, 3])).squeeze(1) / 4

    # Fill gaps in green channel
    rgb_volume[:, 1, 0::2, 0::2] = G1.squeeze(1)
    rgb_volume[:, 1, 0::2, 1::2] = (G1 + torch.roll(G1, -1, dims=3) + G2 + torch.roll(G2, 1, dims=2)).squeeze(1) / 4
    rgb_volume[:, 1, 1::2, 0::2] = (G1 + torch.roll(G1, -1, dims=2) + G2 + torch.roll(G2, 1, dims=3)).squeeze(1) / 4
    rgb_volume[:, 1, 1::2, 1::2] = G2.squeeze(1)

    # Fill gaps in red channel
    rgb_volume[:, 0, 0::2, 0::2] = R.squeeze(1)
    rgb_volume[:, 0, 0::2, 1::2] = (R + torch.roll(R, -1, dims=3)).squeeze(1) / 2
    rgb_volume[:, 0, 1::2, 0::2] = (R + torch.roll(R, -1, dims=2)).squeeze(1) / 2
    rgb_volume[:, 0, 1::2, 1::2] = (R + torch.roll(R, -1, dims=2) + torch.roll(R, -1, dims=3) + torch.roll(R, [-1, -1], dims=[2, 3])).squeeze(1) / 4

    rgb_volume = torch.flip(rgb_volume.transpose(-1,-2), [-1]) # rotate 90 degrees clockwise to portrait mode
    
    return rgb_volume
    
def de_item(bundle):
    """ Call .item() on all dictionary items
        removes unnecessary extra dimension
    """

    bundle['motion'] = bundle['motion'].item()
    bundle['characteristics'] = bundle['characteristics'].item()
        
    for i in range(bundle['num_raw_frames']):
        bundle[f'raw_{i}'] = bundle[f'raw_{i}'].item()

def mask(encoding, mask_coef):
    mask_coef = 0.4 + 0.6*mask_coef
    # interpolate to size of encoding
    mask = torch.zeros_like(encoding[0:1])
    mask_ceil = int(np.ceil(mask_coef * encoding.shape[1]))
    mask[:,:mask_ceil] = 1.0
    
    return encoding * mask

def interpolate(signal, times):
    if signal.shape[-1] == 1:
        return signal.squeeze(-1)
    elif signal.shape[-1] == 2:
        return interpolate_linear(signal, times)
    else:
        return interpolate_cubic_hermite(signal, times)

@torch.jit.script
def interpolate_cubic_hermite(signal, times):
    # Interpolate a signal using cubic Hermite splines
    # signal: (B, C, T) or (B, T)
    # times: (B, T)

    if len(signal.shape) == 3:  # B,C,T
        times = times.unsqueeze(1)
        times = times.repeat(1, signal.shape[1], 1)

    N = signal.shape[-1]

    times_scaled = times * (N - 1)
    indices = torch.floor(times_scaled).long()

    # Clamping to avoid out-of-bounds indices
    indices = torch.clamp(indices, 0, N - 2)
    left_indices = torch.clamp(indices - 1, 0, N - 1)
    right_indices = torch.clamp(indices + 1, 0, N - 1)
    right_right_indices = torch.clamp(indices + 2, 0, N - 1)

    t = (times_scaled - indices.float())

    p0 = torch.gather(signal, -1, left_indices)
    p1 = torch.gather(signal, -1, indices)
    p2 = torch.gather(signal, -1, right_indices)
    p3 = torch.gather(signal, -1, right_right_indices)

    # One-sided derivatives at the boundaries
    m0 = torch.where(left_indices == indices, (p2 - p1), (p2 - p0) / 2)
    m1 = torch.where(right_right_indices == right_indices, (p2 - p1), (p3 - p1) / 2)

    # Hermite basis functions
    h00 = (1 + 2*t) * (1 - t)**2
    h10 = t * (1 - t)**2
    h01 = t**2 * (3 - 2*t)
    h11 = t**2 * (t - 1)

    interpolation = h00 * p1 + h10 * m0 + h01 * p2 + h11 * m1

    if len(signal.shape) == 3:  # remove extra singleton dimension
        interpolation = interpolation.squeeze(-1)

    return interpolation


@torch.jit.script
def interpolate_linear(signal, times):
    # Interpolate a signal using linear interpolation
    # signal: (B, C, T) or (B, T)
    # times: (B, T)

    if len(signal.shape) == 3:  # B,C,T
        times = times.unsqueeze(1)
        times = times.repeat(1, signal.shape[1], 1)

    # Scale times to be between 0 and N - 1
    times_scaled = times * (signal.shape[-1] - 1)

    indices = torch.floor(times_scaled).long()
    right_indices = (indices + 1).clamp(max=signal.shape[-1] - 1)

    t = (times_scaled - indices.float())

    p0 = torch.gather(signal, -1, indices)
    p1 = torch.gather(signal, -1, right_indices)

    # Linear basis functions
    h00 = (1 - t)
    h01 = t

    interpolation = h00 * p0 + h01 * p1

    if len(signal.shape) == 3:  # remove extra singleton dimension
        interpolation = interpolation.squeeze(-1)

    return interpolation

@torch.jit.script
def convert_quaternions_to_rot(quaternions):
    """ Convert quaternions (wxyz) to 3x3 rotation matrices.
        Adapted from: https://automaticaddison.com/how-to-convert-a-quaternion-to-a-rotation-matrix
    """

    qw, qx, qy, qz = quaternions[:,0], quaternions[:,1], quaternions[:,2], quaternions[:,3]
    
    R00 = 2 * ((qw * qw) + (qx * qx)) - 1
    R01 = 2 * ((qx * qy) - (qw * qz))
    R02 = 2 * ((qx * qz) + (qw * qy))
     
    R10 = 2 * ((qx * qy) + (qw * qz))
    R11 = 2 * ((qw * qw) + (qy * qy)) - 1
    R12 = 2 * ((qy * qz) - (qw * qx))
     
    R20 = 2 * ((qx * qz) - (qw * qy))
    R21 = 2 * ((qy * qz) + (qw * qx))
    R22 = 2 * ((qw * qw) + (qz * qz)) - 1
     
    R = torch.stack([R00, R01, R02, R10, R11, R12, R20, R21, R22], dim=-1)
    R = R.reshape(-1,3,3)
               
    return R

def multi_interp(x, xp, fp):
    """ Simple extension of np.interp for independent
        linear interpolation of all axes of fp
        sample signal fp with timestamps xp at new timestamps x
    """
    if torch.is_tensor(fp):
        out = [torch.tensor(np.interp(x, xp, fp[:,i]), dtype=fp.dtype) for i in range(fp.shape[-1])]
        return torch.stack(out, dim=-1)
    else:
        out = [np.interp(x, xp, fp[:,i]) for i in range(fp.shape[-1])]
        return np.stack(out, axis=-1)
    
def parse_ccm(s):
    ccm = torch.tensor([eval(x.group()) for x in re.finditer(r"[-+]?\d+/\d+|[-+]?\d+\.\d+|[-+]?\d+", s)])
    ccm = ccm.reshape(3,3)
    return ccm

def parse_tonemap_curve(data_string):
    channels = re.findall(r'(R|G|B):\[(.*?)\]', data_string)
    result_array = np.zeros((3, len(channels[0][1].split('),')), 2))

    for i, (_, channel_data) in enumerate(channels):
        pairs = channel_data.split('),')
        for j, pair in enumerate(pairs):
            x, y = map(float, re.findall(r'([\d\.]+)', pair))
            result_array[i, j] = (x, y)
    return result_array

def apply_tonemap_curve(image, tonemap):
    # apply tonemap curve to each color channel
    image_toned = image.clone().cpu().numpy()
    
    for i in range(3):
        x_vals, y_vals = tonemap[i][:, 0], tonemap[i][:, 1]
        image_toned[i] = np.interp(image_toned[i], x_vals, y_vals)

    # Convert back to PyTorch tensor
    image_toned = torch.tensor(image_toned, dtype=torch.float32)
    
    return image_toned
        
def debatch(batch):
    """ Collapse batch and channel dimension together
    """
    debatched = []
    
    for x in batch:
        if len(x.shape) <=1:
            raise Exception("This tensor is to small to debatch.")
        elif len(x.shape) == 2:
            debatched.append(x.reshape(x.shape[0] * x.shape[1]))
        else:
            debatched.append(x.reshape(x.shape[0] * x.shape[1], *x.shape[2:]))

    return debatched
    
def colorize_tensor(value, vmin=None, vmax=None, cmap=None, colorbar=False, height=9.6, width=7.2):
    """ Convert tensor to 3 channel RGB array according to colors from cmap
        similar usage as plt.imshow
    """
    assert len(value.shape) == 2 # H x W
    
    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(width,height)
    a = ax.imshow(value.detach().cpu(), vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_axis_off()
    if colorbar:
        cbar = plt.colorbar(a, fraction=0.05)
        cbar.ax.tick_params(labelsize=30)
    plt.tight_layout()
    plt.close()
    
    # Draw figure on canvas
    fig.canvas.draw()

    # Convert the figure to numpy array, read the pixel values and reshape the array
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Normalize into 0-1 range for TensorBoard(X). Swap axes for newer versions where API expects colors in first dim
    img = img / 255.0
    
    return torch.tensor(img).permute(2,0,1).float()
