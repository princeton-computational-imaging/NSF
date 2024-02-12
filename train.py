import argparse
import commentjson as json
import numpy as np
import os
import re
import pickle

import tinycudann as tcnn

from utils import utils
from utils.utils import debatch
import matplotlib.pyplot as plt

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl

#########################################################################################################
################################################ DATASET ################################################
#########################################################################################################

class BundleDataset(Dataset):
    def __init__(self, args, load_volume=False):
        self.args = args
        print("Loading from:", self.args.bundle_path)

        if self.args.rgb_data:
            self.init_RGB(self.args, load_volume)
        else:
            self.init_RAW(self.args, load_volume)

    def init_RGB(self, args, load_volume=False):

        self.lens_distortion = torch.tensor([0.0,0.0,0.0,0.0,0.0]).float()
        self.ccm = torch.tensor(np.eye(3)).float()
        self.tonemap_curve = torch.tensor(np.linspace(0,1,65)[None,:,None]).float().repeat(3,1,2) # identity tonemap curve

        bundle = dict(np.load(args.bundle_path, allow_pickle=True))

        if "reference_img" in bundle.keys():
            self.reference_img = torch.tensor(bundle["reference_img"]).float()
        if "translations" in bundle.keys():
            self.translations = torch.tensor(bundle["translations"]).float() # T,3
        
        # not efficient, load twice, but we're missing metadata for img sizes
        self.rgb_volume = torch.tensor(bundle["rgb_volume"]).float()[:,:3] # T,C,H,W
        
        self.intrinsics = torch.tensor(bundle["intrinsics"]).float() # T,3,3
        self.intrinsics = self.intrinsics.transpose(1, 2)
        self.intrinsics_inv = torch.inverse(self.intrinsics)
        self.rotations = torch.tensor(bundle["rotations"]).float() # T,3,3
        self.reference_rotation = self.rotations[0]
        self.camera_to_world = self.reference_rotation.T @ self.rotations
        
        self.num_frames = self.rgb_volume.shape[0]
        self.img_channels = self.rgb_volume.shape[1]
        self.img_height = self.rgb_volume.shape[2]
        self.img_width = self.rgb_volume.shape[3]

        if args.frames is not None:
            # subsample frames
            self.num_frames = len(args.frames)
            self.rotations = self.rotations[args.frames]
            self.camera_to_world = self.camera_to_world[args.frames]
            self.intrinsics = self.intrinsics[args.frames]
            self.intrinsics_inv = self.intrinsics_inv[args.frames]

        self.load_volume()

        self.frame_batch_size = 2 * (self.args.point_batch_size // self.num_frames // 2) # nearest even cut
        self.point_batch_size = self.frame_batch_size * self.num_frames # nearest multiple of num_frames
        self.num_batches = self.args.num_batches

        self.sin_epoch = 0.0 # fraction of training complete
        self.frame_cutoff = self.num_frames
        print("Frame Count: ", self.num_frames)


    def  init_RAW(self, args, load_volume=False):
        bundle = np.load(args.bundle_path, allow_pickle=True)
        
        self.characteristics = bundle['characteristics'].item() # camera characteristics
        self.motion = bundle['motion'].item()
        self.frame_timestamps = torch.tensor([bundle[f'raw_{i}'].item()['timestamp'] for i in range(bundle['num_raw_frames'])])
        self.motion_timestamps = torch.tensor(self.motion['timestamp'])

        self.quaternions = torch.tensor(self.motion['quaternion']).float() # T',4, has different timestamps from frames
        # our scene is +z towards scene convention, but phone is +z towards face convention
        # so we need to rotate 180 degrees around y axis, or equivalently flip over z,y
        self.quaternions[:,2] = -self.quaternions[:,2] # invert y
        self.quaternions[:,3] = -self.quaternions[:,3] # invert z

        self.quaternions = utils.multi_interp(self.frame_timestamps, self.motion_timestamps, self.quaternions)
        self.rotations = utils.convert_quaternions_to_rot(self.quaternions)

        self.reference_quaternion = self.quaternions[0]
        self.reference_rotation = self.rotations[0]
        
        self.camera_to_world = self.reference_rotation.T @ self.rotations
        
        self.intrinsics = torch.tensor(np.array([bundle[f'raw_{i}'].item()['intrinsics'] for i in range(bundle['num_raw_frames'])])).float()  # T,3,3  
        # swap cx,cy -> landscape to portrait
        cx, cy = self.intrinsics[:, 2, 1].clone(), self.intrinsics[:, 2, 0].clone()
        self.intrinsics[:, 2, 0], self.intrinsics[:, 2, 1] = cx, cy
        # transpose to put cx,cy in right column
        self.intrinsics = self.intrinsics.transpose(1, 2)
        self.intrinsics_inv = torch.inverse(self.intrinsics)

        self.lens_distortion = bundle['raw_0'].item()['lens_distortion']
        self.tonemap_curve = torch.tensor(bundle['raw_0'].item()['tonemap_curve'])
        self.ccm = utils.parse_ccm(bundle['raw_0'].item()['android']['colorCorrection.transform'])

        self.num_frames = bundle['num_raw_frames'].item()
        self.img_channels = 3
        self.img_height = bundle['raw_0'].item()['width'] # rotated 90
        self.img_width = bundle['raw_0'].item()['height']
        self.rgb_volume = torch.ones([self.num_frames, self.img_channels, 3,3]).float() # T,C,3,3, tiny fake volume for lazy loading

        if args.frames is not None:
            # subsample frames
            self.num_frames = len(args.frames)
            self.frame_timestamps = self.frame_timestamps[args.frames]
            self.quaternions = self.quaternions[args.frames]
            self.rotations = self.rotations[args.frames]
            self.camera_to_world = self.camera_to_world[args.frames]
            self.intrinsics = self.intrinsics[args.frames]
            self.intrinsics_inv = self.intrinsics_inv[args.frames]

        if load_volume:
            self.load_volume()

        self.frame_batch_size = 2 * (self.args.point_batch_size // self.num_frames // 2) # nearest even cut
        self.point_batch_size = self.frame_batch_size * self.num_frames # nearest multiple of num_frames
        self.num_batches = self.args.num_batches

        self.sin_epoch = 0.0 # fraction of training complete
        self.frame_cutoff = self.num_frames
        print("Frame Count: ", self.num_frames)
    
    def load_volume(self):
        if self.args.rgb_data: 
            bundle = dict(np.load(self.args.bundle_path, allow_pickle=True))
            self.rgb_volume = torch.tensor(bundle["rgb_volume"]).float()[:,:3]
        else: # need to unpack RAW data
            bundle = dict(np.load(self.args.bundle_path, allow_pickle=True))
            utils.de_item(bundle)

            self.rgb_volume = (utils.raw_to_rgb(bundle)) # T,C,H,W

            if self.args.max_percentile < 100: # cut off highlights for scaling (long-tail-distribution)
                self.rgb_volume = self.rgb_volume/np.percentile(self.rgb_volume, self.args.max_percentile)

            self.rgb_volume = self.rgb_volume.clamp(0,1)

        if self.args.frames is not None:
            self.rgb_volume = self.rgb_volume[self.args.frames]  # subsample frames


    def __len__(self):
        return self.num_batches  # arbitrary as we continuously generate random samples
        
    def __getitem__(self, idx):
        if self.args.frame_cutoff:
            self.frame_cutoff = min(int((0.1 + 2*self.sin_epoch) * self.num_frames), self.num_frames) # gradually increase frame cutoff
        else:
            self.frame_cutoff = self.num_frames

        uv = torch.rand((self.frame_batch_size * self.frame_cutoff), 2)*0.98 + 0.01 # uniform random in [0.01,0.99]
            
        # t is time for all frames, looks like [0, 0,... 0, 1/N, 1/N, ..., 1/N, 2/N, 2/N, ..., 2/N, etc.]
        t = (torch.linspace(0,1,self.num_frames)[:self.frame_cutoff]).repeat_interleave(self.frame_batch_size)[:,None] # point_batch_size, 1
        
        return self.generate_samples(t, uv)

    def generate_samples(self, t, uv):
        """ generate samples from dataset and camera parameters for training
        """
            
        # create frame_batch_size of quaterions for each frame
        camera_to_world = (self.camera_to_world[:self.frame_cutoff]).repeat_interleave(self.frame_batch_size, dim=0)
        # create frame_batch_size of intrinsics for each frame
        intrinsics = (self.intrinsics[:self.frame_cutoff]).repeat_interleave(self.frame_batch_size, dim=0)
        intrinsics_inv = (self.intrinsics_inv[:self.frame_cutoff]).repeat_interleave(self.frame_batch_size, dim=0)
        
        # sample grid
        grid_uv = ((uv - 0.5) * 2).reshape(self.frame_cutoff,self.frame_batch_size,1,2)
        rgb_samples = F.grid_sample(self.rgb_volume[:self.frame_cutoff], grid_uv, mode="bilinear", padding_mode="border", align_corners=True)
        # samples get returned in shape: num_frames x channels x frame_batch_size x 1 for some reason
        rgb_samples = rgb_samples.permute(0,2,1,3).squeeze().flatten(0,1) # point_batch_size x channels

        return t, uv, camera_to_world, intrinsics, intrinsics_inv, rgb_samples

    def sample_frame(self, uv, frame):
        """ sample frame [frame] at coordinates u,v
        """

        grid_uv = ((uv - 0.5) * 2)[None,:,None,:] # 1,point_batch_size,1,2
        rgb_samples = F.grid_sample(self.rgb_volume[frame:frame+1], grid_uv, mode="bilinear", padding_mode="border", align_corners=True)
        rgb_samples = rgb_samples.squeeze().permute(1,0) # point_batch_size, C
        
        return rgb_samples

#########################################################################################################
################################################ MODELS #################$###############################
#########################################################################################################
    
class RotationModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.stabilize = False
        self.delta_control_points = torch.nn.Parameter(data=torch.zeros(1, 3, self.args.camera_control_points, dtype=torch.float32), requires_grad=True)

    def forward(self, camera_to_world, t):
        delta_control_points = self.delta_control_points.repeat(t.shape[0],1,1)
        rotation_deltas = utils.interpolate(delta_control_points, t)
        rx, ry, rz = rotation_deltas[:,0], rotation_deltas[:,1], rotation_deltas[:,2]
        r0 = torch.zeros_like(rx)
        
        rotation_offsets = torch.stack([torch.stack([ r0, -rz,  ry], dim=-1),
                                        torch.stack([ rz,  r0, -rx], dim=-1),
                                        torch.stack([-ry,  rx,  r0], dim=-1)], dim=-1)
        
        return camera_to_world + self.args.rotation_weight * rotation_offsets
        
class TranslationModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.stabilize = False
        self.delta_control_points = torch.nn.Parameter(data=torch.zeros(1, 3, self.args.camera_control_points, dtype=torch.float32), requires_grad=True)

    def forward(self, t):
        control_points = self.args.translation_weight * self.delta_control_points.repeat(t.shape[0],1,1)
        translation = utils.interpolate(control_points, t)

        return translation

class PlaneModel(pl.LightningModule):
    """ Plane reprojection model with learnable z-depth
    """
    def __init__(self, args, depth):
        super().__init__()

        self.args = args
        self.depth = torch.nn.Parameter(data=torch.tensor([depth/1.0], dtype=torch.float32), requires_grad=True)

        self.u0 = torch.nn.Parameter(data=torch.tensor([1.0, 0.0], dtype=torch.float32), requires_grad=False)
        self.v0 = torch.nn.Parameter(data=torch.tensor([0.0, 1.0], dtype=torch.float32), requires_grad=False)

    def forward(self, ray_origins, ray_directions):
        # termination is just plane depth - ray origin z
        termination = ((1.0 * self.depth) - ray_origins[:, 2]).unsqueeze(1)

        # compute intersection points (N x 3)
        intersection_points = ray_origins + (termination * ray_directions)

        # project to (u, v) coordinates (N x 1 for each), avoid zero div
        u =  0.5 + 0.4 * torch.sum(intersection_points[:, :2] * (self.u0 / (torch.abs(termination) + 1e-6)), dim=1)
        v =  0.5 + 0.4 * torch.sum(intersection_points[:, :2] * (self.v0 / (torch.abs(termination) + 1e-6)), dim=1)
        uv = torch.stack((u, v), dim=1)

        return uv.clamp(0, 1)  # ensure UV coordinates stay within neural field bounds

class PlaneTransmissionModel(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        with open(f"config/config_{args.transmission_image_grid_size}.json") as config_image:
            config_image = json.load(config_image)
        with open(f"config/config_{args.transmission_flow_grid_size}.json" ) as config_flow:
            config_flow = json.load(config_flow)

        self.args = args

        self.encoding_image = tcnn.Encoding(n_input_dims=2, encoding_config=config_image["encoding"])
        self.encoding_flow = tcnn.Encoding(n_input_dims=2, encoding_config=config_flow["encoding"])

        self.network_image = tcnn.Network(n_input_dims=self.encoding_image.n_output_dims, n_output_dims=3, network_config=config_image["network"])
        self.network_flow = tcnn.Network(n_input_dims=self.encoding_flow.n_output_dims, 
                                         n_output_dims=2*(self.args.transmission_control_points_flow), network_config=config_flow["network"])

        self.model_plane = PlaneModel(args, args.transmission_initial_depth)
        self.initial_rgb = torch.nn.Parameter(data=torch.zeros([1,3], dtype=torch.float32), requires_grad=True)

    def forward(self, t, ray_origins, ray_directions, sin_epoch):
        uv_plane = self.model_plane(ray_origins, ray_directions)
        

        flow = self.network_flow(utils.mask(self.encoding_flow(uv_plane), sin_epoch)) # B x 2 
        
        flow = flow.reshape(-1,2,self.args.transmission_control_points_flow)
        flow = 0.01 * utils.interpolate(flow, t)

        rgb = self.network_image(utils.mask(self.encoding_image(uv_plane + flow), sin_epoch)).float()
        rgb = (self.initial_rgb + rgb).clamp(0,1)

        return rgb, flow
    
class PlaneObstructionModel(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        with open(f"config/config_{args.obstruction_image_grid_size}.json") as config_image:
            config_image = json.load(config_image)
        with open(f"config/config_{args.obstruction_alpha_grid_size}.json" ) as config_alpha:
            config_alpha = json.load(config_alpha)
        with open(f"config/config_{args.obstruction_flow_grid_size}.json" ) as config_flow:
            config_flow = json.load(config_flow)

        self.args = args

        self.encoding_image = tcnn.Encoding(n_input_dims=2, encoding_config=config_image["encoding"])
        self.encoding_alpha = tcnn.Encoding(n_input_dims=2, encoding_config=config_alpha["encoding"])
        self.encoding_flow = tcnn.Encoding(n_input_dims=2, encoding_config=config_flow["encoding"])

        self.network_image = tcnn.Network(n_input_dims=self.encoding_image.n_output_dims, n_output_dims=3, network_config=config_image["network"])
        self.network_alpha = tcnn.Network(n_input_dims=self.encoding_alpha.n_output_dims, n_output_dims=1, network_config=config_alpha["network"])
        self.network_flow = tcnn.Network(n_input_dims=self.encoding_flow.n_output_dims, n_output_dims=2*(self.args.obstruction_control_points_flow), network_config=config_flow["network"])

        self.model_plane = PlaneModel(args, args.obstruction_initial_depth)
        self.initial_alpha = torch.nn.Parameter(data=torch.tensor(args.obstruction_initial_alpha, dtype=torch.float32), requires_grad=True) 
        self.initial_rgb = torch.nn.Parameter(data=torch.zeros([1,3], dtype=torch.float32), requires_grad=True)

    def forward(self, t, ray_origins, ray_directions, sin_epoch):
        uv_plane = self.model_plane(ray_origins, ray_directions)

        flow = self.network_flow(utils.mask(self.encoding_flow(uv_plane), sin_epoch)) # B x 2
        flow = flow.reshape(-1,2,self.args.obstruction_control_points_flow)
        flow = 0.01 * utils.interpolate(flow, t)

        rgb = self.network_image(utils.mask(self.encoding_image(uv_plane + flow), sin_epoch)).float()
        rgb = (self.initial_rgb + rgb).clamp(0,1)

        alpha = self.network_alpha(utils.mask(self.encoding_alpha(uv_plane + flow), sin_epoch)).float()
        alpha = torch.sigmoid((-torch.log(1/self.initial_alpha - 1) + self.args.alpha_temperature * alpha))

        return rgb, flow, alpha
        
#########################################################################################################
################################################ NETWORK ################################################
#########################################################################################################
    
class BundleMLP(pl.LightningModule):
    def __init__(self, args, cached_bundle=None):
        super().__init__()
        # load network configs

        self.args = args
        if cached_bundle is None:
             self.bundle = BundleDataset(self.args)
        else:
            with open(cached_bundle, 'rb') as file:
                self.bundle = pickle.load(file)

        self.img_width = self.bundle.img_width
        self.img_height = self.bundle.img_height
        self.lens_distortion = self.bundle.lens_distortion
        self.num_frames = args.num_frames = self.bundle.num_frames
        if args.frames is None:
            self.args.frames = list(range(self.num_frames))

        self.model_transmission = PlaneTransmissionModel(args)
        self.model_obstruction = PlaneObstructionModel(args)
        self.model_translation = TranslationModel(args)
        self.model_rotation = RotationModel(args)

        self.sin_epoch = 1.0
        self.save_hyperparameters()

    def load_volume(self):
        self.bundle.load_volume()
        
    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        #constant lr
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.0)
        
        return [optimizer], [scheduler]

        
    def forward(self, t, ray_origins, ray_directions):
        """ Forward model pass, estimate motion, implicit depth + image.
        """

        rgb_transmission, flow_transmission = self.model_transmission(t, ray_origins, ray_directions, self.sin_epoch)
        rgb_obstruction, flow_obstruction, alpha_obstruction = self.model_obstruction(t, ray_origins, ray_directions, self.sin_epoch)

        if self.args.single_plane:
            rgb_combined = rgb_transmission
        else:
            rgb_combined = rgb_transmission * (1 - alpha_obstruction) + rgb_obstruction * alpha_obstruction
        
        return rgb_combined, rgb_transmission, rgb_obstruction, flow_transmission, flow_obstruction, alpha_obstruction

    def generate_ray_directions(self, uv, camera_to_world, intrinsics_inv):
        u, v = uv[:,0:1] * self.img_width, uv[:,1:2] * self.img_height
        uv1 = torch.cat([u, v, torch.ones_like(uv[:,0:1])], dim=1) # N x 3
        # scale by image width/height
        xy1 = torch.bmm(intrinsics_inv, uv1.unsqueeze(2)).squeeze(2) # N x 3 
        xy = xy1[:,0:2]

        f_div_cx = -1 / intrinsics_inv[:,0,2]
        f_div_cy = -1 / intrinsics_inv[:,1,2]

        r2 = torch.sum(xy**2, dim=1, keepdim=True) # N x 1
        r4 = r2**2
        r6 = r2**3
        kappa1, kappa2, kappa3 = self.lens_distortion[0:3]

        # apply lens distortion correction
        xy = xy * (1 + kappa1*r2 + kappa2*r4 + kappa3*r6)
        
        xy = xy * torch.min(f_div_cx[:, None], f_div_cy[:, None]) # scale long dimension to -1, 1
        ray_directions = torch.cat([xy, torch.ones_like(xy[:,0:1])], dim=1) # N x 3
        ray_directions = torch.bmm(camera_to_world, ray_directions.unsqueeze(2)).squeeze(2) # apply camera rotation
        ray_directions = ray_directions / ray_directions[:,2:3] # normalize by z    
        return ray_directions
    
    def training_step(self, train_batch, batch_idx):
        t, uv, camera_to_world, intrinsics, intrinsics_inv, rgb_reference = debatch(train_batch) # collapse batch + point dimensions
        
        camera_to_world = self.model_rotation(camera_to_world, t) # apply rotation offset
        ray_origins = self.model_translation(t) # camera center in world coordinates
        ray_directions = self.generate_ray_directions(uv, camera_to_world, intrinsics_inv)

        rgb_combined, rgb_transmission, rgb_obstruction, flow_transmission, flow_obstruction, alpha_obstruction = self.forward(t, ray_origins, ray_directions)   
       
        loss = 0.0

        rgb_loss = torch.abs((rgb_combined - rgb_reference)/(rgb_combined.detach() + 0.001))
        self.log('loss/rgb', rgb_loss.mean())
        loss += rgb_loss.mean()

        self.log(f'plane_depth/image', self.model_transmission.model_plane.depth)
        self.log(f'plane_depth/obstruction', self.model_obstruction.model_plane.depth)

        if (np.abs(self.args.alpha_weight) > 0 and self.sin_epoch) > 0.6:
            alpha_loss = self.args.alpha_weight * self.sin_epoch * alpha_obstruction
            self.log('loss/alpha', alpha_loss.mean())
            loss += alpha_loss.mean()

        return loss

    def color_and_tone(self, rgb_samples, height, width):
        """ Apply CCM and tone curve to raw samples
        """

        img = self.bundle.ccm.to(rgb_samples.device) @ rgb_samples.T
        img = img.reshape(3, height, width)
        img = utils.apply_tonemap_curve(img, self.bundle.tonemap_curve)
            
        return img
        
    def make_grid(self, height, width, u_lims, v_lims):
        """ Create (u,v) meshgrid with size (height,width) extent (u_lims, v_lims)
        """
        u = torch.linspace(u_lims[0], u_lims[1], width)
        v = torch.linspace(v_lims[0], v_lims[1], height)
        u_grid, v_grid = torch.meshgrid([u, v], indexing="xy") # u/v grid
        return torch.stack((u_grid.flatten(), v_grid.flatten())).t()
        
    def generate_img(self, frame, height=960, width=720, u_lims=[0,1], v_lims=[0,1]):
        """ Produce reference image for tensorboard/visualization
        """
        device = self.device
        uv = self.make_grid(height, width, u_lims, v_lims)
    
        rgb_samples = self.bundle.sample_frame(uv, frame).to(device)
        img = self.color_and_tone(rgb_samples, height, width)
            
        return img
        
    def generate_outputs(self, frame=0, height=720, width=540, u_lims=[0,1], v_lims=[0,1], time=None):
        """ Use forward model to sample implicit image I(u,v), depth D(u,v) and raw images
            at reprojected u,v, coordinates. Results should be aligned (sampled at (u',v'))
        """
        device = self.device
        uv = self.make_grid(height, width, u_lims, v_lims)
        if time is None:
            t = torch.tensor(frame/(self.bundle.num_frames - 1), dtype=torch.float32).repeat(uv.shape[0])[:,None] # num_points x 1
        else:
            t = torch.tensor(time, dtype=torch.float32).repeat(uv.shape[0])[:,None] # num_points x 1
            frame = int(np.floor(time * (self.bundle.num_frames - 1)))
    
        rgb_reference = self.bundle.sample_frame(uv, frame).to(device)
        intrinsics_inv = self.bundle.intrinsics_inv[frame:frame+2] # 2 x 3 x 3
        camera_to_world = self.bundle.camera_to_world[frame:frame+2] # 2 x 3 x 3

        if time is None or time >= 1.0: # select exact frame timestamp
            intrinsics_inv = intrinsics_inv[0:1]
            camera_to_world = camera_to_world[0:1]
        else: # interpolate between frames
            fraction = time * (self.bundle.num_frames - 1) - frame
            intrinsics_inv = intrinsics_inv[0:1] * (1 - fraction) + intrinsics_inv[1:2] * fraction
            camera_to_world = camera_to_world[0:1] * (1 - fraction) + camera_to_world[1:2] * fraction

        intrinsics_inv = intrinsics_inv.repeat(uv.shape[0],1,1) # num_points x 3 x 3
        camera_to_world = camera_to_world.repeat(uv.shape[0],1,1) # num_points x 3 x 3
        
        with torch.no_grad():
            rgb_combined_chunks = []
            rgb_transmission_chunks = []
            rgb_obstruction_chunks = []
            flow_transmission_chunks = []
            flow_obstruction_chunks = []
            alpha_obstruction_chunks = []
            
            chunk_size = 42 * self.args.point_batch_size
            for i in range((t.shape[0] // chunk_size) + 1):
                t_chunk, uv_chunk = t[i*chunk_size:(i+1)*chunk_size].to(device), uv[i*chunk_size:(i+1)*chunk_size].to(device)
                intrinsics_inv_chunk = intrinsics_inv[i*chunk_size:(i+1)*chunk_size].to(device)
                camera_to_world_chunk = camera_to_world[i*chunk_size:(i+1)*chunk_size].to(device)

                camera_to_world_chunk = self.model_rotation(camera_to_world_chunk, t_chunk) # apply rotation offset
                ray_origins = self.model_translation(t_chunk) # camera center in world coordinates
                ray_directions = self.generate_ray_directions(uv_chunk, camera_to_world_chunk, intrinsics_inv_chunk)
                rgb_combined, rgb_transmission, rgb_obstruction, flow_transmission, flow_obstruction, alpha_obstruction = self.forward(t_chunk, ray_origins, ray_directions)

                rgb_combined_chunks.append(rgb_combined.detach().cpu())
                rgb_transmission_chunks.append(rgb_transmission.detach().cpu())
                rgb_obstruction_chunks.append(rgb_obstruction.detach().cpu())
                flow_transmission_chunks.append(flow_transmission.detach().cpu())
                flow_obstruction_chunks.append(flow_obstruction.detach().cpu())
                alpha_obstruction_chunks.append(alpha_obstruction.detach().cpu())

            rgb_combined = torch.cat(rgb_combined_chunks, dim=0)

            rgb_reference = self.color_and_tone(rgb_reference, height, width)
            rgb_combined = self.color_and_tone(rgb_combined, height, width)
            rgb_transmission = self.color_and_tone(torch.cat(rgb_transmission_chunks, dim=0), height, width)
            rgb_obstruction = self.color_and_tone(torch.cat(rgb_obstruction_chunks, dim=0), height, width)
            flow_transmission = torch.cat(flow_transmission_chunks, dim=0).reshape(height, width, 2)
            flow_obstruction = torch.cat(flow_obstruction_chunks, dim=0).reshape(height, width, 2)
            alpha_obstruction = torch.cat(alpha_obstruction_chunks, dim=0).reshape(height, width)
            
        return rgb_reference, rgb_combined, rgb_transmission, rgb_obstruction, flow_transmission, flow_obstruction, alpha_obstruction
    
    def save_outputs(self, path, high_res=False):
        os.makedirs(f"outputs/{self.args.name + path}", exist_ok=True)
        if high_res:
            rgb_reference, rgb_combined, rgb_transmission, rgb_obstruction, flow_transmission, flow_obstruction, alpha_occlusion = model.generate_outputs(frame=0, height=2560, width=1920, u_lims=[0,1], v_lims=[0,1], time=0.0)
            np.save(f"outputs/{self.args.name + path}/flow_transmission.npy", flow_transmission.detach().cpu().numpy())
            np.save(f"outputs/{self.args.name + path}/flow_obstruction.npy", flow_obstruction.detach().cpu().numpy())
        else:
            rgb_reference, rgb_combined, rgb_transmission, rgb_obstruction, flow_transmission, flow_obstruction, alpha_occlusion = model.generate_outputs(frame=0, height=1080, width=810, u_lims=[0,1], v_lims=[0,1], time=0.0)

        plt.imsave(f"outputs/{self.args.name + path}/reference.png", rgb_reference.permute(1,2,0).detach().cpu().numpy())
        plt.imsave(f"outputs/{self.args.name + path}/alpha.png", alpha_occlusion.detach().cpu().numpy(), cmap="gray")
        plt.imsave(f"outputs/{self.args.name + path}/transmission.png", rgb_transmission.permute(1,2,0).detach().cpu().numpy())
        plt.imsave(f"outputs/{self.args.name + path}/obstruction.png", rgb_obstruction.permute(1,2,0).detach().cpu().numpy())
        plt.imsave(f"outputs/{self.args.name + path}/combined.png", rgb_combined.permute(1,2,0).detach().cpu().numpy())

    
#########################################################################################################
############################################### VALIDATION ##############################################
#########################################################################################################
        
class ValidationCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        
    def on_train_epoch_start(self, trainer, model):
        model.sin_epoch =  min(1.0, 0.05 + np.sin(model.current_epoch/(model.args.max_epochs - 1) * np.pi/2)) # progression of training
        trainer.train_dataloader.dataset.sin_epoch = model.sin_epoch
        print(f" Sin of Current Epoch: {model.sin_epoch:.3f}")

        if model.sin_epoch > 0.4:
            # unlock flow model
            model.model_transmission.encoding_flow.requires_grad_(True)
            model.model_transmission.network_flow.requires_grad_(True)
            model.model_obstruction.encoding_flow.requires_grad_(True)
            model.model_obstruction.network_flow.requires_grad_(True)

            model.model_transmission.network_flow.train(True)
            model.model_transmission.encoding_flow.train(True)
            model.model_obstruction.encoding_flow.train(True)
            model.model_obstruction.network_flow.train(True)
                 
        if model.args.fast: # skip tensorboarding except for beginning and end
            if model.current_epoch == model.args.max_epochs - 1 or model.current_epoch == 0:
                pass
            else:
                return
            
        # for i, frame in enumerate([0, model.bundle.num_frames//2, model.bundle.num_frames-1]): # can sample more frames
        for i, frame in enumerate([0]):
            rgb_reference, rgb_combined, rgb_transmission, rgb_obstruction, flow_transmission, flow_obstruction, alpha_obstruction = model.generate_outputs(frame)
            model.logger.experiment.add_image(f'pred/{i}_rgb_combined', rgb_combined, global_step=trainer.global_step)
            model.logger.experiment.add_image(f'pred/{i}_rgb_transmission', rgb_transmission, global_step=trainer.global_step)
            model.logger.experiment.add_image(f'pred/{i}_rgb_obstruction', rgb_obstruction, global_step=trainer.global_step)
            model.logger.experiment.add_image(f'pred/{i}_rgb_obstruction_alpha', rgb_obstruction * alpha_obstruction, global_step=trainer.global_step)
            model.logger.experiment.add_image(f'pred/{i}_alpha_obstruction', utils.colorize_tensor(alpha_obstruction, vmin=0, vmax=1, cmap="gray"), global_step=trainer.global_step)

            
            if model.args.save_video: # save the evolution of the model
                model.save_outputs(path=f"/{model.current_epoch}")
            
    def on_train_start(self, trainer, model):
        pl.seed_everything(42) # the answer to life, the universe, and everything

        # initialize rgb as average color of first frame of data (minimize the amount the rgb models have to learn)
        model.model_transmission.initial_rgb.data = torch.mean(model.bundle.rgb_volume[0], dim=(1,2))[None,:].to(model.device)
        model.model_obstruction.initial_rgb.data = torch.mean(model.bundle.rgb_volume[0], dim=(1,2))[None,:].to(model.device)
        
        model.logger.experiment.add_text("args", str(model.args))

        for i, frame in enumerate([0, model.bundle.num_frames//2, model.bundle.num_frames-1]): 
            rgb_raw = model.generate_img(frame)
            model.logger.experiment.add_image(f'gt/{i}_rgb_raw', rgb_raw, global_step=trainer.global_step)

            
    def on_train_end(self, trainer, model):
        checkpoint_dir = os.path.join("checkpoints", model.args.name, "last.ckpt")
        bundle_dir = os.path.join("checkpoints", model.args.name, "bundle.pkl")
        trainer.save_checkpoint(checkpoint_dir)
            
        model.save_outputs(path=f"-final", high_res=True)
        
        with open(bundle_dir, 'wb') as file:
            model.bundle.rgb_volume = torch.ones([model.bundle.num_frames, model.bundle.img_channels, 3,3]).float()
            pickle.dump(model.bundle, file)

if __name__ == "__main__":
    
    # argparse
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--point_batch_size', type=int, default=2**18, help="Number of points to sample per dataloader index.")
    parser.add_argument('--num_batches', type=int, default=80, help="Number of training batches.")
    parser.add_argument('--max_percentile', type=float, default=100, help="Percentile of brightest pixels to cut.")
    parser.add_argument('--frames', type=str, help="Which subset of frames to use for training, e.g. 0,10,20,30,40")
    parser.add_argument('--rgb_data', action='store_true', help="Input data is pre-processed RGB.")
    
    # model
    parser.add_argument('--camera_control_points', type=int, default=22, help="Spline control points for translation/rotation model.")
    parser.add_argument('--alpha_weight', type=float, default=1e-2, help="Alpha regularization weight.")
    parser.add_argument('--rotation_weight', type=float, default=1e-3, help="Scale learned rotation.")
    parser.add_argument('--translation_weight', type=float, default=1e-2, help="Scale learned translation.")
    parser.add_argument('--alpha_temperature', type=float, default=1.0, help="Temperature for sigmoid in alpha matte calculation.")

    # planes
    parser.add_argument('--obstruction_control_points_flow', type=int, default=11, help="Spline control points for flow models.")
    parser.add_argument('--obstruction_flow_grid_size', type=str, default="tiny", help="Obstruction flow grid size (small, medium, large).")
    parser.add_argument('--obstruction_image_grid_size', type=str, default="large", help="Obstruction image grid size (small, medium, large).")
    parser.add_argument('--obstruction_alpha_grid_size', type=str, default="large", help="Obstruction alpha grid size (small, medium, large).")
    parser.add_argument('--obstruction_initial_depth', type=float, default=1.0, help="Obstruction initial plane depth.")
    parser.add_argument('--obstruction_initial_alpha', type=float, default=0.5, help="Obstruction initial alpha.")
    parser.add_argument('--transmission_control_points_flow', type=int, default=11, help="Spline control points for flow models.")
    parser.add_argument('--transmission_flow_grid_size', type=str, default="tiny", help="Transmission flow grid size (small, medium, large).")
    parser.add_argument('--transmission_image_grid_size', type=str, default="large", help="Transmission image grid size (small, medium, large).")
    parser.add_argument('--transmission_initial_depth', type=float, default=0.4, help="Transmission initial plane depth.")
    parser.add_argument('--single_plane', action='store_true', help="Use single plane model.")
    
    # training
    parser.add_argument('--bundle_path', type=str, required=True, help="Path to frame_bundle.npz")
    parser.add_argument('--name', type=str, required=True, help="Experiment name for logs and checkpoints.")
    parser.add_argument('--max_epochs', type=int, default=75, help="Number of training epochs.")
    parser.add_argument('--lr', type=float, default=3e-5, help="Learning rate.")
    parser.add_argument('--save_video', action='store_true', help="Store training outputs at each epoch for visualization.")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument('--debug', action='store_true', help="Debug mode, only use 1 batch.")
    parser.add_argument('--frame_cutoff', action='store_true', help="Use frame cutoff.")
    parser.add_argument('--fast', action='store_true', help="Fast mode.")


    args = parser.parse_args()
    # parse plane args
    print(args)
    if args.frames is not None: 
        args.frames = [int(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", args.frames)]

    # model
    model = BundleMLP(args)
    model.load_volume()

    # freeze flow at the start of training as it will otherwise fight the camera model during early image fitting
    # these can be omitted at the cost of learning really weird camera translations
    model.model_transmission.encoding_flow.requires_grad_(False)
    model.model_transmission.encoding_flow.train(False)
    model.model_transmission.network_flow.requires_grad_(False)
    model.model_transmission.network_flow.train(False)
    model.model_obstruction.encoding_flow.requires_grad_(False)
    model.model_obstruction.encoding_flow.train(False)
    model.model_obstruction.network_flow.requires_grad_(False)
    model.model_obstruction.network_flow.train(False)

    # dataset
    bundle = model.bundle
    train_loader = DataLoader(bundle, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=True, prefetch_factor=1)


    torch.set_float32_matmul_precision('high')

    # training
    lr_callback = pl.callbacks.LearningRateMonitor()
    logger = pl.loggers.TensorBoardLogger(save_dir=os.getcwd(), version=args.name, name="lightning_logs")
    validation_callback = ValidationCallback()
    trainer = pl.Trainer(accelerator="gpu", devices=torch.cuda.device_count(), num_nodes=1, strategy="auto", max_epochs=args.max_epochs,
                         logger=logger, callbacks=[validation_callback, lr_callback], enable_checkpointing=False, fast_dev_run=args.debug)
    trainer.fit(model, train_loader)
