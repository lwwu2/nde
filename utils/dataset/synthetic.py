import torch
import torch.nn.functional as NF
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T
import math
from tqdm import tqdm


def get_ray_directions(H, W, focal):
    """ get camera ray directions in camera space """
    x_coords = torch.linspace(0.5, W-0.5, W)
    y_coords = torch.linspace(0.5, H-0.5, H)
    j, i = torch.meshgrid([y_coords, x_coords])
    directions = \
        torch.stack([(i-W/2)/focal, -(j-H/2)/focal, -torch.ones_like(i)], -1)
    return directions


def get_rays(directions, c2w):
    """ get rays in world space"""
    rays_d = directions @ c2w[:, :3].T
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    rays_o = c2w[:, 3].expand(rays_d.shape)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)
    return rays_o, rays_d


""" dataloader for nerf and refnerf synthetic dataset that applies dynamic batch size strategy """
class SyntheticDataset(Dataset):
    def __init__(self, root_dir, split='train', pixel=True,img_hw=(800, 800),batch_size=None,device='cpu'):
        self.root_dir = root_dir
        self.split = split
        self.img_hw = img_hw
        
        
        self.batch_size = batch_size # initial batch size
        self.pixel = pixel # load by pixels or images
        self.device= torch.device(device)
        
        
        self.transform = T.ToTensor()
        self.white_back = True # assume white background
        
        pose_path = os.path.join(self.root_dir,
                               f"transforms_{self.split}.json")
        if not os.path.exists(pose_path):
            pose_path = os.path.join(self.root_dir,
                               f"transforms_test.json")
        with open(pose_path, 'r') as f:
            self.meta = json.load(f)

        h, w = self.img_hw
        self.focal = 0.5*800/np.tan(0.5*self.meta['camera_angle_x'])
        self.focal *= self.img_hw[1]/800

        # near and far plane for synthetic nerf dataset
        self.near = 2.0
        self.far = 6.0
        self.bounds = np.array([self.near, self.far])
        
        
        self.directions = get_ray_directions(h, w, self.focal) # (h, w, 3)
        
        if self.split == 'train' and self.pixel: # create cache of all rays and rgb data
            self.image_paths = []
            self.poses = []
            self.all_rays = []
            self.all_rgbs = []
            for i in tqdm(range(len(self.meta['frames']))):
                frame = self.meta['frames'][i]
                pose = np.array(frame['transform_matrix'])[:3, :4]
                self.poses += [pose]
                c2w = torch.FloatTensor(pose)

                image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
                self.image_paths += [image_path]
                img = Image.open(image_path)
                img = img.resize(self.img_hw[::-1], Image.LANCZOS)
                img = self.transform(img).float() # (4, h, w)
                if img.shape[0] == 4: # rgba image
                    img = img.view(4, -1).permute(1, 0) # (h*w, 4) RGBA
                    img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB
                else: # rgb image
                    img = img.view(3,-1).T
                self.all_rgbs += [img]
                
                rays_o, rays_d = get_rays(self.directions, c2w) # both (h*w, 3)

                self.all_rays += [torch.cat([rays_o, rays_d],1).float()] # (h*w, 8)

            self.all_rays = torch.cat(self.all_rays, 0).to(self.device)
            self.all_rgbs = torch.cat(self.all_rgbs, 0).to(self.device)

    def update_num_rays(self,batch_size):
        self.batch_size = batch_size

    def __len__(self):
        if self.split == 'train' and self.pixel:
            return len(self.poses)
        if self.split == 'val':
            return 8 # only validate 8 images
        return len(self.meta['frames'])

    def __getitem__(self, idx):
        if self.split == 'train' and self.pixel: # load random pixels
            image_id = torch.randint(0,len(self.poses),size=(self.batch_size,),
                                    device=self.device)
            x = torch.randint(0,self.img_hw[1],size=(self.batch_size,),
                             device=self.device)
            y = torch.randint(0,self.img_hw[0],size=(self.batch_size,),
                             device=self.device)
            idx = y*self.img_hw[1]+x+image_id*self.img_hw[0]*self.img_hw[1]
            
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]}
            
        else: # load single image
            frame = self.meta['frames'][idx]
            c2w = torch.FloatTensor(frame['transform_matrix'])[:3, :4]

            img = Image.open(os.path.join(self.root_dir, f"{frame['file_path']}.png"))
            img = img.resize(self.img_hw[::-1], Image.LANCZOS)
            img = self.transform(img) # (4, H, W)
            valid_mask = (img[-1]>0).flatten() # (H*W) valid color area
            
            if img.shape[0] == 4: # rgba image
                img = img.view(4, -1).permute(1, 0) # (h*w, 4) RGBA
                img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB
            else: # rgb image
                img = img.view(3,-1).T
            rays_o, rays_d = get_rays(self.directions, c2w)

            rays = torch.cat([rays_o, rays_d, 
                              self.near*torch.ones_like(rays_o[:, :1]),
                              self.far*torch.ones_like(rays_o[:, :1])],
                              1) # (H*W, 8)

            sample = {'rays': rays,
                      'rgbs': img,
                      'c2w': c2w,
                      'valid_mask': valid_mask}

        return sample