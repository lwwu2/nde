import torch
import torch.nn.functional as NF
from torch.utils.data import Dataset

from pathlib import Path
from skimage.io import imread
import os
import pickle
import cv2
import numpy as np
from tqdm import tqdm

from .plyfile import PlyData

""" dataloader of nero real dataset using dynamic batch size strategy
    modified from: https://github.com/liuyuan-pal/NeRO/blob/main/dataset/database.py 
"""


def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)
    
def resize_img(img, ratio):
    # if ratio>=1.0: return img
    h, w, _ = img.shape
    hn, wn = int(np.round(h * ratio)), int(np.round(w * ratio))
    img_out = cv2.resize(downsample_gaussian_blur(img, ratio), (wn, hn), cv2.INTER_LINEAR)
    return img_out



class GlossyRealDataset(Dataset):
    meta_info={
        'bear': {'forward': np.asarray([0.539944,-0.342791,0.341446],np.float32), 'up': np.asarray((0.0512875,-0.645326,-0.762183),np.float32),},
        'coral': {'forward': np.asarray([0.004226,-0.235523,0.267582],np.float32), 'up': np.asarray((0.0477973,-0.748313,-0.661622),np.float32),},
        'maneki': {'forward': np.asarray([-2.336584, -0.406351, 0.482029], np.float32), 'up': np.asarray((-0.0117387, -0.738751, -0.673876), np.float32), },
        'bunny': {'forward': np.asarray([0.437076,-1.672467,1.436961],np.float32), 'up': np.asarray((-0.0693234,-0.644819,-.761185),np.float32),},
        'vase': {'forward': np.asarray([-0.911907, -0.132777, 0.180063], np.float32), 'up': np.asarray((-0.01911, -0.738918, -0.673524), np.float32), },
    }
    def __init__(self, root,split='train',bkgd=None,pixel=True,batch_size=None,device='cpu'):
        self.object_name= root.split('/')[-1]

        self.split = split
        self.batch_size = batch_size
        self.pixel = pixel # whether load pixels or images
        self.root = root
        self._parse_colmap()
        self._normalize()
        self.device = torch.device(device)

        h, w, _ = imread(f'{self.root}/images/{self.image_names[self.img_ids[0]]}').shape
        max_len = 1024
        self.max_len = max_len
        ratio = float(max_len) / max(h, w)
        th, tw = int(ratio*h), int(ratio*w)
        rh, rw = th / h, tw / w
        self.ratio = ratio
        self.img_hw = (th,tw)
        
        for img_id in self.img_ids:
            K = self.Ks[img_id]
            self.Ks[img_id] = torch.from_numpy(np.diag([rw,rh,1.0]) @ K).float()
            self.poses[img_id] = torch.from_numpy(self.poses[img_id]).float()
            
        coords = torch.stack(torch.meshgrid(torch.arange(self.img_hw[0]),torch.arange(self.img_hw[1])),-1)[:,:,[1,0]]
        coords = torch.cat([coords+0.5,torch.ones_like(coords[...,0:1])],-1)
        self.dirs = coords.reshape(-1,3)@torch.inverse(self.Ks[1]).T
        
        
        # split train test set
        img_num = len(self.img_ids)
        split_step = int(img_num*0.1)

        test_idxs = torch.arange(split_step,img_num,split_step)
        train_idxs =torch.tensor([i for i in range(img_num) if i not in test_idxs]).long()
        if self.split == 'train':
            self.split_idxs = train_idxs.numpy()
        elif self.split == 'exp':
            self.split_idxs = np.arange(img_num)
        else:
            self.split_idxs = test_idxs.numpy()
        
        
        
        
        all_rays = []
        all_rgbs = []
        tmins = []
        tmaxs = []
        human_poses = []
        if self.split == 'train' and self.pixel:
            for split_idx in tqdm(range(len(self.split_idxs))):
                i = self.split_idxs[split_idx]
                img_id = self.img_ids[i]
                img_path = Path(f'{self.root}/images_raw_{self.max_len}/{self.image_names[img_id]}') 
                if not img_path.exists():
                    img = imread(f'{self.root}/images/{self.image_names[img_id]}')
                    img = resize_img(img, self.ratio)
                else:
                    img = imread(img_path)

                img = torch.from_numpy(img.astype(np.float32)).to(self.device)/255.0
                
                rays,near_,far_,human_pose = self._get_rays(self.poses[img_id])
                tmins.append(near_.reshape(-1).to(self.device))
                tmaxs.append(far_.reshape(-1).to(self.device))
                all_rays.append(rays)
                all_rgbs.append(img.reshape(-1,3))
                human_poses.append(human_pose)
            self.all_rays = torch.cat(all_rays,0).to(self.device)
            self.all_rgbs = torch.cat(all_rgbs,0).to(self.device)
            self.tmins = torch.cat(tmins,0)
            self.tmaxs = torch.cat(tmaxs,0)
            self.human_poses = torch.stack(human_poses,0).to(self.device)
    
    def update_num_rays(self,batch_size):
        self.batch_size = batch_size
    
    def __len__(self,):
        if self.split == 'train' and self.pixel:
            return 100
        elif self.split == 'val':
            return min(8,len(self.split_idxs))
        return len(self.split_idxs)
        
    def __getitem__(self,idx):
        if self.split == 'train' and self.pixel:
            image_id = torch.randint(0,len(self.split_idxs),size=(self.batch_size,),
                                    device=self.device)
            x = torch.randint(0,self.img_hw[1],size=(self.batch_size,),
                             device=self.device)
            y = torch.randint(0,self.img_hw[0],size=(self.batch_size,),
                             device=self.device)
            idx = y*self.img_hw[1]+x+image_id*self.img_hw[0]*self.img_hw[1]
            
            return {
                'rays': self.all_rays[idx],
                'rgbs': self.all_rgbs[idx],
                'tmins': self.tmins[idx],
                'tmaxs': self.tmaxs[idx],
                'human_pose': self.human_poses[image_id]
            }
        idx = self.split_idxs[idx]
        img_path = Path(f'{self.root}/images_raw_{self.max_len}/{self.image_names[idx+1]}') 
        if not img_path.exists():
            img = imread(f'{self.root}/images/{self.image_names[idx+1]}')
            img = resize_img(img, self.ratio)
        else:
            img = imread(img_path)
            
        img = torch.from_numpy(img.astype(np.float32))/255.0
        img = img.reshape(-1,3)
        
        rays,tmins,tmaxs,human_pose = self._get_rays(self.poses[idx+1])
        
        human_pose = torch.ones(len(img),3,4)*human_pose[None]
        
        return {
            'rgbs': img,
            'rays': rays,
            'tmins': tmins.reshape(-1),
            'tmaxs': tmaxs.reshape(-1),
            'human_pose': human_pose}
    
    
    def _get_rays(self, pose):
        rays_d = self.dirs
        
        rays_o = (pose[:,:3].T@(-pose[:,3:])).T*torch.ones_like(rays_d)
        rays_d = rays_d@pose[:,:3]
        rays_d = NF.normalize(rays_d,dim=-1)
        
        near,far = self.near_far_from_sphere(rays_o,rays_d)
        human_pose = self.get_human_coordinate_pose(pose)
        return torch.cat([rays_o,rays_d],-1),near,far,human_pose
    
    @staticmethod
    def near_far_from_sphere(rays_o, rays_d):
        a = torch.sum(rays_d ** 2, dim=-1)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        near = torch.clamp(near, min=1e-3)
        return near, far
    
    def get_human_coordinate_pose(self,pose):
        cam_cen = (-pose[:,:3].T@pose[:,3:])[...,0]
        cam_cen[...,2] = 0
        
        Y = torch.zeros(3)
        Y[2] = -1.0
        Z = pose[2,:3].clone()
        Z[2] = 0
        Z = NF.normalize(Z,dim=-1)
        X = torch.cross(Y,Z)
        R = torch.stack([X,Y,Z],0)
        t = -R@cam_cen[:,None]
        return torch.cat([R,t],-1)
    
    
    def _parse_colmap(self):
        self.poses, self.Ks, self.image_names, self.img_ids = read_pickle(f'{self.root}/cache.pkl')
    
    def _load_point_cloud(self, pcl_path):
        with open(pcl_path, "rb") as f:
            plydata = PlyData.read(f)
            xyz = np.stack([np.array(plydata["vertex"][c]).astype(float) for c in ("x", "y", "z")], axis=1)
        return xyz

    def _compute_rotation(self, vert, forward):
        y = np.cross(vert, forward)
        x = np.cross(y, vert)

        vert = vert/np.linalg.norm(vert)
        x = x/np.linalg.norm(x)
        y = y/np.linalg.norm(y)
        R = np.stack([x, y, vert], 0)
        return R

    def _normalize(self):
        ref_points = self._load_point_cloud(f'{self.root}/object_point_cloud.ply')
        max_pt, min_pt = np.max(ref_points, 0), np.min(ref_points, 0)
        center = (max_pt + min_pt) * 0.5
        offset = -center # x1 = x0 + offset
        scale = 1 / np.max(np.linalg.norm(ref_points - center[None,:], 2, 1)) # x2 = scale * x1
        up, forward = self.meta_info[self.object_name]['up'], self.meta_info[self.object_name]['forward']
        up, forward = up/np.linalg.norm(up), forward/np.linalg.norm(forward)
        R_rec = self._compute_rotation(up, forward) # x3 = R_rec @ x2
        self.ref_points = scale * (ref_points + offset) @ R_rec.T
        self.scale_rect = scale
        self.offset_rect = offset
        self.R_rect = R_rec

        for img_id, pose in self.poses.items():
            R, t = pose[:,:3], pose[:,3]
            R_new = R @ R_rec.T
            t_new = (t - R @ offset) * scale
            self.poses[img_id] = np.concatenate([R_new, t_new[:,None]], -1)