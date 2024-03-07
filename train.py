import torch
import torch.nn.functional as NF
import torch.optim as optim
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import numpy as np
import math

from pathlib import Path
from argparse import Namespace, ArgumentParser
import omegaconf

import nerfacc


from model.volsdf import VolSDF
from model.nerf import NeRF

from utils.estimator import MIPOccGridEstimator
from utils.dataset import SyntheticDataset,GlossyRealDataset
from utils.ops import volume_rendering, volume_rendering_bkgd, gamma



class ModelTrainer(pl.LightningModule):
    def __init__(self, hparams: Namespace, *args, **kwargs):
        super(ModelTrainer,self).__init__()
        self.save_hyperparameters(hparams)
        
        self.is_real = hparams.dataset != 'synthetic'
        aabb = list(hparams.aabb)
         
            
        # set up occupancy grid estimator
        self.estimator = nerfacc.OccGridEstimator(
                         roi_aabb=aabb, resolution=128,levels=1)
        self.estimator2 = MIPOccGridEstimator(roi_aabb=aabb,resolution=128,levels=5)
        if self.is_real: # background occupancy grid estimator
            self.estimator3 = nerfacc.OccGridEstimator(
                         roi_aabb=aabb, resolution=128,levels=5)
            
            
        # set up model    
        self.model = VolSDF(aabb,hparams.sdf,hparams.nde)
        if self.is_real:
            self.model_bkgd = NeRF(aabb)
        
        
        # set up volume rendering
        self.render_step_size = hparams.render_step_size
        self.model.nde.render_step_size = hparams.render_step_size_n
        if self.is_real:
            self.render_step_size_bkgd = hparams.render_step_size_bkgd
        
        
        # set up dataset
        dataset_name,dataset_path = hparams.dataset,hparams.dataset_path
        
        if dataset_name == 'synthetic':
            self.train_dataset = SyntheticDataset(dataset_path,split='train',
                batch_size=self.hparams.batch_size,device=self.hparams.device)
            self.val_dataset = SyntheticDataset(dataset_path,split='val',
                                      device=self.hparams.device)
            self.bkgd = torch.ones
            self.near = 2
            self.far = 6
        elif dataset_name == 'glossyreal':
            self.train_dataset = GlossyRealDataset(dataset_path,split='train',
                        batch_size=self.hparams.batch_size,device=self.hparams.device)
            self.val_dataset = GlossyRealDataset(dataset_path,split='val',device=self.hparams.device)
            
        self.img_hw = self.val_dataset.img_hw
        
    def __repr__(self):
        return repr(self.hparams)
    
    def configure_optimizers(self):
        if(self.hparams.optimizer == 'SGD'):
            opt = optim.SGD
        if(self.hparams.optimizer == 'Adam'):
            opt = optim.Adam
        
        optimizer = opt(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)    
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=self.hparams.milestones,gamma=self.hparams.scheduler_rate)
        return [optimizer], [scheduler]
    
    def train_dataloader(self,):
        return DataLoader(self.train_dataset, shuffle=False,batch_size=None)
    
    def val_dataloader(self,):
        return DataLoader(self.val_dataset, shuffle=False,batch_size=None)

    def forward(self, points, view):
        return

    
    def on_train_epoch_start(self,):
        # switch to blend-then-decode strategy for near-field feature
        if self.current_epoch >= 2000:
            self.model.nde.pre_decode = False
    
    def set_beta(self,):
        """ volsdf beta schedule """
        beta_init = 0.1
        beta_end = 0.001
        M = 200000
        t = np.clip(self.global_step/M,0,1)
        beta = beta_init / (1+(beta_init-beta_end)/beta_end*(t**0.8))
        self.model.beta = beta

        
    def training_step(self, batch, batch_idx):
        rays,rgbs_gt = batch['rays'],batch['rgbs']
        rays_x,rays_d = rays[...,:3],rays[...,3:6]
        
        if self.is_real:
            near,far = batch['tmins'],batch['tmaxs']
            human_pose = batch['human_pose']
        else:
            near,far = self.near,self.far
            human_pose = None
        
        
        
        
        # update hyperparameters
        self.set_beta()
        n_update = 16
        if self.global_step > 20000:
            self.model.nde.render_step_size = 1e-2
        #if self.global_step > 40000:
        #    self.model.nde.render_step_size = 5e-3
        
        
        
        
        # update occupancy grid estimator
        def occ_eval_fn(x):
            density = self.model.query_density(x)
            return density*self.render_step_size
        
        self.estimator.update_every_n_steps(
            step = self.global_step,
            occ_eval_fn=occ_eval_fn,
            occ_thre = 1e-2,
            n=n_update
        )
        
        self.estimator2.update_every_n_steps(
            self.global_step,
            self.estimator.occs,self.estimator.binaries[0],
            n=n_update
        )
        
        if self.is_real:
            def occ_eval_fn3(x):
                density = self.model_bkgd.query_density(x)
                return density*self.render_step_size

            self.estimator3.update_every_n_steps(
                step = self.global_step,
                occ_eval_fn=occ_eval_fn3,
                occ_thre = 1e-2,
                n=n_update
            )
        
        
        # get background color
        if self.is_real:
            rgbs_bkgd, ret_bkgd = volume_rendering_bkgd(
                    rays_x,rays_d,far,self.render_step_size_bkgd,
                    self.model_bkgd,self.estimator3)
        else:
            rgbs_bkgd = self.bkgd(3,device=rays_x.device)
        
        # get foreground color
        ret = volume_rendering(
                rays_x, rays_d, near, far, self.render_step_size, 
                rgbs_bkgd,
                self.model, self.estimator, self.estimator2,
                human_pose=human_pose,
                shape_loss=self.is_real&(self.global_step<1000)
        )
        
        
        # handle no sample case
        if ret is None:
            if self.is_real:
                if ret_bkgd==False: # no background to optimize
                    return None
                loss_c = NF.mse_loss(self.gamma(rgbs_bkgd),rgbs_gt)
                loss_render = (self.gamma(rgbs_bkgd)-rgbs_gt).pow(2).sum(-1).add(0.001).sqrt().mean()
                psnr = -10.0 * math.log10(loss_c.clamp_min(1e-5))
                loss = loss_render

                self.log('train/loss', loss, prog_bar=True)
                self.log('train/psnr', psnr)
                return loss
            return None    
        
        
        
        # adjust batch size
        n_rendering_samples = ret['n_rendering_samples']
        if self.hparams.target_sample_batch_size > 0:
            num_rays = len(rgbs_gt)
            num_rays = int(num_rays*
                          (self.hparams.target_sample_batch_size
                           /float(n_rendering_samples)))
            self.train_dataset.update_num_rays(num_rays)
                
        
        
        
        # loss
        loss_eikonal = ((ret['sdf_grad'].pow(2).sum(-1)+1e-12).sqrt()-1).pow(2).mean()
        loss_eikonal = 1e-1*loss_eikonal
        
        # near-field regularization loss
        loss_n = 1e-2*NF.mse_loss(gamma(ret['rgb_n']),rgbs_gt) 
        
        # NeRO shape stablization loss
        loss_shape = 1e-1*ret['loss_shape']
        
        rgbs = ret['rgb']
        loss_c = NF.mse_loss(gamma(rgbs),rgbs_gt)
        loss_render = (gamma(rgbs)-rgbs_gt).pow(2).sum(-1).add(0.001).sqrt().mean()
        psnr = -10.0 * math.log10(loss_c.clamp_min(1e-5))
        loss = loss_render + loss_eikonal + loss_n + loss_shape
        
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/psnr', psnr)
        
        return loss
    
    
    
    def validation_step(self, batch, batch_idx):
        rays,rgbs_gt = batch['rays'], batch['rgbs']
        if self.is_real:
            near,far = batch['tmins'],batch['tmaxs']
            human_pose = batch['human_pose']
        else:
            near,far = self.near,self.far
            human_pose = None
        
        rgbs = []
        depths = []
        normals = []
        roughness = []
        Ls = []
        batch_size = 1000
        S = -1
        
        for b in range(math.ceil(len(rays)*1.0/batch_size)):
            b0 = b*batch_size
            b1 = min(b0+batch_size,len(rays))
            
            ray = rays[b0:b1]
            ray_x,ray_d = ray[...,:3],ray[...,3:6]
            
            
            if self.is_real:
                rgbs_bkgd,ret_bkgd = volume_rendering_bkgd(
                        ray_x,ray_d,far[b0:b1],self.render_step_size_bkgd,
                        self.model_bkgd,self.estimator3)
                
                ret = volume_rendering(
                    ray_x, ray_d, near[b0:b1], far[b0:b1], self.render_step_size, 
                    rgbs_bkgd,
                    self.model, self.estimator, self.estimator2,
                    human_pose=human_pose[b0:b1]
                )
            else:
                rgbs_bkgd = self.bkgd(3,device=ray_x.device)
                ret = volume_rendering(
                    ray_x, ray_d, near, far, self.render_step_size, 
                    rgbs_bkgd,
                    self.model, self.estimator, self.estimator2,
                    human_pose=None
                )
                

            
            if ret is None:
                if self.is_real:
                    rgbs.append(rgbs_bkgd)
                else:
                    rgbs.append(torch.ones_like(ray_x))
                depths.append(torch.zeros_like(ray_x[...,0]))
                normals.append(torch.zeros_like(ray_x))
                roughness.append(torch.zeros_like(ray_x[...,0]))
                Ls.append(torch.zeros_like(ray_x))
            else:
                rgbs.append(ret['rgb'])
                depths.append(ret['depth'].squeeze(-1))
                normals.append(ret['normal'])
                roughness.append(ret['roughness'].squeeze(-1))
                Ls.append(ret['Ls'])
                
        rgbs = gamma(torch.cat(rgbs)).clamp(0,1)
        normals = torch.cat(normals)
        Ls = gamma(torch.cat(Ls)).clamp(0,1)
        
        roughness = torch.cat(roughness)
        depths = torch.cat(depths)
        depths /= (depths.max()+1e-4)
        
        
        loss_c = NF.mse_loss(rgbs,rgbs_gt)
        loss = loss_c
        psnr = -10*math.log10(loss_c.clamp_min(1e-12))
        
        self.log('val/loss',loss)
        self.log('val/psnr',psnr)
        
        self.logger.experiment.add_image('val/gt_image',
                    rgbs_gt.reshape(*self.img_hw,3).permute(2,0,1),batch_idx)
        self.logger.experiment.add_image('val/inf_image',
                    rgbs.reshape(*self.img_hw,3).permute(2,0,1),batch_idx)
        self.logger.experiment.add_image('val/inf_depth',
                    depths.reshape(1,*self.img_hw).expand(3,*self.img_hw),batch_idx)
        self.logger.experiment.add_image('val/roughness',
                roughness.reshape(1,*self.img_hw).expand(3,*self.img_hw),batch_idx)
        self.logger.experiment.add_image('val/normal',
                    normals.reshape(*self.img_hw,3).permute(2,0,1),batch_idx)
        self.logger.experiment.add_image('val/Ls',
                    Ls.reshape(*self.img_hw,3).permute(2,0,1),batch_idx)
        return



if __name__ == '__main__':

    torch.manual_seed(9)
    torch.cuda.manual_seed(9)


    # add PROGRAM level args
    parser = ArgumentParser()
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--config', type=str, default='configs/synthetic.yaml')
    parser.add_argument('--max_epochs', type=int, default=4000)
    parser.add_argument('--log_path', type=str, default='./logs')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints')
    parser.add_argument('--resume', dest='resume', action='store_true')
    parser.add_argument('--device', type=int, required=False,default=None)
    parser.set_defaults(resume=False)
    
    args_cli = parser.parse_args()
    
    # read config file and merge
    args = omegaconf.OmegaConf.load(args_cli.config)
    args = omegaconf.OmegaConf.merge(args,omegaconf.OmegaConf.create(vars(args_cli)))
    experiment_name = args.experiment_name

    
    # setup checkpoint loading
    checkpoint_path = Path(args.checkpoint_path) / experiment_name
    log_path = Path(args.log_path)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(checkpoint_path, monitor='val/loss', save_top_k=1, save_last=True,every_n_epochs=5)
    
    logger = TensorBoardLogger(log_path, name=experiment_name)
    
    last_ckpt = checkpoint_path / 'last.ckpt' if args.resume else None
    if (last_ckpt is None) or (not (last_ckpt.exists())):
        last_ckpt = None
    else:
        last_ckpt = str(last_ckpt)
    
    
    # setup model trainer
    model = ModelTrainer(args)
    
    trainer = Trainer(
        accelerator='gpu', devices=[args.device], 
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=1,
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=40
    )

    trainer.fit(
        model, 
        ckpt_path=last_ckpt, 
    )