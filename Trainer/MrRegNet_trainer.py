from Trainer.Trainer_base import Trainer
from utils.loss import PET_Loss

from utils.utils import apply_deformation_using_disp
from networks.VecInt import VecInt
from datetime import datetime

import torch.nn as nn
import torch
import torch.nn.functional as F


class MrRegNet_Trainer(Trainer):
    def __init__(self, args):
        assert args.method in ['Mr', 'Mr-diff']
        # setting log name first!
        self.log_name = f'{args.method}_{args.loss}(seg_tv{args.alpha_tv}_dicenew{args.alpha_dice}_suvr{args.alpha_suvr}_sca{args.alp_sca})'
        self.method = args.method

        self.args = args
        self.out_channels = 3
        self.out_layers = 3

        self.loss_fn = PET_Loss(args.loss, args.alpha_tv, args.alpha_dice, args.alpha_suvr, alpha_scale=args.alp_sca)
        self.reset_logs()

        if 'diff' in args.method:
            self.integrate = VecInt(inshape=(160, 192, 160), nsteps=7, multi=True)

        super().__init__(args)

    def forward(self, img, template, stacked_input, img_segs, temp_segs, epoch=0, val=False):
        out_list, res_out_list = self.model(stacked_input)
        if val:
            out_list = out_list[-1:]
        
        tot_loss = torch.tensor(0.0).to(img.device)
        # iteration accross resolution level
        for i, (out, res_out) in enumerate(zip(out_list, res_out_list)):
            cur_img = F.interpolate(img, size=out.shape[2:], mode='nearest')
            cur_template = F.interpolate(template, size=out.shape[2:], mode='nearest') 
            cur_temp_segs = [F.interpolate(s, size=out.shape[2:], mode='nearest') for s in temp_segs]
            cur_img_segs = [F.interpolate(s, size=out.shape[2:], mode='nearest') for s in img_segs]
            
            if self.method == 'Mr':
                deformed_img = apply_deformation_using_disp(cur_img, out)
                deformed_segs = [apply_deformation_using_disp(s, out, mode='bilinear') for s in cur_img_segs]
            elif self.method == 'Mr-diff':
                # velocity field to deformation field
                accumulate_disp = self.integrate(out)
                deformed_img = apply_deformation_using_disp(cur_img, accumulate_disp)
                deformed_segs = [apply_deformation_using_disp(s, out, mode='bilinear') for s in cur_img_segs]
            
            # loss, sim_loss, smoo_loss = self.loss_fn(deformed_img, cur_template, res_out) #IMPORTANT: change out to res_out
            loss, sim_loss, tv_loss, dice_loss, suvr_loss = self.loss_fn(cur_img, cur_img_segs, deformed_img, deformed_segs, cur_template, cur_temp_segs, res_out, idx=i)

            tot_loss += loss

            self.log_dict['Loss_tot'] += loss.item()
            self.log_dict['Loss_sim'] += sim_loss
            self.log_dict['Loss_tv'] += tv_loss
            self.log_dict['Loss_dice'] += dice_loss
            self.log_dict['Loss_suvr'] += suvr_loss

            self.log_dict[f'Loss_sim/res{i+1}'] += sim_loss
            self.log_dict[f'Loss_tv/res{i+1}'] += tv_loss
            self.log_dict[f'Loss_dice/res{i+1}'] += dice_loss
            self.log_dict[f'Loss_suvr/res{i+1}'] += suvr_loss
        
        return tot_loss, deformed_img, deformed_segs

    def log(self, epoch, phase=None):
        if phase not in ['train', 'valid']:
            raise ValueError("Trainer's log function can only get phase ['train', 'valid'], but received", phase)

        if phase == 'train':
            num = len(self.train_loader)
            tag = 'Train'
        elif phase == 'valid':
            num = len(self.val_loader)
            tag = 'Val'
        
        for key, value in self.log_dict.items():
            self.writer.add_scalar(f"{tag}/{key}", value/num, epoch)

    def reset_logs(self):
        # for multi-resolution layer, deterministic version (Mr)
        self.log_dict = {
            'Loss_tot':0.0,
            'Loss_sim':0.0,
            'Loss_tv':0.0,
            'Loss_dice':0.0,
            'Loss_suvr':0.0,
            'Loss_sim/res1':0.0,
            'Loss_sim/res2':0.0,
            'Loss_sim/res3':0.0,
            'Loss_tv/res1':0.0,
            'Loss_tv/res2':0.0,
            'Loss_tv/res3':0.0,
            'Loss_dice/res1':0.0,
            'Loss_dice/res2':0.0,
            'Loss_dice/res3':0.0,
            'Loss_suvr/res1':0.0,
            'Loss_suvr/res2':0.0,
            'Loss_suvr/res3':0.0
        }
