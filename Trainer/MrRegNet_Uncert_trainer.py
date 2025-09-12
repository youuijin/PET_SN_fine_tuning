from Trainer.Trainer_base import Trainer
from utils.loss import Uncert_Loss

from utils.utils import apply_deformation_using_disp
from networks.VecInt import VecInt
from datetime import datetime

import torch
import torch.nn.functional as F

class MrRegNet_Uncert_Trainer(Trainer):
    def __init__(self, args):
        assert args.method in ['Mr-Un', 'Mr-Un-diff']
        # setting log name first!
        if args.reg is None:
            self.log_name = f'{args.method}-reskl_{args.loss}'
        else:
            self.log_name = f'{args.method}-reskl_{args.loss}({args.reg}_{args.alpha}_{args.sca_fn}_{args.alp_sca})'
        self.method = args.method

        config={
            "method": args.method,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "loss":args.loss,
            "reg": args.reg,
            "image_sigma": args.image_sigma,
            "prior_lambda": args.prior_lambda
        }

        self.args = args
        self.out_channels = 6
        self.out_layers = 3

        self.loss_fn = Uncert_Loss(args.reg, args.image_sigma, args.prior_lambda)
        self.reset_logs()

        if 'diff' in args.method:
            self.integrate = VecInt(inshape=(160, 192, 160), nsteps=7, multi=True)

        super().__init__(args, config)

    def forward(self, img, template, stacked_input, epoch=0, val=False):
        mean_list, std_list, res_mean_list, res_std_list = self.model(stacked_input)
        if val:
            mean_list = mean_list[-1:]
            std_list = std_list[-1:]
        
        tot_loss = torch.tensor(0.0).to(img.device)
        # iteration accross resolution level
        for i, (mean, std, res_mean, res_std) in enumerate(zip(mean_list, std_list, res_mean_list, res_std_list)):
            cur_img = F.interpolate(img, size=mean.shape[2:], mode='nearest')
            cur_template = F.interpolate(template, size=mean.shape[2:], mode='nearest') 

            if val == False:
                eps_r = torch.randn_like(mean)
                sampled_disp = mean + eps_r * std
            else:
                sampled_disp = mean
            
            if self.method == 'Mr-Un':
                deformed_img = apply_deformation_using_disp(cur_img, sampled_disp)
            elif self.method == 'Mr-Un-diff':
                # velocity field to deformation field
                accumulate_disp = self.integrate(sampled_disp)
                deformed_img = apply_deformation_using_disp(cur_img, accumulate_disp)
            
            # loss, sim_loss, smoo_loss = self.loss_fn(deformed_img, cur_template, out)
            if i == self.out_layers-1:
                only_kl = False
            else:
                only_kl = True
            loss, sim_loss, smoo_loss, sigma_loss, sigma_var = self.loss_fn(deformed_img, cur_template, res_mean, res_std, only_kl) #IMPORTANT: change out to res_out

            tot_loss += loss

            self.log_dict['Loss_tot'] += loss.item()
            self.log_dict['Loss_sim'] += sim_loss
            self.log_dict['Loss_reg'] += smoo_loss

            self.log_dict[f'Loss_sim/res{i+1}'] += sim_loss
            self.log_dict[f'Loss_reg/res{i+1}'] += smoo_loss
        
        return tot_loss, deformed_img

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
            'Loss_reg':0.0,
            'Loss_sim/res1':0.0,
            'Loss_sim/res2':0.0,
            'Loss_sim/res3':0.0,
            'Loss_reg/res1':0.0,
            'Loss_reg/res2':0.0,
            'Loss_reg/res3':0.0
        }
