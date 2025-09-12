from Trainer.Trainer_base import Trainer
from utils.loss import PET_Loss

from utils.utils import apply_deformation_using_disp
from networks.VecInt import VecInt

class VoxelMorph_Trainer(Trainer):
    def __init__(self, args):
        assert args.method in ['VM', 'VM-diff']
        # setting log name first!
        self.log_name = f'{args.method}_{args.loss}(seg_tv{args.alpha_tv}_dice{args.alpha_dice}_suvr{args.alpha_suvr})'
        self.method = args.method

        self.args = args
        self.out_channels = 3
        self.out_layers = 1

        self.loss_fn = PET_Loss(args.loss, args.alpha_tv, args.alpha_dice, args.alpha_suvr)
        self.reset_logs()

        if 'diff' in args.method:
            self.integrate = VecInt(inshape=(160, 192, 160), nsteps=7)

        super().__init__(args)

    def forward(self, img, template, stacked_input, img_segs, temp_segs, epoch=0, val=False):
        out_list, _ = self.model(stacked_input)
        out = out_list[-1] # use only last one
        if self.method == 'VM':
            deformed_img = apply_deformation_using_disp(img, out)
            deformed_segs = [apply_deformation_using_disp(s, out, mode='bilinear') for s in img_segs] # Important! : using bilinear interpolation to get gradient
            self.disp_field = out
        elif self.method == 'VM-diff':
            # velocity field to deformation field
            accumulate_disp = self.integrate(out)
            deformed_img = apply_deformation_using_disp(img, accumulate_disp)
            deformed_segs = [apply_deformation_using_disp(s, accumulate_disp, mode='bilinear') for s in img_segs]
            self.disp_field = accumulate_disp
        
        loss, sim_loss, tv_loss, dice_loss, suvr_loss = self.loss_fn(img, img_segs, deformed_img, deformed_segs, template, temp_segs, out)
        # print(sim_loss, tv_loss, dice_loss, suvr_loss)

        self.log_dict['Loss_tot'] += loss.item()
        self.log_dict['Loss_sim'] += sim_loss
        self.log_dict['Loss_tv'] += tv_loss
        self.log_dict['Loss_dice'] += dice_loss
        self.log_dict['Loss_suvr'] += suvr_loss
        
        return loss, deformed_img, deformed_segs

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
        # for single layer, deterministic version (VM)
        self.log_dict = {
            'Loss_tot':0.0,
            'Loss_sim':0.0,
            'Loss_tv':0.0,
            'Loss_dice':0.0,
            'Loss_suvr':0.0
        }

    def get_disp(self):
        return self.disp_field
