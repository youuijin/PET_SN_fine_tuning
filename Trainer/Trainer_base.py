import os, torch, shutil, warnings
import torch.optim as optim
import matplotlib.pyplot as plt
from networks.network_utils import set_model
from utils.dataset import set_dataloader_usingcsv, set_datapath, set_paired_dataloader_usingcsv
from utils.utils import set_seed, save_middle_slices, save_middle_slices_mfm, print_with_timestamp, save_grid_spline

from datetime import datetime
from zoneinfo import ZoneInfo

from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, args, config=None):
        set_seed(seed=args.seed)

        # train options
        self.epochs = args.epochs
        self.val_interval = args.val_interval
        self.save_interval = args.save_interval
        self.save_num = args.save_num
        if args.pretrained_path is not None:
            if args.freeze:
                self.log_name = f'{self.log_name}_fFT_{args.freeze_type}'
            else:
                self.log_name = f'{self.log_name}_FT' # finetuning
        
        if args.epochs != 200:
            self.log_name = f'{self.log_name}_epochs{args.epochs}'
        if args.lr_scheduler == 'multistep':
            self.log_name = f'{self.log_name}_sche(multi_{args.lr_milestones})'
        if args.transform:
            self.log_name = f'{self.log_name}_aug'

        # add start time
        now = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%m-%d_%H-%M")
        self.log_name = f'{self.log_name}_{now}'

        self.writer = SummaryWriter(log_dir=f'{args.log_dir}/pair_{args.pair_train}/{args.dataset}/{args.model}/{self.log_name}')
        
        # Setting Model
        self.model = set_model(args.model, out_channels=self.out_channels, out_layers=self.out_layers)
        if args.pretrained_path is not None:
            self.model.load_state_dict(torch.load(args.pretrained_path, weights_only=True,map_location=torch.device('cpu')))
        self.model = self.model.cuda()

        # self.train_data_path, self.val_data_path = set_datapath(args.dataset, args.numpy)
        if not args.pair_train:
            self.train_loader, self.val_loader, self.save_loader = set_dataloader_usingcsv(args.dataset, 'data/data_list', args.template_path, args.batch_size, numpy=args.numpy, transform=args.transform)
            self.save_dir = f'./results/template/saved_models/{args.dataset}/{args.model}'
        else:
            self.train_loader, self.val_loader, self.save_loader = set_paired_dataloader_usingcsv(args.dataset, 'data/data_list', args.batch_size, numpy=args.numpy, transform=args.transform)
            self.save_dir = f'./results/pair/saved_models/{args.dataset}/{args.model}'
        
        os.makedirs(f'{self.save_dir}/completed', exist_ok=True)
        os.makedirs(f'{self.save_dir}/not_finished', exist_ok=True)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        # set learning rate scheduler 
        if args.lr_scheduler == 'none':
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1.0)
        elif args.lr_scheduler == 'multistep':
            milestones = [int(i)*len(self.train_loader) for i in args.lr_milestones.split(',')]
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.1)

        if args.freeze:
            self.model, self.optimizer = set_model_freeze(self.model, self.optimizer, args.lr, args.freeze_type)
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")
        #     self.lr_scheduler.step(args.start_epoch * len(self.train_loader))

    def train(self):
        best_loss = 1e+9
        cnt = 0
        # save template img
        for epoch in range(0, self.epochs):
            self.train_1_epoch(epoch)
            if epoch % self.val_interval == 0:
                with torch.no_grad():
                    self.valid(epoch)
                    cur_loss = self.log_dict['Loss_tot']
                    if best_loss > cur_loss:
                        cnt = 0
                        best_loss = cur_loss
                        torch.save(self.model.state_dict(), f'{self.save_dir}/not_finished/{self.log_name}_best.pt')
                    else: 
                        cnt+=1

                    # if cnt >= 3:
                    #     # early stop
                    #     break
                torch.save(self.model.state_dict(), f'{self.save_dir}/not_finished/{self.log_name}_last.pt')
            if epoch % self.save_interval == 0:
                with torch.no_grad():
                    self.save_imgs(epoch, self.save_num)

        # move trained model to complete folder 
        try:
            shutil.move(f'{self.save_dir}/not_finished/{self.log_name}_best.pt', f'{self.save_dir}/completed/{self.log_name}_best.pt')
            shutil.move(f'{self.save_dir}/not_finished/{self.log_name}_last.pt', f'{self.save_dir}/completed/{self.log_name}_last.pt')
        except Exception as e:
            print_with_timestamp(f"Failed to move {self.save_dir}/not_finished/{self.log_name}.pt: {e}")

    def train_1_epoch(self, epoch):
        self.reset_logs()
        self.model.train()
        tot_loss = 0.
        for (img, template, _, _, _, img_segs, temp_segs) in self.train_loader:
            img, template = img.unsqueeze(1).cuda(), template.unsqueeze(1).cuda() # [B, D, H, W] -> [B, 1, D, H, W]
            img_segs, temp_segs = [i.unsqueeze(1).cuda() for i in img_segs], [t.unsqueeze(1).cuda() for t in temp_segs]
            stacked_input = torch.cat([img, template], dim=1) # [B, 2, D, H, W]

            # forward & calculate loss in child trainer
            loss, _, _ = self.forward(img, template, stacked_input, img_segs, temp_segs, epoch)
            tot_loss += loss.item()

            # backward & update model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

        print_with_timestamp(f'Epoch {epoch}: train loss {round(tot_loss/len(self.train_loader), 4)}')

        # log into wandb
        self.log(epoch, phase='train')

    def valid(self, epoch):
        self.reset_logs()
        self.model.eval()
        tot_loss = 0.
        for (img, template, _, _, _, img_segs, temp_segs) in self.val_loader:
            img, template = img.unsqueeze(1).cuda(), template.unsqueeze(1).cuda() # [B, D, H, W] -> [B, 1, D, H, W]
            img_segs, temp_segs = [i.unsqueeze(1).cuda() for i in img_segs], [t.unsqueeze(1).cuda() for t in temp_segs]
            stacked_input = torch.cat([img, template], dim=1) # [B, 2, D, H, W]

            # forward & calculate loss in child trainer
            loss, _, _ = self.forward(img, template, stacked_input, img_segs, temp_segs, epoch, val=True)
            tot_loss += loss.item()

        print_with_timestamp(f'Epoch {epoch}: valid loss {round(tot_loss/len(self.val_loader), 4)}')

        self.log(epoch, phase='valid')

    def save_imgs(self, epoch, num):
        self.model.eval()
        for idx, (img, template, _, _, _, img_segs, temp_segs) in enumerate(self.save_loader):
            if idx >= num:
                break
            img, template = img.unsqueeze(1).cuda(), template.unsqueeze(1).cuda() # [B, D, H, W] -> [B, 1, D, H, W]
            img_segs, temp_segs = [i.unsqueeze(1).cuda() for i in img_segs], [t.unsqueeze(1).cuda() for t in temp_segs]
            stacked_input = torch.cat([img, template], dim=1) # [B, 2, D, H, W]

            # forward & calculate loss in child trainer
            _, deformed_img, deformed_segs = self.forward(img, template, stacked_input, img_segs, temp_segs, epoch, val=True)
            disp = self.get_disp()

            fig = save_middle_slices_mfm(img, template, deformed_img, epoch, idx)
            self.writer.add_figure(f'deformed_slices_img{idx}', fig, epoch)
            plt.close(fig)
            
            for name, (img_seg, temp_seg, deformed_seg) in enumerate(zip(img_segs, temp_segs, deformed_segs)):
                fig = save_middle_slices_mfm(img_seg, temp_seg, deformed_seg, epoch, idx)
                self.writer.add_figure(f'deformed_slices_img{idx}_seg{name}', fig, epoch)
                plt.close(fig)

            fig = save_grid_spline(disp)
            self.writer.add_figure(f'disps_img{idx}', fig, epoch)
            plt.close(fig)

        print_with_timestamp(f'Epoch {epoch}: Successfully saved {num} images')


import torch
import torch.nn as nn

def set_model_freeze(model: nn.Module, optimizer, lr, freeze_type):
    if freeze_type == 's1':
        # freeze enc1, enc2
        # train  enc3, enc4, dec1~dec4, upsample1~3, flows
        for p in model.enc1.parameters(): p.requires_grad = False
        for p in model.enc2.parameters(): p.requires_grad = False

        for p in model.enc3.parameters(): p.requires_grad = True
        for p in model.enc4.parameters(): p.requires_grad = True
        for p in model.dec1.parameters(): p.requires_grad = True
        for p in model.dec2.parameters(): p.requires_grad = True
        for p in model.dec3.parameters(): p.requires_grad = True
        for p in model.dec4.parameters(): p.requires_grad = True
        for p in model.upsample1.parameters(): p.requires_grad = True
        for p in model.upsample2.parameters(): p.requires_grad = True
        for p in model.upsample3.parameters(): p.requires_grad = True

        for m in model.flows:
            if isinstance(m, nn.Conv3d):
                for p in m.parameters(): p.requires_grad = True
    elif freeze_type == 's2':
        # freeze enc1~enc4
        # train  dec1~dec4, upsample1~3, flows
        for p in model.enc1.parameters(): p.requires_grad = False
        for p in model.enc2.parameters(): p.requires_grad = False
        for p in model.enc3.parameters(): p.requires_grad = False
        for p in model.enc4.parameters(): p.requires_grad = False

        for p in model.dec1.parameters(): p.requires_grad = True
        for p in model.dec2.parameters(): p.requires_grad = True
        for p in model.dec3.parameters(): p.requires_grad = True
        for p in model.dec4.parameters(): p.requires_grad = True
        for p in model.upsample1.parameters(): p.requires_grad = True
        for p in model.upsample2.parameters(): p.requires_grad = True
        for p in model.upsample3.parameters(): p.requires_grad = True

        for m in model.flows:
            if isinstance(m, nn.Conv3d):
                for p in m.parameters(): p.requires_grad = True
    elif freeze_type == 's3':
        # freeze enc3, enc4, dec1, upsample1
        # train  enc1, enc2, dec2~dec4, upsample2, upsample3, flows
        for p in model.enc1.parameters(): p.requires_grad = True
        for p in model.enc2.parameters(): p.requires_grad = True

        for p in model.enc3.parameters(): p.requires_grad = False
        for p in model.enc4.parameters(): p.requires_grad = False
        for p in model.dec1.parameters(): p.requires_grad = False
        for p in model.upsample1.parameters(): p.requires_grad = False

        for p in model.dec2.parameters(): p.requires_grad = True
        for p in model.dec3.parameters(): p.requires_grad = True
        for p in model.dec4.parameters(): p.requires_grad = True
        for p in model.upsample2.parameters(): p.requires_grad = True
        for p in model.upsample3.parameters(): p.requires_grad = True

        for m in model.flows:
            if isinstance(m, nn.Conv3d):
                for p in m.parameters(): p.requires_grad = True
    elif freeze_type == 's4':
        # freeze enc1~enc4, dec1~dec4, upsample1~upsample3
        # train  flows
        for p in model.enc1.parameters(): p.requires_grad = False
        for p in model.enc2.parameters(): p.requires_grad = False
        for p in model.enc3.parameters(): p.requires_grad = False
        for p in model.enc4.parameters(): p.requires_grad = False

        for p in model.dec1.parameters(): p.requires_grad = False
        for p in model.dec2.parameters(): p.requires_grad = False
        for p in model.dec3.parameters(): p.requires_grad = False
        for p in model.dec4.parameters(): p.requires_grad = False
        for p in model.upsample1.parameters(): p.requires_grad = False
        for p in model.upsample2.parameters(): p.requires_grad = False
        for p in model.upsample3.parameters(): p.requires_grad = False

        for m in model.flows:
            if isinstance(m, nn.Conv3d):
                for p in m.parameters(): p.requires_grad = True
    else:
        raise ValueError("Check freeze option:", freeze_type)
    # 
    # 1) 기존 방식 = enc1, 2, 3,은 freeze, 나머지는 train
    # for p in model.enc1.parameters(): p.requires_grad = False
    # for p in model.enc2.parameters(): p.requires_grad = False
    # for p in model.enc3.parameters(): p.requires_grad = False

    # for p in model.enc4.parameters(): p.requires_grad = True
    # for p in model.dec1.parameters(): p.requires_grad = True
    # for p in model.dec2.parameters(): p.requires_grad = True
    # for p in model.dec3.parameters(): p.requires_grad = True
    # for p in model.dec4.parameters(): p.requires_grad = True
    # for p in model.upsample1.parameters(): p.requires_grad = True
    # for p in model.upsample2.parameters(): p.requires_grad = True
    # for p in model.upsample3.parameters(): p.requires_grad = True

    # for m in model.flows:
    #     if isinstance(m, nn.Conv3d):
    #         for p in m.parameters(): p.requires_grad = True

    # 2) Optimizer (동일 learning rate)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr)

    return model, optimizer
