import torch
import torch.nn as nn
import torch.nn.functional as F

class U_Net(nn.Module):
    def __init__(self, in_channels=2, out_channels=3, out_layers=1):
        super(U_Net, self).__init__()

        assert out_channels in [3, 6]
        assert out_layers in [1, 2, 3, 4]

        self.out_layers = out_layers
        self.out_channels = out_channels
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.enc2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.enc3 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.enc4 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )

        self.dec1 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )
        # Decoder (Up-sampling & Skip Connection 적용)
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose3d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2)
        )
        self.dec2 = nn.Sequential(
            nn.Conv3d(32+32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2)
        )
        self.dec3 = nn.Sequential(
            nn.Conv3d(32+32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.upsample3 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2)
        )
        self.dec4 = nn.Sequential(
            nn.Conv3d(16+16, 8, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2)
		)

        self.flows = nn.ModuleList([nn.Identity() for _ in range(4)])
        for i, res in enumerate([32, 32, 32, 8]):
            if 4 - self.out_layers <= i:
                self.flows[i] = nn.Conv3d(res, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)

        # decoding
        if self.out_channels == 6:
            means = [] # only residual
            stds = [] # model output = log \sigma^2 -> exp(0.5*log \sigma^2)
            
            x5 = self.dec1(x4)
            concated = self.flows[0](x5)
            means.append(concated[:, :3])
            stds.append(torch.exp(0.5 * concated[:, 3:])) # output = log variance
            up_x5 = self.upsample1(x5)
            
            x6 = self.dec2(torch.cat([x3, up_x5], dim=1))
            concated = self.flows[1](x6)
            means.append(concated[:, :3])
            stds.append(torch.exp(0.5 * concated[:, 3:]))
            up_x6 = self.upsample2(x6)
            
            x7 = self.dec3(torch.cat([x2, up_x6], dim=1))
            concated = self.flows[2](x7)
            means.append(concated[:, :3])
            stds.append(torch.exp(0.5 * concated[:, 3:]))
            up_x7 = self.upsample3(x7)
            
            x8 = self.dec4(torch.cat([x1, up_x7], dim=1))
            concated = self.flows[3](x8)
            means.append(concated[:, :3])
            stds.append(torch.exp(0.5 * concated[:, 3:]))

            means, stds = means[-self.out_layers:], stds[-self.out_layers:]

            tot_means = self.combine_residuals(means)
            tot_stds = self.combine_residuals_std(stds)
            
            return tot_means, tot_stds, means, stds
        
        elif self.out_channels == 3:
            disp = []
            x5 = self.dec1(x4)
            concated = self.flows[0](x5)
            disp.append(concated)
            up_x5 = self.upsample1(x5)
            
            x6 = self.dec2(torch.cat([x3, up_x5], dim=1))
            concated = self.flows[1](x6)
            disp.append(concated)
            up_x6 = self.upsample2(x6)
            
            x7 = self.dec3(torch.cat([x2, up_x6], dim=1))
            concated = self.flows[2](x7)
            disp.append(concated)
            up_x7 = self.upsample3(x7)
            
            x8 = self.dec4(torch.cat([x1, up_x7], dim=1))
            concated = self.flows[3](x8)
            disp.append(concated)

            disp = disp[-self.out_layers:]

            tot_disp = self.combine_residuals(disp)
            
            return tot_disp, disp
    
    def combine_residuals(self, flows):
        tot_flows = [flows[0]]
        for f in flows[1:]:
            prev = F.interpolate(tot_flows[-1], size=f.shape[2:], mode='trilinear')
            tot_flows.append(prev + f)
        return tot_flows
    
    def combine_residuals_std(self, stds):
        tot_vars = [stds[0]]  # σ₁²
        for s in stds[1:]:
            prev = F.interpolate(tot_vars[-1], size=s.shape[2:], mode='trilinear', align_corners=True)
            tot_vars.append(torch.sqrt(prev ** 2 + s ** 2))
        return tot_vars