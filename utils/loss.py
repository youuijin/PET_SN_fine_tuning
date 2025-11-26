from utils.loss_utils import *
import torch.nn.functional as F

class Train_Loss:
    def __init__(self, loss, reg, alpha, alpha_scale=1.0, scale_fn='exp'):
        if loss == 'MSE':
            self.loss_fn_sim = MSE_loss
        elif loss == 'NCC':
            self.loss_fn_sim = NCC_loss
        
        if reg is None:
            self.loss_fn_reg, self.reg_alpha = [], []
        else:
            assert len(reg.split("_")) == len(alpha.split("_"))
            self.loss_fn_reg, self.reg_alpha = [], [float(a) for a in alpha.split("_")]
            for r in reg.split('_'):
                if r == "tv":
                    loss_fn = tv_loss
                elif r == "l2":
                    loss_fn = l2_loss
                elif r == "jac":
                    loss_fn = jac_det_loss
                self.loss_fn_reg += [loss_fn]
        
        # scale factor for multi-resolution method
        self.alpha_scale = alpha_scale
        self.scale_fn = scale_fn

    def __call__(self, moved, fixed, disp, idx=0):
        sim_loss = self.loss_fn_sim(moved, fixed)
        reg_loss = torch.tensor(0.0, device=moved.device)
        for loss_fn, alpha in zip(self.loss_fn_reg, self.reg_alpha):
            # exponential
            if self.scale_fn == 'exp':
                alpha = (self.alpha_scale ** idx) * alpha
            # linear
            elif self.scale_fn == 'linear':
                alpha = alpha*(self.alpha_scale-1)/2*idx+alpha

            cur_loss = loss_fn(disp)
            reg_loss += alpha * cur_loss

        tot_loss = sim_loss + reg_loss
        return tot_loss, sim_loss.item(), reg_loss.item()

class PET_Loss:
    def __init__(self, loss, alpha_tv=1.0, alpha_dice=1.0, alpha_suvr=1.0, alpha_scale=1.0, scale_fn='exp', transition=False):
        if loss == 'MSE':
            self.loss_fn_sim = MSE_loss
        elif loss == 'NCC':
            self.loss_fn_sim = NCC_loss
        elif loss == 'none':
            self.loss_fn_sim = self.none_fn

        self.loss_fn_tv = self.tv_loss_l2
        self.loss_fn_dice = self.dice_loss
        self.loss_fn_suvr = self.suvr_ratio_consistency_loss
        self.alpha_tv = alpha_tv
        self.alpha_dice = alpha_dice
        self.alpha_suvr = alpha_suvr

        # scale factor for multi-resolution method
        self.alpha_scale = alpha_scale
        self.scale_fn = scale_fn

        self.transition = transition
        self.trans_alpha = 0. # to calculate alpha

    def none_fn(self, moved_pet, fixed_pet):
        return torch.tensor(0.0).to(moved_pet.device)

    def tv_loss_l2(self, displace):
        """
        displace: Tensor of shape [B, 3, D, H, W]
        TV loss는 인접 voxel 간의 L2 차이의 평균을 구하는 방식입니다.
        """
        # Depth 방향 차이 (D axis)
        dz = torch.mean((displace[:, :, 1:, :, :] - displace[:, :, :-1, :, :])**2)
        # Height 방향 차이 (H axis)
        dy = torch.mean((displace[:, :, :, 1:, :] - displace[:, :, :, :-1, :])**2)
        # Width 방향 차이 (W axis)
        dx = torch.mean((displace[:, :, :, :, 1:] - displace[:, :, :, :, :-1])**2)

        loss = (dx + dy + dz)/3.
        return loss.mean()

    # DICE: hard mask version
    # def dice_loss(self, seg_after, seg_temp, eps=1e-6):
    #     def dice_bin(a, b):
    #         # dtype/디바이스 정리 + binary 보장
    #         a = (a > 0).to(torch.float32)
    #         b = (b > 0).to(torch.float32)

    #         # 합산 축(배치 제외)
    #         dims = tuple(range(1, a.ndim))
    #         inter = (a * b).sum(dim=dims)
    #         denom = a.sum(dim=dims) + b.sum(dim=dims)

    #         # 분모가 0인 경우(두 마스크 모두 비어있는 케이스)는 dice=1로 처리
    #         dice = (2 * inter + eps) / (denom + eps)
    #         dice = torch.where(denom > 0, dice, torch.ones_like(dice))

    #         return dice.mean()  # 배치 평균

    #     seg_after_gm, seg_after_wm = seg_after
    #     seg_temp_gm, seg_temp_wm = seg_temp

    #     gm_dice = dice_bin(seg_after_gm, seg_temp_gm)
    #     wm_dice = dice_bin(seg_after_wm, seg_temp_wm)
    #     dice_avg = (wm_dice + gm_dice)/2.

    #     return (1.0 - dice_avg)

    # DICE: only GM, WM version
    # def dice_loss(self, seg_after, seg_temp, eps=1e-6, normalize=True):
    #     """
    #     seg_after: [gm_after, wm_after]  # bilinear로 워핑된 연속값 텐서 (0~1 권장)
    #     seg_temp : [gm_temp,  wm_temp ]  # 템플릿 마스크 (하드 or 소프트 모두 OK)
    #     반환: 1 - 평균 Soft Dice (GM/WM 평균)

    #     지원 shape: 각 텐서 [B,1,D,H,W] 또는 [B,D,H,W]
    #     """

    #     def to_float_wo_channel(x):
    #         # [B,1,D,H,W] -> [B,D,H,W]
    #         if x.ndim == 5 and x.shape[1] == 1:
    #             x = x.squeeze(1)
    #         return x.to(torch.float32)

    #     # 1) 입력 정리
    #     gm_a, wm_a = map(to_float_wo_channel, seg_after)  # 연속값 (soft)
    #     gm_t, wm_t = map(to_float_wo_channel, seg_temp)   # 하드(0/1) 또는 연속값

    #     # 2) 채널 결합: [B,2,D,H,W]
    #     pred = torch.stack([gm_a, wm_a], dim=1)
    #     tgt  = torch.stack([gm_t, wm_t], dim=1)

    #     # # 3) (선택) 예측 확률 재정규화: 채널합이 1 되도록 (겹침/틈 방지용)
    #     # if normalize:
    #     #     pred = pred.clamp(0, 1)
    #     #     pred = pred / (pred.sum(dim=1, keepdim=True) + eps)

    #     # 4) Soft Dice (클래스별 평균 후 다시 평균) — BG 없음(GM/WM 두 채널만)
    #     dims = (2, 3, 4)
    #     inter = (pred * tgt).sum(dim=dims)                                   # [B,2]
    #     denom = (pred.pow(2).sum(dim=dims) + tgt.pow(2).sum(dim=dims))       # [B,2]
    #     dice  = (2.0 * inter + eps) / (denom + eps)                          # [B,2]
    #     dice_loss  = 1.0 - dice.mean()                                            # 스칼라

    #     return dice_loss

    def dice_loss(self, seg_after, seg_temp, eps=1e-6, normalize=True):
        """
        seg_after: # bilinear로 워핑된 연속값 텐서 (0~1 권장)
        seg_temp : # 템플릿 마스크 (하드 or 소프트 모두 OK)
        반환: 1 - 평균 Soft Dice (모든 region에 대한 평균)

        지원 shape: 각 텐서 [B,1,D,H,W] 또는 [B,D,H,W]
        """

        def to_float_wo_channel(x):
            # [B,1,D,H,W] -> [B,D,H,W]
            if x.ndim == 5 and x.shape[1] == 1:
                x = x.squeeze(1)
            return x.to(torch.float32)

        # 1) 입력 정리
        sa_1, sa_2, sa_3, sa_4, sa_5, sa_6 = map(to_float_wo_channel, seg_after)  # 연속값 (soft)
        st_1, st_2, st_3, st_4, st_5, st_6 = map(to_float_wo_channel, seg_temp)   # 하드(0/1) 또는 연속값

        # 2) 채널 결합: [B,6,D,H,W]
        pred = torch.stack([sa_1, sa_2, sa_3, sa_4, sa_5, sa_6], dim=1)
        tgt  = torch.stack([st_1, st_2, st_3, st_4, st_5, st_6], dim=1)

        # # 3) (선택) 예측 확률 재정규화: 채널합이 1 되도록 (겹침/틈 방지용)
        # if normalize:
        #     pred = pred.clamp(0, 1)
        #     pred = pred / (pred.sum(dim=1, keepdim=True) + eps)

        # 4) Soft Dice (클래스별 평균 후 다시 평균) — BG 없음(GM/WM 두 채널만)
        dims = (2, 3, 4)
        inter = (pred * tgt).sum(dim=dims)                                   # [B,6]
        denom = (pred.pow(2).sum(dim=dims) + tgt.pow(2).sum(dim=dims))       # [B,6]
        dice  = (2.0 * inter + eps) / (denom + eps)                          # [B,6]
        dice_loss  = 1.0 - dice.mean()                                       # scalar

        return dice_loss
    
    # def suvr_ratio_consistency_loss(self, seg_before, img_before, seg_temp, img_after, eps=1e-6):
    #     # seg_*: 정렬 전/후 세그 (GM/WM 레이블), img_*: 정렬 전/후 PET
    #     gm_bef, wm_bef = seg_before
    #     gm_aft, wm_aft = seg_temp

    #     def mean_within(mask, img):
    #         mask = (mask > 0).to(img.dtype)
    #         # 채널 차원 정리
    #         if img.ndim == 5 and img.shape[1] == 1: img = img.squeeze(1)
    #         if mask.ndim == 5 and mask.shape[1] == 1: mask = mask.squeeze(1)
    #         dims = tuple(range(1, img.ndim))
    #         num = (img * mask).sum(dim=dims)
    #         den = mask.sum(dim=dims)
    #         return num / (den + eps)  # [B]

    #     gm_b = mean_within(gm_bef, img_before)  # [B]
    #     wm_b = mean_within(wm_bef, img_before)  # [B]
    #     gm_a = mean_within(gm_aft, img_after)   # [B]
    #     wm_a = mean_within(wm_aft, img_after)   # [B]

    #     r_b = gm_b / (wm_b + eps)  # 정렬 전 GM/WM 비
    #     r_a = gm_a / (wm_a + eps)  # 정렬 후 GM/WM 비
    #     return (r_b - r_a).abs().mean()   # L1 norm

    def suvr_ratio_consistency_loss(self, seg_before, img_before, seg_temp, img_after, eps=1e-6, detach_ref=False):
        def _to_BDHW(x):
            # [B, 1, D, H, W] → [B, D, H, W], 이미 [B, D, H, W]면 그대로
            if x.ndim == 5 and x.shape[1] == 1:
                return x[:, 0]
            return x

        def mean_within(mask, img):
            img = _to_BDHW(img)
            mask = _to_BDHW(mask)
            # 마스크 이진화(마스크는 보통 레이블/확률이며, 여기서는 >0을 내부영역으로 사용)
            m = (mask > 0).to(img.dtype)
            # [B, ...] → [B, N]으로 펴서 합
            num = (img * m).flatten(1).sum(1)             # [B]
            den = m.flatten(1).sum(1)                     # [B]
            return num / (den + eps)                      # [B]

        # 참조 영역 평균 (리스트의 마지막 레벨 사용)
        ref_before = mean_within(seg_before[-1], img_before)   # [B]
        ref_after  = mean_within(seg_temp[-1],   img_after)    # [B]
        if detach_ref:
            ref_before = ref_before.detach()
            ref_after  = ref_after.detach()

        # 각 스케일(마지막 제외)에서 평균을 계산해 쌓음 → [S-1, B]
        b_means = torch.stack([mean_within(sb, img_before) for sb in seg_before[:-1]], dim=0)
        a_means = torch.stack([mean_within(st, img_after) for st in seg_temp[:-1]],  dim=0)

        # SUVR = 영역평균 / 참조영역평균
        ref_before = ref_before.unsqueeze(0)  # [1, B]로 브로드캐스트
        ref_after  = ref_after.unsqueeze(0)   # [1, B]
        r_b = b_means / (ref_before + eps)    # [S-1, B]
        r_a = a_means / (ref_after  + eps)    # [S-1, B]

        # L1 일관성 손실(스케일, 배치 평균)
        return F.l1_loss(r_b, r_a, reduction='mean')

    def __call__(self, moving_pet, moving_seg, moved_pet, moved_seg, fixed_pet, fixed_seg, disp, idx=0):
        '''
        - seg shape: [2, [B, 1, H, W, D]]
        - moving_pet, moving_seg: original
        - moved_pet, moved_seg: deformed
        - fixed_pet, fixed_seg: PET Template
        - disp: displacement field (for regularizer)
        '''
        sim_loss = self.loss_fn_sim(moved_pet, fixed_pet)
        tv_loss = self.loss_fn_tv(disp)
        dice_loss = self.loss_fn_dice(moved_seg, fixed_seg)
        suvr_loss = self.loss_fn_suvr(moving_seg, moving_pet, fixed_seg, moved_pet)

        # set layer-wise alpha
        alpha_tv = self.alpha_tv * (self.alpha_scale**idx) # smaller
        alpha_dice = self.alpha_dice * (self.alpha_scale**(-idx)) ## larger
        alpha_suvr = self.alpha_suvr * (self.alpha_scale**(-idx)) ## larger

        if not self.transition:
            tot_loss = sim_loss + alpha_tv*tv_loss + alpha_dice*dice_loss + alpha_suvr*suvr_loss
        else:
            tot_loss = (1-self.trans_alpha)*sim_loss + self.trans_alpha*(alpha_tv*tv_loss + alpha_dice*dice_loss + alpha_suvr*suvr_loss)

        return tot_loss, sim_loss.item(), tv_loss.item(), dice_loss.item(), suvr_loss.item()