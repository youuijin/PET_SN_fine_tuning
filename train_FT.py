# multi_runner.py

import argparse
from main import main  # main.py에 정의된 함수
import copy

def get_base_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='FDG_PET', choices=['FDG_MRI', 'FDG_PET'])
    parser.add_argument("--model", type=str, default='U_Net', choices=['U_Net'])
    parser.add_argument("--template_path", type=str, default="data/core_MNI152_PET_1mm.nii")
    # parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--pretrained_path", default=None)
    parser.add_argument("--freeze", action='store_true', default=False)
    parser.add_argument("--freeze_type", type=str, default='s1')
    parser.add_argument("--transform", default=False, action='store_true')

    # training options # TODO: 정리
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--method", type=str, default='VM', choices=['VM', 'Mr', 'VM-Un', 'Mr-Un', 'VM-diff', 'Mr-diff', 'VM-Un-diff'])
    parser.add_argument("--loss", type=str, default="NCC", choices=["NCC", "MSE", "none"])
    parser.add_argument("--numpy", action='store_true', default=True)
    
    # for regularizer
    parser.add_argument("--alpha_suvr", type=float, default=1.0)
    parser.add_argument("--alpha_tv", type=float, default=1.0)
    parser.add_argument("--alpha_dice", type=float, default=1.0)
    parser.add_argument("--alp_sca", type=float, default=1.0)
    parser.add_argument("--sca_fn", type=str, default='exp', choices=['exp', 'linear'])

    # validation options
    parser.add_argument("--val_interval", type=int, default=5)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--save_num", type=int, default=2)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr_scheduler", type=str, default='none', choices=['none', 'multistep'])
    parser.add_argument("--lr_milestones", type=str, default=None)
    parser.add_argument("--pair_train", default=False, action='store_true')

    parser.add_argument("--transition", default=False, action='store_true')
    parser.add_argument("--transition_period", default=1.0, type=float)


    # log options
    parser.add_argument("--log_dir", type=str, default='logs')
    return parser.parse_args([])  # 빈 리스트로 기본 args 객체만 생성

def run_all():
    base_args = get_base_args()
    base_args.epochs = 100
    base_args.model = 'U_Net'
    base_args.loss = 'NCC'
    base_args.pretrained_path = 'pretrained_models/OASIS/pair/VM-diff_NCC(tv_24.0)_epochs400_07-02_01-02_last.pt'

    base_args.numpy = True
    base_args.template_path = 'data/core_MNI152_PET_1mm.npy'
    base_args.transition = True
    base_args.transition_period = 0.3
    # base_args.pair_train = True
    # base_args.freeze = True

    base_args.transform = True

    configs = [
        ("VM-diff", 0.5, 0.5, 0.5, None)
    ]
    for method, alpha_tv, alpha_dice, alpha_suvr, freeze_type in configs:
        args = copy.deepcopy(base_args)
        args.method = method
        args.alpha_tv = alpha_tv
        args.alpha_dice = alpha_dice
        args.alpha_suvr = alpha_suvr
        args.freeze_type = freeze_type

        if 'Mr' in method:
            args.alp_sca = 0.5

        main(args)

if __name__ == "__main__":
    import os
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    os.environ["NUMEXPR_NUM_THREADS"] = "4"

    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)

    run_all()
