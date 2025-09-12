from Trainer.VoxelMorph_trainer import VoxelMorph_Trainer
# from Trainer.VoxelMorph_Uncert_trainer import VoxelMorph_Uncert_Trainer
from Trainer.MrRegNet_trainer import MrRegNet_Trainer
# from Trainer.MrRegNet_Uncert_trainer import MrRegNet_Uncert_Trainer
# from Trainer.VoxelMorph_Aleo_Uncert_trainer import VoxelMorph_Aleatoric_Uncert_Trainer
# from Trainer.VoxelMorph_Semantic_Free_Aware_Trainer import VoxelMorph_Semantic_Free_Aware_Trainer
# from Trainer.VoxelMorph_Semantic_Free_Aware_Each_Trainer import VoxelMorph_Semantic_Free_Aware_Each_Trainer

def set_trainer(args):
    if args.method == "VM" or args.method == "VM-diff":
        trainer = VoxelMorph_Trainer(args)
    # elif args.method == "VM-Un" or args.method == 'VM-Un-diff':
    #     trainer = VoxelMorph_Uncert_Trainer(args)
    # elif args.method == "VM-Al-Un":
    #     trainer = VoxelMorph_Aleatoric_Uncert_Trainer(args)
    elif args.method == "Mr" or args.method =='Mr-diff':
        trainer = MrRegNet_Trainer(args)
    # elif args.method == "Mr-Un" or args.method =='Mr-Un-diff':
    #     trainer = MrRegNet_Uncert_Trainer(args) 
    # #     #TODO: Scaling accross multi-resolution level
    # elif args.method == "VM-SFA" or args.method == "VM-SFA-diff":
    #     trainer = VoxelMorph_Semantic_Free_Aware_Trainer(args)
    # elif args.method == "VM-SFAeach" or args.method == "VM-SFAeach-diff":
    #     trainer = VoxelMorph_Semantic_Free_Aware_Each_Trainer(args)
    else:
        raise ValueError("Error! Undefined Method:", args.method)
    
    return trainer