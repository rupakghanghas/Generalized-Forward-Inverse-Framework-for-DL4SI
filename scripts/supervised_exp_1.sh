#!/bin/bash

#SBATCH --cpus-per-task=16 # this requests 1 node, 16 core. 
#SBATCH --time=6-00:00:00 
#SBATCH --gres=gpu:1
#SBATCH --partition=dgx_normal_q



export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
ulimit -u 65536

module reset
module load Anaconda3/2020.11

# new virtual environment
source activate env


###############################################################################################################
############# DATASET : CURVEVEL-A
###############################################################################################################

####################################################
# # UNetInverseModel - Large (34.68M) depth=2, repeat=2
####################################################

# FlatVel-A
# python -u train_inverse.py -ds flatvel-a -o ./SupervisedExperiment/FlatVel-A/UNetInverseModel_33M/ --tensorboard -eb 150 -m UNetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/flatvel_a_train.txt -v ./train_test_splits/flatvel_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 2

# # FlatVel-B
# python -u train_inverse.py -ds flatvel-b -o ./SupervisedExperiment/FlatVel-B/UNetInverseModel_33M/ --tensorboard -eb 150 -m UNetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/flatvel_b_train.txt -v ./train_test_splits/flatvel_b_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 2

# # CurveVel-A
# python -u train_inverse.py -ds curvevel-a -o ./SupervisedExperiment/CurveVel-A/UNetInverseModel_33M/ --tensorboard -eb 150 -m UNetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/curvevel_a_train.txt -v ./train_test_splits/curvevel_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 2

# # CurveVel-B
# python -u train_inverse.py -ds curvevel-b -o ./SupervisedExperiment/CurveVel-B/UNetInverseModel_33M/ --tensorboard -eb 150 -m UNetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/curvevel_b_train.txt -v ./train_test_splits/curvevel_b_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 2

# # FlatFault-A
# python -u train_inverse.py -ds flatfault-a -o ./SupervisedExperiment/FlatFault-A/UNetInverseModel_33M/ --tensorboard -eb 150 -m UNetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/flatfault_a_train.txt -v ./train_test_splits/flatfault_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 2

# # FlatFault-B
# python -u train_inverse.py -ds flatfault-b -o ./SupervisedExperiment/FlatFault-B/UNetInverseModel_33M/ --tensorboard -eb 150 -m UNetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/flatfault_b_train.txt -v ./train_test_splits/flatfault_b_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 2

# # CurveFault-A
# python -u train_inverse.py -ds curvefault-a -o ./SupervisedExperiment/CurveFault-A/UNetInverseModel_33M/ --tensorboard -eb 150 -m UNetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/curvefault_a_train.txt -v ./train_test_splits/curvefault_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 2

# # CurveFault-B
# python -u train_inverse.py -ds curvefault-b -o ./SupervisedExperiment/CurveFault-B/UNetInverseModel_33M/ --tensorboard -eb 150 -m UNetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/curvefault_b_train.txt -v ./train_test_splits/curvefault_b_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 2

# # Style-A
# python -u train_inverse.py -ds style-a -o ./SupervisedExperiment/Style-A/UNetInverseModel_33M/ --tensorboard -eb 150 -m UNetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/style_a_train.txt -v ./train_test_splits/style_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 2

# # Style-B
# python -u train_inverse.py -ds style-b -o ./SupervisedExperiment/Style-B/UNetInverseModel_33M/ --tensorboard -eb 150 -m UNetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/style_b_train.txt -v ./train_test_splits/style_b_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 2


####################################################
# # UNetInverseModel - Medium (23.48M) depth=1, repeat=6
####################################################


####################################################
# # UNetInverseModel - Small (17.86M) depth=2, repeat=1
####################################################

# FlatVel-A
# python -u train_inverse.py -ds flatvel-a -o ./SupervisedExperiment/FlatVel-A/UNetInverseModel_17M/ --tensorboard -eb 150 -m UNetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/flatvel_a_train.txt -v ./train_test_splits/flatvel_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 1

# # FlatVel-B
# python -u train_inverse.py -ds flatvel-b -o ./SupervisedExperiment/FlatVel-B/UNetInverseModel_17M/ --tensorboard -eb 150 -m UNetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/flatvel_b_train.txt -v ./train_test_splits/flatvel_b_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 1

# # CurveVel-A
# python -u train_inverse.py -ds curvevel-a -o ./SupervisedExperiment/CurveVel-A/UNetInverseModel_17M/ --tensorboard -eb 150 -m UNetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/curvevel_a_train.txt -v ./train_test_splits/curvevel_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 1

# # CurveVel-B
# python -u train_inverse.py -ds curvevel-b -o ./SupervisedExperiment/CurveVel-B/UNetInverseModel_17M/ --tensorboard -eb 150 -m UNetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/curvevel_b_train.txt -v ./train_test_splits/curvevel_b_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 1

# # FlatFault-A
# python -u train_inverse.py -ds flatfault-a -o ./SupervisedExperiment/FlatFault-A/UNetInverseModel_17M/ --tensorboard -eb 150 -m UNetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/flatfault_a_train.txt -v ./train_test_splits/flatfault_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 1

# # FlatFault-B
# python -u train_inverse.py -ds flatfault-b -o ./SupervisedExperiment/FlatFault-B/UNetInverseModel_17M/ --tensorboard -eb 150 -m UNetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/flatfault_b_train.txt -v ./train_test_splits/flatfault_b_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 1

# # CurveFault-A
# python -u train_inverse.py -ds curvefault-a -o ./SupervisedExperiment/CurveFault-A/UNetInverseModel_17M/ --tensorboard -eb 150 -m UNetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/curvefault_a_train.txt -v ./train_test_splits/curvefault_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 1

# # CurveFault-B
# python -u train_inverse.py -ds curvefault-b -o ./SupervisedExperiment/CurveFault-B/UNetInverseModel_17M/ --tensorboard -eb 150 -m UNetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/curvefault_b_train.txt -v ./train_test_splits/curvefault_b_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 1

# # Style-A
# python -u train_inverse.py -ds style-a -o ./SupervisedExperiment/Style-A/UNetInverseModel_17M/ --tensorboard -eb 150 -m UNetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/style_a_train.txt -v ./train_test_splits/style_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 1

# # Style-B
# python -u train_inverse.py -ds style-b -o ./SupervisedExperiment/Style-B/UNetInverseModel_17M/ --tensorboard -eb 150 -m UNetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/style_b_train.txt -v ./train_test_splits/style_b_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 1


####################################################
# # IUnetInverseModel
####################################################

# FlatVel-A
# python -u train_inverse.py -ds flatvel-a -o ./SupervisedExperiment/FlatVel-A/IUnetInverseModel/ --tensorboard -eb 150 -m IUnetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/flatvel_a_train.txt -v ./train_test_splits/flatvel_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adamax" --lr 0.002 

# # FlatVel-B
# python -u train_inverse.py -ds flatvel-b -o ./SupervisedExperiment/FlatVel-B/IUnetInverseModel/ --tensorboard -eb 150 -m IUnetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/flatvel_b_train.txt -v ./train_test_splits/flatvel_b_val.txt -ap "" --mask_factor 0.0 --optimizer "Adamax" --lr 0.002  

# # CurveVel-A
# python -u train_inverse.py -ds curvevel-a -o ./SupervisedExperiment/CurveVel-A/IUnetInverseModel/ --tensorboard -eb 150 -m IUnetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/curvevel_a_train.txt -v ./train_test_splits/curvevel_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adamax" --lr 0.002 

# # CurveVel-B
# python -u train_inverse.py -ds curvevel-b -o ./SupervisedExperiment/CurveVel-B/IUnetInverseModel/ --tensorboard -eb 150 -m IUnetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/curvevel_b_train.txt -v ./train_test_splits/curvevel_b_val.txt -ap "" --mask_factor 0.0 --optimizer "Adamax" --lr 0.002 

# # FlatFault-A
# python -u train_inverse.py -ds flatfault-a -o ./SupervisedExperiment/FlatFault-A/IUnetInverseModel/ --tensorboard -eb 150 -m IUnetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/flatfault_a_train.txt -v ./train_test_splits/flatfault_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adamax" --lr 0.002 

# # FlatFault-B
# python -u train_inverse.py -ds flatfault-b -o ./SupervisedExperiment/FlatFault-B/IUnetInverseModel/ --tensorboard -eb 150 -m IUnetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/flatfault_b_train.txt -v ./train_test_splits/flatfault_b_val.txt -ap "" --mask_factor 0.0 --optimizer "Adamax" --lr 0.002 

# # CurveFault-A
# python -u train_inverse.py -ds curvefault-a -o ./SupervisedExperiment/CurveFault-A/IUnetInverseModel/ --tensorboard -eb 150 -m IUnetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/curvefault_a_train.txt -v ./train_test_splits/curvefault_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adamax" --lr 0.002 

# # CurveFault-B
# python -u train_inverse.py -ds curvefault-b -o ./SupervisedExperiment/CurveFault-B/IUnetInverseModel/ --tensorboard -eb 150 -m IUnetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/curvefault_b_train.txt -v ./train_test_splits/curvefault_b_val.txt -ap "" --mask_factor 0.0 --optimizer "Adamax" --lr 0.002 

# # Style-A
# python -u train_inverse.py -ds style-a -o ./SupervisedExperiment/Style-A/IUnetInverseModel/ --tensorboard -eb 150 -m IUnetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/style_a_train.txt -v ./train_test_splits/style_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adamax" --lr 0.002 

# # Style-B
# python -u train_inverse.py -ds style-b -o ./SupervisedExperiment/Style-B/IUnetInverseModel/ --tensorboard -eb 150 -m IUnetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/style_b_train.txt -v ./train_test_splits/style_b_val.txt -ap "" --mask_factor 0.0 --optimizer "Adamax" --lr 0.002 


####################################################
# # Invertible X-Net with cycle loss (Warmup schedule)
####################################################

# # FlatVel-A
# python -u train_invertible_x_net.py -ds flatvel-a -o ./SupervisedExperiment/FlatVel-A/Invertible_XNet_cycle_warmup/ --optimizer "Adamax" --lr 0.002 --tensorboard -eb 150 -m IUNET --mask_factor 0.0 --lambda_amp 10.0 --lambda_vel 1.0 --lambda_vgg_vel 0.1 --lambda_vgg_amp 0.3 --lambda_cycle_vel 0.0 --lambda_cycle_amp 1.0 --warmup_cycle_epochs 100 -t ./train_test_splits/flatvel_a_train.txt -v ./train_test_splits/flatvel_a_val.txt -ap ""

# # FlatVel-B
# python -u train_invertible_x_net.py -ds flatvel-b -o ./SupervisedExperiment/FlatVel-B/Invertible_XNet_cycle_warmup/ --optimizer "Adamax" --lr 0.002 --tensorboard -eb 150 -m IUNET --mask_factor 0.0 --lambda_amp 10.0 --lambda_vel 1.0 --lambda_vgg_vel 0.1 --lambda_vgg_amp 0.3 --lambda_cycle_vel 0.0 --lambda_cycle_amp 1.0 --warmup_cycle_epochs 100 -t ./train_test_splits/flatvel_b_train.txt -v ./train_test_splits/flatvel_b_val.txt -ap ""

# # CurveVel-A
# python -u train_invertible_x_net.py -ds curvevel-a -o ./SupervisedExperiment/CurveVel-A/Invertible_XNet_cycle_warmup/ --optimizer "Adamax" --lr 0.002 --tensorboard -eb 150 -m IUNET --mask_factor 0.0 --lambda_amp 10.0 --lambda_vel 1.0 --lambda_vgg_vel 0.1 --lambda_vgg_amp 0.3 --lambda_cycle_vel 0.0 --lambda_cycle_amp 1.0 --warmup_cycle_epochs 100 -t ./train_test_splits/curvevel_a_train.txt -v ./train_test_splits/curvevel_a_val.txt -ap ""

# # CurveVel-B
# python -u train_invertible_x_net.py -ds curvevel-b -o ./SupervisedExperiment/CurveVel-B/Invertible_XNet_cycle_warmup/ --optimizer "Adamax" --lr 0.002 --tensorboard -eb 150 -m IUNET --mask_factor 0.0 --lambda_amp 10.0 --lambda_vel 1.0 --lambda_vgg_vel 0.1 --lambda_vgg_amp 0.3 --lambda_cycle_vel 0.0 --lambda_cycle_amp 1.0 --warmup_cycle_epochs 100 -t ./train_test_splits/curvevel_b_train.txt -v ./train_test_splits/curvevel_b_val.txt -ap ""

# # FlatFault-A
# python -u train_invertible_x_net.py -ds flatfault-a -o ./SupervisedExperiment/FlatFault-A/Invertible_XNet_cycle_warmup/ --optimizer "Adamax" --lr 0.002 --tensorboard -eb 150 -m IUNET --mask_factor 0.0 --lambda_amp 10.0 --lambda_vel 1.0 --lambda_vgg_vel 0.1 --lambda_vgg_amp 0.3 --lambda_cycle_vel 0.0 --lambda_cycle_amp 1.0 --warmup_cycle_epochs 100 -t ./train_test_splits/flatfault_a_train.txt -v ./train_test_splits/flatfault_a_val.txt -ap ""

# # FlatFault-B
# python -u train_invertible_x_net.py -ds flatfault-b -o ./SupervisedExperiment/FlatFault-B/Invertible_XNet_cycle_warmup/ --optimizer "Adamax" --lr 0.002 --tensorboard -eb 150 -m IUNET --mask_factor 0.0 --lambda_amp 10.0 --lambda_vel 1.0 --lambda_vgg_vel 0.1 --lambda_vgg_amp 0.3 --lambda_cycle_vel 0.0 --lambda_cycle_amp 1.0 --warmup_cycle_epochs 100 -t ./train_test_splits/flatfault_b_train.txt -v ./train_test_splits/flatfault_b_val.txt -ap ""

# # CurveFault-A
# python -u train_invertible_x_net.py -ds curvefault-a -o ./SupervisedExperiment/CurveFault-A/Invertible_XNet_cycle_warmup/ --optimizer "Adamax" --lr 0.002 --tensorboard -eb 150 -m IUNET --mask_factor 0.0 --lambda_amp 10.0 --lambda_vel 1.0 --lambda_vgg_vel 0.1 --lambda_vgg_amp 0.3 --lambda_cycle_vel 0.0 --lambda_cycle_amp 1.0 --warmup_cycle_epochs 100 -t ./train_test_splits/curvefault_a_train.txt -v ./train_test_splits/curvefault_a_val.txt -ap ""

# # CurveFault-B
# python -u train_invertible_x_net.py -ds curvefault-b -o ./SupervisedExperiment/CurveFault-B/Invertible_XNet_cycle_warmup/ --optimizer "Adamax" --lr 0.002 --tensorboard -eb 150 -m IUNET --mask_factor 0.0 --lambda_amp 10.0 --lambda_vel 1.0 --lambda_vgg_vel 0.1 --lambda_vgg_amp 0.3 --lambda_cycle_vel 0.0 --lambda_cycle_amp 1.0 --warmup_cycle_epochs 100 -t ./train_test_splits/curvefault_b_train.txt -v ./train_test_splits/curvefault_b_val.txt -ap ""

# # Style-A
# python -u train_invertible_x_net.py -ds style-a -o ./SupervisedExperiment/Style-A/Invertible_XNet_cycle_warmup/ --optimizer "Adamax" --lr 0.002 --tensorboard -eb 150 -m IUNET --mask_factor 0.0 --lambda_amp 10.0 --lambda_vel 1.0 --lambda_vgg_vel 0.1 --lambda_vgg_amp 0.3 --lambda_cycle_vel 0.0 --lambda_cycle_amp 1.0 --warmup_cycle_epochs 100 -t ./train_test_splits/style_a_train.txt -v ./train_test_splits/style_a_val.txt -ap ""

# # Style-B
# python -u train_invertible_x_net.py -ds style-b -o ./SupervisedExperiment/Style-B/Invertible_XNet_cycle_warmup/ --optimizer "Adamax" --lr 0.002 --tensorboard -eb 150 -m IUNET --mask_factor 0.0 --lambda_amp 10.0 --lambda_vel 1.0 --lambda_vgg_vel 0.1 --lambda_vgg_amp 0.3 --lambda_cycle_vel 0.0 --lambda_cycle_amp 1.0 --warmup_cycle_epochs 100 -t ./train_test_splits/style_b_train.txt -v ./train_test_splits/style_b_val.txt -ap ""

####################################################
# # Invertible X-Net without cycle loss
####################################################

# # FlatVel-A
# python -u train_invertible_x_net.py -ds flatvel-a -o ./SupervisedExperiment/FlatVel-A/Invertible_XNet/ --optimizer "Adamax" --lr 0.002 --tensorboard -eb 150 -m IUNET --mask_factor 0.0 --lambda_amp 10.0 --lambda_vel 1.0 --lambda_vgg_vel 0.1 --lambda_vgg_amp 0.3 --lambda_cycle_vel 0.0 --lambda_cycle_amp 0.0 -t ./train_test_splits/flatvel_a_train.txt -v ./train_test_splits/flatvel_a_val.txt -ap ""

# # FlatVel-B
# python -u train_invertible_x_net.py -ds flatvel-b -o ./SupervisedExperiment/FlatVel-B/Invertible_XNet/ --optimizer "Adamax" --lr 0.002 --tensorboard -eb 150 -m IUNET --mask_factor 0.0 --lambda_amp 10.0 --lambda_vel 1.0 --lambda_vgg_vel 0.1 --lambda_vgg_amp 0.3 --lambda_cycle_vel 0.0 --lambda_cycle_amp 0.0 -t ./train_test_splits/flatvel_b_train.txt -v ./train_test_splits/flatvel_b_val.txt -ap ""

# # CurveVel-A
# python -u train_invertible_x_net.py -ds curvevel-a -o ./SupervisedExperiment/CurveVel-A/Invertible_XNet/ --optimizer "Adamax" --lr 0.002 --tensorboard -eb 150 -m IUNET --mask_factor 0.0 --lambda_amp 10.0 --lambda_vel 1.0 --lambda_vgg_vel 0.1 --lambda_vgg_amp 0.3 --lambda_cycle_vel 0.0 --lambda_cycle_amp 0.0 -t ./train_test_splits/curvevel_a_train.txt -v ./train_test_splits/curvevel_a_val.txt -ap ""

# # CurveVel-B
# python -u train_invertible_x_net.py -ds curvevel-b -o ./SupervisedExperiment/CurveVel-B/Invertible_XNet/ --optimizer "Adamax" --lr 0.002 --tensorboard -eb 150 -m IUNET --mask_factor 0.0 --lambda_amp 10.0 --lambda_vel 1.0 --lambda_vgg_vel 0.1 --lambda_vgg_amp 0.3 --lambda_cycle_vel 0.0 --lambda_cycle_amp 0.0 -t ./train_test_splits/curvevel_b_train.txt -v ./train_test_splits/curvevel_b_val.txt -ap ""

# # FlatFault-A
# python -u train_invertible_x_net.py -ds flatfault-a -o ./SupervisedExperiment/FlatFault-A/Invertible_XNet/ --optimizer "Adamax" --lr 0.002 --tensorboard -eb 150 -m IUNET --mask_factor 0.0 --lambda_amp 10.0 --lambda_vel 1.0 --lambda_vgg_vel 0.1 --lambda_vgg_amp 0.3 --lambda_cycle_vel 0.0 --lambda_cycle_amp 0.0 -t ./train_test_splits/flatfault_a_train.txt -v ./train_test_splits/flatfault_a_val.txt -ap ""

# # FlatFault-B
# python -u train_invertible_x_net.py -ds flatfault-b -o ./SupervisedExperiment/FlatFault-B/Invertible_XNet/ --optimizer "Adamax" --lr 0.002 --tensorboard -eb 150 -m IUNET --mask_factor 0.0 --lambda_amp 10.0 --lambda_vel 1.0 --lambda_vgg_vel 0.1 --lambda_vgg_amp 0.3 --lambda_cycle_vel 0.0 --lambda_cycle_amp 0.0 -t ./train_test_splits/flatfault_b_train.txt -v ./train_test_splits/flatfault_b_val.txt -ap ""

# # CurveFault-A
# python -u train_invertible_x_net.py -ds curvefault-a -o ./SupervisedExperiment/CurveFault-A/Invertible_XNet/ --optimizer "Adamax" --lr 0.002 --tensorboard -eb 150 -m IUNET --mask_factor 0.0 --lambda_amp 10.0 --lambda_vel 1.0 --lambda_vgg_vel 0.1 --lambda_vgg_amp 0.3 --lambda_cycle_vel 0.0 --lambda_cycle_amp 0.0 -t ./train_test_splits/curvefault_a_train.txt -v ./train_test_splits/curvefault_a_val.txt -ap ""

# # CurveFault-B
# python -u train_invertible_x_net.py -ds curvefault-b -o ./SupervisedExperiment/CurveFault-B/Invertible_XNet/ --optimizer "Adamax" --lr 0.002 --tensorboard -eb 150 -m IUNET --mask_factor 0.0 --lambda_amp 10.0 --lambda_vel 1.0 --lambda_vgg_vel 0.1 --lambda_vgg_amp 0.3 --lambda_cycle_vel 0.0 --lambda_cycle_amp 0.0 -t ./train_test_splits/curvefault_b_train.txt -v ./train_test_splits/curvefault_b_val.txt -ap ""

# # Style-A
# python -u train_invertible_x_net.py -ds style-a -o ./SupervisedExperiment/Style-A/Invertible_XNet/ --optimizer "Adamax" --lr 0.002 --tensorboard -eb 150 -m IUNET --mask_factor 0.0 --lambda_amp 10.0 --lambda_vel 1.0 --lambda_vgg_vel 0.1 --lambda_vgg_amp 0.3 --lambda_cycle_vel 0.0 --lambda_cycle_amp 0.0 -t ./train_test_splits/style_a_train.txt -v ./train_test_splits/style_a_val.txt -ap ""

# # Style-B
# python -u train_invertible_x_net.py -ds style-b -o ./SupervisedExperiment/Style-B/Invertible_XNet/ --optimizer "Adamax" --lr 0.002 --tensorboard -eb 150 -m IUNET --mask_factor 0.0 --lambda_amp 10.0 --lambda_vel 1.0 --lambda_vgg_vel 0.1 --lambda_vgg_amp 0.3 --lambda_cycle_vel 0.0 --lambda_cycle_amp 0.0 -t ./train_test_splits/style_b_train.txt -v ./train_test_splits/style_b_val.txt -ap ""

####################################################
# # InversionNet
####################################################

# FlatVel-A
# python -u train_inverse.py -ds flatvel-a -o ./SupervisedExperiment/FlatVel-A/InversionNet/ --tensorboard -eb 150 -m InversionNet --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/flatvel_a_train.txt -v ./train_test_splits/flatvel_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" 

# # FlatVel-B
# python -u train_inverse.py -ds flatvel-b -o ./SupervisedExperiment/FlatVel-B/InversionNet/ --tensorboard -eb 150 -m InversionNet --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/flatvel_b_train.txt -v ./train_test_splits/flatvel_b_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" 

# # CurveVel-A
# python -u train_inverse.py -ds curvevel-a -o ./SupervisedExperiment/CurveVel-A/InversionNet/ --tensorboard -eb 150 -m InversionNet --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/curvevel_a_train.txt -v ./train_test_splits/curvevel_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam"  

# # CurveVel-B
# python -u train_inverse.py -ds curvevel-b -o ./SupervisedExperiment/CurveVel-B/InversionNet/ --tensorboard -eb 150 -m InversionNet --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/curvevel_b_train.txt -v ./train_test_splits/curvevel_b_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" 

# # FlatFault-A
# python -u train_inverse.py -ds flatfault-a -o ./SupervisedExperiment/FlatFault-A/InversionNet/ --tensorboard -eb 150 -m InversionNet --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/flatfault_a_train.txt -v ./train_test_splits/flatfault_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" 

# # FlatFault-B
# python -u train_inverse.py -ds flatfault-b -o ./SupervisedExperiment/FlatFault-B/InversionNet/ --tensorboard -eb 150 -m InversionNet --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/flatfault_b_train.txt -v ./train_test_splits/flatfault_b_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" 

# # CurveFault-A
# python -u train_inverse.py -ds curvefault-a -o ./SupervisedExperiment/CurveFault-A/InversionNet/ --tensorboard -eb 150 -m InversionNet --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/curvefault_a_train.txt -v ./train_test_splits/curvefault_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" 

# # CurveFault-B
# python -u train_inverse.py -ds curvefault-b -o ./SupervisedExperiment/CurveFault-B/InversionNet/ --tensorboard -eb 150 -m InversionNet --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/curvefault_b_train.txt -v ./train_test_splits/curvefault_b_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam"  

# # Style-A
# python -u train_inverse.py -ds style-a -o ./SupervisedExperiment/Style-A/InversionNet/ --tensorboard -eb 150 -m InversionNet --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/style_a_train.txt -v ./train_test_splits/style_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" 

# # Style-B
# python -u train_inverse.py -ds style-b -o ./SupervisedExperiment/Style-B/InversionNet/ --tensorboard -eb 150 -m InversionNet --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/style_b_train.txt -v ./train_test_splits/style_b_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" 

exit;