#!/bin/bash

#SBATCH --cpus-per-task=16 # this requests 1 node, 16 core. 
#SBATCH --time=5-00:00:00 
#SBATCH --gres=gpu:1
#SBATCH --partition=dgx_normal_q


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
ulimit -u 65536

pwd

module spider Anaconda3/2020.11
# module conda Anaconda3/2020.11
# module init
conda init
source .bashrc

pwd
source activate venv


###############################################################################################################
############# DATASET : CURVEVEL-A
###############################################################################################################

########################
# Without Cycle Loss
########################

# # X-Net / without Cycle Loss / 0 % masking factor or 100% training data 
python -u train_invertible_x_net.py -ds curvevel-a -o ./Invertible_X_Net/CurveVel-A/Cycle_Loss_0_Mask_Factor_0/lambda_amp_10_lambda_vel_1/ --tensorboard -eb 150 -m IUNET --mask_factor 0.0 --lambda_amp 10.0 --lambda_vel 1.0 --lambda_vgg_vel 0.1 --lambda_vgg_amp 0.3 --lambda_cycle_vel 0.0 --lambda_cycle_amp 0.0 -t ./train_test_splits/curvevel_a_train_48.txt -v ./train_test_splits/curvevel_a_val_48.txt -ap ""

# # X-Net / without Cycle Loss / 20 % masking factor 
# python -u train_invertible_x_net.py -ds curvevel-a -o ./Invertible_X_Net/CurveVel-A/Cycle_Loss_0_Mask_Factor_20/lambda_amp_10_lambda_vel_1/ --tensorboard -eb 150 -m IUNET --mask_factor 0.2 --lambda_amp 10.0 --lambda_vel 1.0 --lambda_vgg_vel 0.1 --lambda_vgg_amp 0.3 --lambda_cycle_vel 0.0 --lambda_cycle_amp 0.0 -t ./train_test_splits/curvevel_a_train_48.txt -v ./train_test_splits/curvevel_a_val_48.txt -ap ""

# # X-Net / without Cycle Loss / 40 % masking factor 
# python -u train_invertible_x_net.py -ds curvevel-a -o ./Invertible_X_Net/CurveVel-A/Cycle_Loss_0_Mask_Factor_40/lambda_amp_10_lambda_vel_1/ --tensorboard -eb 150 -m IUNET --mask_factor 0.4 --lambda_amp 10.0 --lambda_vel 1.0 --lambda_vgg_vel 0.1 --lambda_vgg_amp 0.3 --lambda_cycle_vel 0.0 --lambda_cycle_amp 0.0 -t ./train_test_splits/curvevel_a_train_48.txt -v ./train_test_splits/curvevel_a_val_48.txt -ap ""

# # X-Net / without Cycle Loss / 60 % masking factor 
# python -u train_invertible_x_net.py -ds curvevel-a -o ./Invertible_X_Net/CurveVel-A/Cycle_Loss_0_Mask_Factor_60/lambda_amp_10_lambda_vel_1/ --tensorboard -eb 150 -m IUNET --mask_factor 0.6 --lambda_amp 10.0 --lambda_vel 1.0 --lambda_vgg_vel 0.1 --lambda_vgg_amp 0.3 --lambda_cycle_vel 0.0 --lambda_cycle_amp 0.0 -t ./train_test_splits/curvevel_a_train_48.txt -v ./train_test_splits/curvevel_a_val_48.txt -ap ""

# # X-Net / without Cycle Loss / 80 % masking factor 
# python -u train_invertible_x_net.py -ds curvevel-a -o ./Invertible_X_Net/CurveVel-A/Cycle_Loss_0_Mask_Factor_80/lambda_amp_10_lambda_vel_1/ --tensorboard -eb 150 -m IUNET --mask_factor 0.8 --lambda_amp 10.0 --lambda_vel 1.0 --lambda_vgg_vel 0.1 --lambda_vgg_amp 0.3 --lambda_cycle_vel 0.0 --lambda_cycle_amp 0.0 -t ./train_test_splits/curvevel_a_train_48.txt -v ./train_test_splits/curvevel_a_val_48.txt -ap ""


########################
# Cycle Loss
########################

# # X-Net / without Cycle Loss / 0 % masking factor or 100% training data 
# python -u train_invertible_x_net.py -ds curvevel-a -o ./Debug/Invertible_X_Net/CurveVel-A/Cycle_Loss_1_Mask_Factor_0/lambda_amp_10_lambda_vel_1/ --tensorboard -eb 150 -m IUNET --mask_factor 0.0 --lambda_amp 10.0 --lambda_vel 1.0 --lambda_vgg_vel 0.1 --lambda_vgg_amp 0.3 --lambda_cycle_vel 0.0 --lambda_cycle_amp 1.0 -t ./train_test_splits/curvevel_a_train_48.txt -v ./train_test_splits/curvevel_a_val_48.txt -ap ""

# # X-Net / without Cycle Loss / 20 % masking factor 
# python -u train_invertible_x_net.py -ds curvevel-a -o ./Invertible_X_Net/CurveVel-A/Cycle_Loss_1_Mask_Factor_20/lambda_amp_10_lambda_vel_1/ --tensorboard -eb 150 -m IUNET --mask_factor 0.2 --lambda_amp 10.0 --lambda_vel 1.0 --lambda_vgg_vel 0.1 --lambda_vgg_amp 0.3 --lambda_cycle_vel 0.0 --lambda_cycle_amp 1.0 -t ./train_test_splits/curvevel_a_train_48.txt -v ./train_test_splits/curvevel_a_val_48.txt -ap ""

# # X-Net / without Cycle Loss / 40 % masking factor 
# python -u train_invertible_x_net.py -ds curvevel-a -o ./Invertible_X_Net/CurveVel-A/Cycle_Loss_1_Mask_Factor_40/lambda_amp_10_lambda_vel_1/ --tensorboard -eb 150 -m IUNET --mask_factor 0.4 --lambda_amp 10.0 --lambda_vel 1.0 --lambda_vgg_vel 0.1 --lambda_vgg_amp 0.3 --lambda_cycle_vel 0.0 --lambda_cycle_amp 1.0 -t ./train_test_splits/curvevel_a_train_48.txt -v ./train_test_splits/curvevel_a_val_48.txt -ap ""

# # X-Net / without Cycle Loss / 60 % masking factor 
# python -u train_invertible_x_net.py -ds curvevel-a -o ./Invertible_X_Net/CurveVel-A/Cycle_Loss_1_Mask_Factor_60/lambda_amp_10_lambda_vel_1/ --tensorboard -eb 150 -m IUNET --mask_factor 0.6 --lambda_amp 10.0 --lambda_vel 1.0 --lambda_vgg_vel 0.1 --lambda_vgg_amp 0.3 --lambda_cycle_vel 0.0 --lambda_cycle_amp 1.0 -t ./train_test_splits/curvevel_a_train_48.txt -v ./train_test_splits/curvevel_a_val_48.txt -ap ""

# # X-Net / without Cycle Loss / 80 % masking factor 
# python -u train_invertible_x_net.py -ds curvevel-a -o ./Invertible_X_Net/CurveVel-A/Cycle_Loss_1_Mask_Factor_80/lambda_amp_10_lambda_vel_1/ --tensorboard -eb 150 -m IUNET --mask_factor 0.8 --lambda_amp 10.0 --lambda_vel 1.0 --lambda_vgg_vel 0.1 --lambda_vgg_amp 0.3 --lambda_cycle_vel 0.0 --lambda_cycle_amp 1.0 -t ./train_test_splits/curvevel_a_train_48.txt -v ./train_test_splits/curvevel_a_val_48.txt -ap ""

###############################################################################################################
############# DATASET : FLATVEL-A
###############################################################################################################

########################
# Without Cycle Loss
########################

# # X-Net / without Cycle Loss / 0 % masking factor or 100% training data 
# python -u train_invertible_x_net.py -ds flatvel-a -o ./Invertible_X_Net/FlatVel-A/Cycle_Loss_0_Mask_Factor_0/lambda_amp_10_lambda_vel_1/ --tensorboard -eb 150 -m IUNET --mask_factor 0.0 --lambda_amp 10.0 --lambda_vel 1.0 --lambda_vgg_vel 0.1 --lambda_vgg_amp 0.3 --lambda_cycle_vel 0.0 --lambda_cycle_amp 0.0 -t ./train_test_splits/flatvel_a_train_48.txt -v ./train_test_splits/flatvel_a_val_48.txt -ap ""

# # X-Net / without Cycle Loss / 20 % masking factor 
# python -u train_invertible_x_net.py -ds flatvel-a -o ./Invertible_X_Net/FlatVel-A/Cycle_Loss_0_Mask_Factor_20/lambda_amp_10_lambda_vel_1/ --tensorboard -eb 150 -m IUNET --mask_factor 0.2 --lambda_amp 10.0 --lambda_vel 1.0 --lambda_vgg_vel 0.1 --lambda_vgg_amp 0.3 --lambda_cycle_vel 0.0 --lambda_cycle_amp 0.0 -t ./train_test_splits/flatvel_a_train_48.txt -v ./train_test_splits/flatvel_a_val_48.txt -ap ""

# # X-Net / without Cycle Loss / 40 % masking factor 
# python -u train_invertible_x_net.py -ds flatvel-a -o ./Invertible_X_Net/FlatVel-A/Cycle_Loss_0_Mask_Factor_40/lambda_amp_10_lambda_vel_1/ --tensorboard -eb 150 -m IUNET --mask_factor 0.4 --lambda_amp 10.0 --lambda_vel 1.0 --lambda_vgg_vel 0.1 --lambda_vgg_amp 0.3 --lambda_cycle_vel 0.0 --lambda_cycle_amp 0.0 -t ./train_test_splits/flatvel_a_train_48.txt -v ./train_test_splits/flatvel_a_val_48.txt -ap ""

# # X-Net / without Cycle Loss / 60 % masking factor 
# python -u train_invertible_x_net.py -ds flatvel-a -o ./Invertible_X_Net/FlatVel-A/Cycle_Loss_0_Mask_Factor_60/lambda_amp_10_lambda_vel_1/ --tensorboard -eb 150 -m IUNET --mask_factor 0.6 --lambda_amp 10.0 --lambda_vel 1.0 --lambda_vgg_vel 0.1 --lambda_vgg_amp 0.3 --lambda_cycle_vel 0.0 --lambda_cycle_amp 0.0 -t ./train_test_splits/flatvel_a_train_48.txt -v ./train_test_splits/flatvel_a_val_48.txt -ap ""

# # X-Net / without Cycle Loss / 80 % masking factor 
# python -u train_invertible_x_net.py -ds flatvel-a -o ./Invertible_X_Net/FlatVel-A/Cycle_Loss_0_Mask_Factor_80/lambda_amp_10_lambda_vel_1/ --tensorboard -eb 150 -m IUNET --mask_factor 0.8 --lambda_amp 10.0 --lambda_vel 1.0 --lambda_vgg_vel 0.1 --lambda_vgg_amp 0.3 --lambda_cycle_vel 0.0 --lambda_cycle_amp 0.0 -t ./train_test_splits/flatvel_a_train_48.txt -v ./train_test_splits/flatvel_a_val_48.txt -ap ""


########################
# Cycle Loss
########################

# # X-Net / without Cycle Loss / 0 % masking factor or 100% training data 
# python -u train_invertible_x_net.py -ds flatvel-a -o ./Invertible_X_Net/FlatVel-A/Cycle_Loss_1_Mask_Factor_0/lambda_amp_10_lambda_vel_1/ --tensorboard -eb 150 -m IUNET --mask_factor 0.0 --lambda_amp 10.0 --lambda_vel 1.0 --lambda_vgg_vel 0.1 --lambda_vgg_amp 0.3 --lambda_cycle_vel 0.0 --lambda_cycle_amp 1.0 -t ./train_test_splits/flatvel_a_train_48.txt -v ./train_test_splits/flatvel_a_val_48.txt -ap ""

# # X-Net / without Cycle Loss / 20 % masking factor 
# python -u train_invertible_x_net.py -ds flatvel-a -o ./Invertible_X_Net/FlatVel-A/Cycle_Loss_1_Mask_Factor_20/lambda_amp_10_lambda_vel_1/ --tensorboard -eb 150 -m IUNET --mask_factor 0.2 --lambda_amp 10.0 --lambda_vel 1.0 --lambda_vgg_vel 0.1 --lambda_vgg_amp 0.3 --lambda_cycle_vel 0.0 --lambda_cycle_amp 1.0 -t ./train_test_splits/flatvel_a_train_48.txt -v ./train_test_splits/flatvel_a_val_48.txt -ap ""

# # X-Net / without Cycle Loss / 40 % masking factor 
# python -u train_invertible_x_net.py -ds flatvel-a -o ./Invertible_X_Net/FlatVel-A/Cycle_Loss_1_Mask_Factor_40/lambda_amp_10_lambda_vel_1/ --tensorboard -eb 150 -m IUNET --mask_factor 0.4 --lambda_amp 10.0 --lambda_vel 1.0 --lambda_vgg_vel 0.1 --lambda_vgg_amp 0.3 --lambda_cycle_vel 0.0 --lambda_cycle_amp 1.0 -t ./train_test_splits/flatvel_a_train_48.txt -v ./train_test_splits/flatvel_a_val_48.txt -ap ""

# # X-Net / without Cycle Loss / 60 % masking factor 
# python -u train_invertible_x_net.py -ds flatvel-a -o ./Invertible_X_Net/FlatVel-A/Cycle_Loss_1_Mask_Factor_60/lambda_amp_10_lambda_vel_1/ --tensorboard -eb 150 -m IUNET --mask_factor 0.6 --lambda_amp 10.0 --lambda_vel 1.0 --lambda_vgg_vel 0.1 --lambda_vgg_amp 0.3 --lambda_cycle_vel 0.0 --lambda_cycle_amp 1.0 -t ./train_test_splits/flatvel_a_train_48.txt -v ./train_test_splits/flatvel_a_val_48.txt -ap ""

# # X-Net / without Cycle Loss / 80 % masking factor 
# python -u train_invertible_x_net.py -ds flatvel-a -o ./Invertible_X_Net/FlatVel-A/Cycle_Loss_1_Mask_Factor_80/lambda_amp_10_lambda_vel_1/ --tensorboard -eb 150 -m IUNET --mask_factor 0.8 --lambda_amp 10.0 --lambda_vel 1.0 --lambda_vgg_vel 0.1 --lambda_vgg_amp 0.3 --lambda_cycle_vel 0.0 --lambda_cycle_amp 1.0 -t ./train_test_splits/flatvel_a_train_48.txt -v ./train_test_splits/flatvel_a_val_48.txt -ap ""

exit;
