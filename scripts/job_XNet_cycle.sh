#!/bin/bash

#SBATCH --cpus-per-task=16 # this requests 1 node, 16 core. 
#SBATCH --time=5-00:00:00 
#SBATCH --gres=gpu:1
#SBATCH --partition=a100_normal_q


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
ulimit -u 65536

module spider Anaconda3/2020.11
conda init
source ~/.bashrc

source activate venv

which python
python -V

###############################################################################################################
############# DATASET : CURVEVEL-A
###############################################################################################################

##############################
# With Cycle Loss
##############################

# Mask Factor = 90
# python -u train_invertible_x_net.py -ds curvevel-a -o ./RESULTS_Invertible_XNet/CurveVel-A/Joint/Cycle_Loss_1_Mask_Factor_90/lambda_amp_10_lambda_vel_1/ --optimizer "Adamax" --lr 0.002 --tensorboard -eb 150 -m IUNet --mask_factor 0.9 --lambda_amp 10.0 --lambda_vel 1.0 --lambda_vgg_vel 0.1 --lambda_vgg_amp 0.3 --lambda_cycle_vel 0.0 --lambda_cycle_amp 1.0 -t ./train_test_splits/curvevel_a_train_48.txt -v ./train_test_splits/curvevel_a_val_48.txt -ap ""

# Mask Factor = 80
# python -u train_invertible_x_net.py -ds curvevel-a -o ./RESULTS_Invertible_XNet/CurveVel-A/Joint/Cycle_Loss_1_Mask_Factor_80/lambda_amp_10_lambda_vel_1/ --optimizer "Adamax" --lr 0.002 --tensorboard -eb 150 -m IUNet --mask_factor 0.8 --lambda_amp 10.0 --lambda_vel 1.0 --lambda_vgg_vel 0.1 --lambda_vgg_amp 0.3 --lambda_cycle_vel 0.0 --lambda_cycle_amp 1.0 -t ./train_test_splits/curvevel_a_train_48.txt -v ./train_test_splits/curvevel_a_val_48.txt -ap ""

# Mask Factor = 60
# python -u train_invertible_x_net.py -ds curvevel-a -o ./RESULTS_Invertible_XNet/CurveVel-A/Joint/Cycle_Loss_1_Mask_Factor_60/lambda_amp_10_lambda_vel_1/ --optimizer "Adamax" --lr 0.002 --tensorboard -eb 150 -m IUNet --mask_factor 0.6 --lambda_amp 10.0 --lambda_vel 1.0 --lambda_vgg_vel 0.1 --lambda_vgg_amp 0.3 --lambda_cycle_vel 0.0 --lambda_cycle_amp 1.0 -t ./train_test_splits/curvevel_a_train_48.txt -v ./train_test_splits/curvevel_a_val_48.txt -ap ""

# Mask Factor = 40
# python -u train_invertible_x_net.py -ds curvevel-a -o ./RESULTS_Invertible_XNet/CurveVel-A/Joint/Cycle_Loss_1_Mask_Factor_40/lambda_amp_10_lambda_vel_1/ --optimizer "Adamax" --lr 0.002 --tensorboard -eb 150 -m IUNet --mask_factor 0.4 --lambda_amp 10.0 --lambda_vel 1.0 --lambda_vgg_vel 0.1 --lambda_vgg_amp 0.3 --lambda_cycle_vel 0.0 --lambda_cycle_amp 1.0 -t ./train_test_splits/curvevel_a_train_48.txt -v ./train_test_splits/curvevel_a_val_48.txt -ap ""

# Mask Factor = 20
# python -u train_invertible_x_net.py -ds curvevel-a -o ./RESULTS_Invertible_XNet/CurveVel-A/Joint/Cycle_Loss_1_Mask_Factor_20/lambda_amp_10_lambda_vel_1/ --optimizer "Adamax" --lr 0.002 --tensorboard -eb 150 -m IUNet --mask_factor 0.2 --lambda_amp 10.0 --lambda_vel 1.0 --lambda_vgg_vel 0.1 --lambda_vgg_amp 0.3 --lambda_cycle_vel 0.0 --lambda_cycle_amp 1.0 -t ./train_test_splits/curvevel_a_train_48.txt -v ./train_test_splits/curvevel_a_val_48.txt -ap ""

# Mask Factor = 0
# python -u train_invertible_x_net.py -ds curvevel-a -o ./RESULTS_Invertible_XNet/CurveVel-A/Joint/Cycle_Loss_1_Mask_Factor_0/lambda_amp_10_lambda_vel_1/ --optimizer "Adamax" --lr 0.002 --tensorboard -eb 150 -m IUNet --mask_factor 0.0 --lambda_amp 10.0 --lambda_vel 1.0 --lambda_vgg_vel 0.1 --lambda_vgg_amp 0.3 --lambda_cycle_vel 0.0 --lambda_cycle_amp 1.0 -t ./train_test_splits/curvevel_a_train_48.txt -v ./train_test_splits/curvevel_a_val_48.txt -ap ""


##############################
# Without Cycle Loss
##############################

# Mask Factor = 90
# python -u train_invertible_x_net.py -ds curvevel-a -o ./RESULTS_Invertible_XNet/CurveVel-A/Joint/Cycle_Loss_0_Mask_Factor_90/lambda_amp_10_lambda_vel_1/ --optimizer "Adamax" --lr 0.002 --tensorboard -eb 150 -m IUNet --mask_factor 0.9 --lambda_amp 10.0 --lambda_vel 1.0 --lambda_vgg_vel 0.1 --lambda_vgg_amp 0.3 --lambda_cycle_vel 0.0 --lambda_cycle_amp 0.0 -t ./train_test_splits/curvevel_a_train_48.txt -v ./train_test_splits/curvevel_a_val_48.txt -ap ""

# Mask Factor = 80
# python -u train_invertible_x_net.py -ds curvevel-a -o ./RESULTS_Invertible_XNet/CurveVel-A/Joint/Cycle_Loss_0_Mask_Factor_80/lambda_amp_10_lambda_vel_1/ --optimizer "Adamax" --lr 0.002 --tensorboard -eb 150 -m IUNet --mask_factor 0.8 --lambda_amp 10.0 --lambda_vel 1.0 --lambda_vgg_vel 0.1 --lambda_vgg_amp 0.3 --lambda_cycle_vel 0.0 --lambda_cycle_amp 0.0 -t ./train_test_splits/curvevel_a_train_48.txt -v ./train_test_splits/curvevel_a_val_48.txt -ap ""

# Mask Factor = 60
# python -u train_invertible_x_net.py -ds curvevel-a -o ./RESULTS_Invertible_XNet/CurveVel-A/Joint/Cycle_Loss_0_Mask_Factor_60/lambda_amp_10_lambda_vel_1/ --optimizer "Adamax" --lr 0.002 --tensorboard -eb 150 -m IUNet --mask_factor 0.6 --lambda_amp 10.0 --lambda_vel 1.0 --lambda_vgg_vel 0.1 --lambda_vgg_amp 0.3 --lambda_cycle_vel 0.0 --lambda_cycle_amp 0.0 -t ./train_test_splits/curvevel_a_train_48.txt -v ./train_test_splits/curvevel_a_val_48.txt -ap ""

# Mask Factor = 40
# python -u train_invertible_x_net.py -ds curvevel-a -o ./RESULTS_Invertible_XNet/CurveVel-A/Joint/Cycle_Loss_0_Mask_Factor_40/lambda_amp_10_lambda_vel_1/ --optimizer "Adamax" --lr 0.002 --tensorboard -eb 150 -m IUNet --mask_factor 0.4 --lambda_amp 10.0 --lambda_vel 1.0 --lambda_vgg_vel 0.1 --lambda_vgg_amp 0.3 --lambda_cycle_vel 0.0 --lambda_cycle_amp 0.0 -t ./train_test_splits/curvevel_a_train_48.txt -v ./train_test_splits/curvevel_a_val_48.txt -ap ""

# Mask Factor = 20
# python -u train_invertible_x_net.py -ds curvevel-a -o ./RESULTS_Invertible_XNet/CurveVel-A/Joint/Cycle_Loss_0_Mask_Factor_20/lambda_amp_10_lambda_vel_1/ --optimizer "Adamax" --lr 0.002 --tensorboard -eb 150 -m IUNet --mask_factor 0.2 --lambda_amp 10.0 --lambda_vel 1.0 --lambda_vgg_vel 0.1 --lambda_vgg_amp 0.3 --lambda_cycle_vel 0.0 --lambda_cycle_amp 0.0 -t ./train_test_splits/curvevel_a_train_48.txt -v ./train_test_splits/curvevel_a_val_48.txt -ap ""

# Mask Factor = 0
# python -u train_invertible_x_net.py -ds curvevel-a -o ./RESULTS_Invertible_XNet/CurveVel-A/Joint/Cycle_Loss_0_Mask_Factor_0/lambda_amp_10_lambda_vel_1/ --optimizer "Adamax" --lr 0.002 --tensorboard -eb 150 -m IUNet --mask_factor 0.0 --lambda_amp 10.0 --lambda_vel 1.0 --lambda_vgg_vel 0.1 --lambda_vgg_amp 0.3 --lambda_cycle_vel 0.0 --lambda_cycle_amp 0.0 -t ./train_test_splits/curvevel_a_train_48.txt -v ./train_test_splits/curvevel_a_val_48.txt -ap ""

exit;