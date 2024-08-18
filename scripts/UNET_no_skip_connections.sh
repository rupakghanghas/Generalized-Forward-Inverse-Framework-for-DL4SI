#!/bin/bash

#SBATCH --cpus-per-task=16 # this requests 1 node, 16 core. 
#SBATCH --time=3-00:00:00 
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

# #FlatVel-A
# python -u train_inverse.py -ds flatvel-a -o ./SupervisedExperiment/FlatVel-A/UNetInverseModel_33M_NoSkip/ --tensorboard -eb 150 -m UNetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/flatvel_a_train.txt -v ./train_test_splits/flatvel_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 2 --skip 0 

# # FlatVel-B
# python -u train_inverse.py -ds flatvel-b -o ./SupervisedExperiment/FlatVel-B/UNetInverseModel_33M_NoSkip/ --tensorboard -eb 150 -m UNetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/flatvel_b_train.txt -v ./train_test_splits/flatvel_b_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 2 --skip 0

# # CurveVel-A
# python -u train_inverse.py -ds curvevel-a -o ./SupervisedExperiment/CurveVel-A/UNetInverseModel_33M_NoSkip/ --tensorboard -eb 150 -m UNetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/curvevel_a_train.txt -v ./train_test_splits/curvevel_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 2 --skip 0

# # CurveVel-B
# python -u train_inverse.py -ds curvevel-b -o ./SupervisedExperiment/CurveVel-B/UNetInverseModel_33M_NoSkip/ --tensorboard -eb 150 -m UNetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/curvevel_b_train.txt -v ./train_test_splits/curvevel_b_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 2 --skip 0

# # FlatFault-A
# python -u train_inverse.py -ds flatfault-a -o ./SupervisedExperiment/FlatFault-A/UNetInverseModel_33M_NoSkip/ --tensorboard -eb 150 -m UNetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/flatfault_a_train.txt -v ./train_test_splits/flatfault_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 2 --skip 0

# # FlatFault-B
# python -u train_inverse.py -ds flatfault-b -o ./SupervisedExperiment/FlatFault-B/UNetInverseModel_33M_NoSkip/ --tensorboard -eb 150 -m UNetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/flatfault_b_train.txt -v ./train_test_splits/flatfault_b_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 2 --skip 0

# # CurveFault-A
# python -u train_inverse.py -ds curvefault-a -o ./SupervisedExperiment/CurveFault-A/UNetInverseModel_33M_NoSkip/ --tensorboard -eb 150 -m UNetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/curvefault_a_train.txt -v ./train_test_splits/curvefault_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 2 --skip 0

# # CurveFault-B
# python -u train_inverse.py -ds curvefault-b -o ./SupervisedExperiment/CurveFault-B/UNetInverseModel_33M_NoSkip/ --tensorboard -eb 150 -m UNetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/curvefault_b_train.txt -v ./train_test_splits/curvefault_b_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 2 --skip 0

# # Style-A
# python -u train_inverse.py -ds style-a -o ./SupervisedExperiment/Style-A/UNetInverseModel_33M_NoSkip/ --tensorboard -eb 150 -m UNetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/style_a_train.txt -v ./train_test_splits/style_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 2 --skip 0

# # Style-B
# python -u train_inverse.py -ds style-b -o ./SupervisedExperiment/Style-B/UNetInverseModel_33M_NoSkip/ --tensorboard -eb 150 -m UNetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/style_b_train.txt -v ./train_test_splits/style_b_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 2 --skip 0


####################################################
# # UNetInverseModel - Small (17.86M) depth=2, repeat=1
####################################################

# # FlatVel-A
# python -u train_inverse.py -ds flatvel-a -o ./SupervisedExperiment/FlatVel-A/UNetInverseModel_17M_NoSkip/ --tensorboard -eb 150 -m UNetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/flatvel_a_train.txt -v ./train_test_splits/flatvel_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 1 --skip 0

# # FlatVel-B
# python -u train_inverse.py -ds flatvel-b -o ./SupervisedExperiment/FlatVel-B/UNetInverseModel_17M_NoSkip/ --tensorboard -eb 150 -m UNetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/flatvel_b_train.txt -v ./train_test_splits/flatvel_b_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 1 --skip 0

# # CurveVel-A
# python -u train_inverse.py -ds curvevel-a -o ./SupervisedExperiment/CurveVel-A/UNetInverseModel_17M_NoSkip/ --tensorboard -eb 150 -m UNetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/curvevel_a_train.txt -v ./train_test_splits/curvevel_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 1 --skip 0

# # CurveVel-B
# python -u train_inverse.py -ds curvevel-b -o ./SupervisedExperiment/CurveVel-B/UNetInverseModel_17M_NoSkip/ --tensorboard -eb 150 -m UNetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/curvevel_b_train.txt -v ./train_test_splits/curvevel_b_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 1 --skip 0

# # FlatFault-A
# python -u train_inverse.py -ds flatfault-a -o ./SupervisedExperiment/FlatFault-A/UNetInverseModel_17M_NoSkip/ --tensorboard -eb 150 -m UNetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/flatfault_a_train.txt -v ./train_test_splits/flatfault_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 1 --skip 0

# # FlatFault-B
# python -u train_inverse.py -ds flatfault-b -o ./SupervisedExperiment/FlatFault-B/UNetInverseModel_17M_NoSkip/ --tensorboard -eb 150 -m UNetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/flatfault_b_train.txt -v ./train_test_splits/flatfault_b_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 1 --skip 0

# # CurveFault-A
# python -u train_inverse.py -ds curvefault-a -o ./SupervisedExperiment/CurveFault-A/UNetInverseModel_17M_NoSkip/ --tensorboard -eb 150 -m UNetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/curvefault_a_train.txt -v ./train_test_splits/curvefault_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 1 --skip 0

# # CurveFault-B
# python -u train_inverse.py -ds curvefault-b -o ./SupervisedExperiment/CurveFault-B/UNetInverseModel_17M_NoSkip/ --tensorboard -eb 150 -m UNetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/curvefault_b_train.txt -v ./train_test_splits/curvefault_b_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 1 --skip 0

# # Style-A
# python -u train_inverse.py -ds style-a -o ./SupervisedExperiment/Style-A/UNetInverseModel_17M_NoSkip/ --tensorboard -eb 150 -m UNetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/style_a_train.txt -v ./train_test_splits/style_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 1 --skip 0

# # Style-B
# python -u train_inverse.py -ds style-b -o ./SupervisedExperiment/Style-B/UNetInverseModel_17M_NoSkip/ --tensorboard -eb 150 -m UNetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/style_b_train.txt -v ./train_test_splits/style_b_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 1 --skip 0



exit;