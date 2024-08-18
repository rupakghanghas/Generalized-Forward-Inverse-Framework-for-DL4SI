#!/bin/bash

#SBATCH --cpus-per-task=16 # this requests 1 node, 16 core. 
#SBATCH --time=5-00:00:00 
#SBATCH --gres=gpu:1
#SBATCH --partition=a100_normal_q



export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
ulimit -u 65536

module reset
module load Anaconda3/2020.11

# new virtual environment
source activate openfwi_env


####################################################
# # UNetInverseModel - Small (34.68M) depth=2, repeat=1
####################################################

# FlatVel-A
#python -u train_inverse.py -ds flatvel-a -o ./SupervisedExperiment/FlatVel-A/UNetInverseModel_17M_Latent64_No_Skip/ --tensorboard -eb 150 -m UNetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/flatvel_a_train.txt -v ./train_test_splits/flatvel_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 1 --latent-dim 64 --skip 0
#python -u train_inverse.py -ds flatvel-a -o ./SupervisedExperiment/FlatVel-A/UNetInverseModel_17M_Latent32_No_Skip/ --tensorboard -eb 150 -m UNetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/flatvel_a_train.txt -v ./train_test_splits/flatvel_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 1 --latent-dim 32 --skip 0
# #python -u train_inverse.py -ds flatvel-a -o ./SupervisedExperiment/FlatVel-A/UNetInverseModel_17M_Latent16_No_Skip/ --tensorboard -eb 150 -m UNetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/flatvel_a_train.txt -v ./train_test_splits/flatvel_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 1 --latent-dim 16 --skip 0
# python -u train_inverse.py -ds flatvel-a -o ./SupervisedExperiment/FlatVel-A/UNetInverseModel_17M_Latent8_No_Skip/ --tensorboard -eb 150 -m UNetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/flatvel_a_train.txt -v ./train_test_splits/flatvel_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 1 --latent-dim 8 --skip 0

# # CurveVel-A
#python -u train_inverse.py -ds curvevel-a -o ./SupervisedExperiment/CurveVel-A/UNetInverseModel_17M_Latent64_No_Skip/ --tensorboard -eb 150 -m UNetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/curvevel_a_train.txt -v ./train_test_splits/curvevel_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 1 --latent-dim 64 --skip 0
#python -u train_inverse.py -ds curvevel-a -o ./SupervisedExperiment/CurveVel-A/UNetInverseModel_17M_Latent32_No_Skip/ --tensorboard -eb 150 -m UNetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/curvevel_a_train.txt -v ./train_test_splits/curvevel_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 1 --latent-dim 32 --skip 0
#python -u train_inverse.py -ds curvevel-a -o ./SupervisedExperiment/CurveVel-A/UNetInverseModel_17M_Latent16_No_Skip/ --tensorboard -eb 150 -m UNetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/curvevel_a_train.txt -v ./train_test_splits/curvevel_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 1 --latent-dim 16 --skip 0
python -u train_inverse.py -ds curvevel-a -o ./SupervisedExperiment/CurveVel-A/UNetInverseModel_17M_Latent8_No_Skip/ --tensorboard -eb 150 -m UNetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/curvevel_a_train.txt -v ./train_test_splits/curvevel_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 1 --latent-dim 8 --skip 0



# # FlatFault-A
#python -u train_inverse.py -ds flatfault-a -o ./SupervisedExperiment/FlatFault-A/UNetInverseModel_17M_Latent64_No_Skip/ --tensorboard -eb 150 -m UNetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/flatfault_a_train.txt -v ./train_test_splits/flatfault_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 1 --latent-dim 64 --skip 0
#python -u train_inverse.py -ds flatfault-a -o ./SupervisedExperiment/FlatFault-A/UNetInverseModel_17M_Latent32_No_Skip/ --tensorboard -eb 150 -m UNetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/flatfault_a_train.txt -v ./train_test_splits/flatfault_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 1 --latent-dim 32 --skip 0
#python -u train_inverse.py -ds flatfault-a -o ./SupervisedExperiment/FlatFault-A/UNetInverseModel_17M_Latent16_No_Skip/ --tensorboard -eb 150 -m UNetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/flatfault_a_train.txt -v ./train_test_splits/flatfault_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 1 --latent-dim 16 --skip 0
# python -u train_inverse.py -ds flatfault-a -o ./SupervisedExperiment/FlatFault-A/UNetInverseModel_17M_Latent8_No_Skip/ --tensorboard -eb 150 -m UNetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/flatfault_a_train.txt -v ./train_test_splits/flatfault_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 1 --latent-dim 8 --skip 0


# # CurveFault-A
# python -u train_inverse.py -ds curvefault-a -o ./SupervisedExperiment/CurveFault-A/UNetInverseModel_17M_Latent64_No_Skip/ --tensorboard -eb 150 -m UNetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/curvefault_a_train.txt -v ./train_test_splits/curvefault_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 1 --latent-dim 64 --skip 0
# python -u train_inverse.py -ds curvefault-a -o ./SupervisedExperiment/CurveFault-A/UNetInverseModel_17M_Latent32_No_Skip/ --tensorboard -eb 150 -m UNetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/curvefault_a_train.txt -v ./train_test_splits/curvefault_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 1 --latent-dim 32 --skip 0
# python -u train_inverse.py -ds curvefault-a -o ./SupervisedExperiment/CurveFault-A/UNetInverseModel_17M_Latent16_No_Skip/ --tensorboard -eb 150 -m UNetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/curvefault_a_train.txt -v ./train_test_splits/curvefault_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 1 --latent-dim 16 --skip 0
python -u train_inverse.py -ds curvefault-a -o ./SupervisedExperiment/CurveFault-A/UNetInverseModel_17M_Latent8_No_Skip/ --tensorboard -eb 150 -m UNetInverseModel --lambda_vel 1.0 --lambda_vgg_vel 0.1 -t ./train_test_splits/curvefault_a_train.txt -v ./train_test_splits/curvefault_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 1 --latent-dim 8 --skip 0

exit;