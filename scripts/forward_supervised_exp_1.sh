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
source activate openfwi_env

###############################################################################################################
############# DATASET : CURVEVEL-A
###############################################################################################################

####################################################
# # UNetForwardModel - Large (34.68M) depth=2, repeat=2 (Running except style)
####################################################

# # FlatVel-A
#  python -u train_forward.py -ds flatvel-a -o ./SupervisedExperiment/FlatVel-A/UNetForwardModel_33M/ --tensorboard -eb 150 -m UNetForwardModel --lambda_amp 1.0 --lambda_vgg_amp 0.1 -t ./train_test_splits/flatvel_a_train.txt -v ./train_test_splits/flatvel_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 2

# # # FlatVel-B
#  python -u train_forward.py -ds flatvel-b -o ./SupervisedExperiment/FlatVel-B/UNetForwardModel_33M/ --tensorboard -eb 150 -m UNetForwardModel --lambda_amp 1.0 --lambda_vgg_amp 0.1 -t ./train_test_splits/flatvel_b_train.txt -v ./train_test_splits/flatvel_b_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 2

# # # CurveVel-A
#  python -u train_forward.py -ds curvevel-a -o ./SupervisedExperiment/CurveVel-A/UNetForwardModel_33M/ --tensorboard -eb 150 -m UNetForwardModel --lambda_amp 1.0 --lambda_vgg_amp 0.1 -t ./train_test_splits/curvevel_a_train.txt -v ./train_test_splits/curvevel_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 2

# # # CurveVel-B
#  python -u train_forward.py -ds curvevel-b -o ./SupervisedExperiment/CurveVel-B/UNetForwardModel_33M/ --tensorboard -eb 150 -m UNetForwardModel --lambda_amp 1.0 --lambda_vgg_amp 0.1 -t ./train_test_splits/curvevel_b_train.txt -v ./train_test_splits/curvevel_b_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 2

# # FlatFault-A
# python -u train_forward.py -ds flatfault-a -o ./SupervisedExperiment/FlatFault-A/UNetForwardModel_33M/ --tensorboard -eb 150 -m UNetForwardModel --lambda_amp 1.0 --lambda_vgg_amp 0.1 -t ./train_test_splits/flatfault_a_train.txt -v ./train_test_splits/flatfault_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 2

# # FlatFault-B
# python -u train_forward.py -ds flatfault-b -o ./SupervisedExperiment/FlatFault-B/UNetForwardModel_33M/ --tensorboard -eb 150 -m UNetForwardModel --lambda_amp 1.0 --lambda_vgg_amp 0.1 -t ./train_test_splits/flatfault_b_train.txt -v ./train_test_splits/flatfault_b_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 2

# # CurveFault-A
# python -u train_forward.py -ds curvefault-a -o ./SupervisedExperiment/CurveFault-A/UNetForwardModel_33M/ --tensorboard -eb 150 -m UNetForwardModel --lambda_amp 1.0 --lambda_vgg_amp 0.1 -t ./train_test_splits/curvefault_a_train.txt -v ./train_test_splits/curvefault_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 2

# # CurveFault-B
# python -u train_forward.py -ds curvefault-b -o ./SupervisedExperiment/CurveFault-B/UNetForwardModel_33M/ --tensorboard -eb 150 -m UNetForwardModel --lambda_amp 1.0 --lambda_vgg_amp 0.1 -t ./train_test_splits/curvefault_b_train.txt -v ./train_test_splits/curvefault_b_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 2

# # Style-A
# python -u train_forward.py -ds style-a -o ./SupervisedExperiment/Style-A/UNetForwardModel_33M/ --tensorboard -eb 150 -m UNetForwardModel --lambda_amp 1.0 --lambda_vgg_amp 0.1 -t ./train_test_splits/style_a_train.txt -v ./train_test_splits/style_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 2

# # Style-B
# python -u train_forward.py -ds style-b -o ./SupervisedExperiment/Style-B/UNetForwardModel_33M/ --tensorboard -eb 150 -m UNetForwardModel --lambda_amp 1.0 --lambda_vgg_amp 0.1 -t ./train_test_splits/style_b_train.txt -v ./train_test_splits/style_b_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 2


####################################################
# # UNetForwardModel - Medium (23.48M) depth=1, repeat=6
####################################################


####################################################
# # UNetForwardModel - Small (17.86M) depth=2, repeat=1  (Running except style)
####################################################

# #FlatVel-A
# python -u train_forward.py -ds flatvel-a -o ./SupervisedExperiment/FlatVel-A/UNetForwardModel_17M/ --tensorboard -eb 150 -m UNetForwardModel --lambda_amp 1.0 --lambda_vgg_amp 0.1 -t ./train_test_splits/flatvel_a_train.txt -v ./train_test_splits/flatvel_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 1

# # FlatVel-B
# python -u train_forward.py -ds flatvel-b -o ./SupervisedExperiment/FlatVel-B/UNetForwardModel_17M/ --tensorboard -eb 150 -m UNetForwardModel --lambda_amp 1.0 --lambda_vgg_amp 0.1 -t ./train_test_splits/flatvel_b_train.txt -v ./train_test_splits/flatvel_b_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 1

# # CurveVel-A
# python -u train_forward.py -ds curvevel-a -o ./SupervisedExperiment/CurveVel-A/UNetForwardModel_17M/ --tensorboard -eb 150 -m UNetForwardModel --lambda_amp 1.0 --lambda_vgg_amp 0.1 -t ./train_test_splits/curvevel_a_train.txt -v ./train_test_splits/curvevel_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 1

# # CurveVel-B
# python -u train_forward.py -ds curvevel-b -o ./SupervisedExperiment/CurveVel-B/UNetForwardModel_17M/ --tensorboard -eb 150 -m UNetForwardModel --lambda_amp 1.0 --lambda_vgg_amp 0.1 -t ./train_test_splits/curvevel_b_train.txt -v ./train_test_splits/curvevel_b_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 1

# # FlatFault-A
# python -u train_forward.py -ds flatfault-a -o ./SupervisedExperiment/FlatFault-A/UNetForwardModel_17M/ --tensorboard -eb 150 -m UNetForwardModel --lambda_amp 1.0 --lambda_vgg_amp 0.1 -t ./train_test_splits/flatfault_a_train.txt -v ./train_test_splits/flatfault_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 1

# # FlatFault-B
# python -u train_forward.py -ds flatfault-b -o ./SupervisedExperiment/FlatFault-B/UNetForwardModel_17M/ --tensorboard -eb 150 -m UNetForwardModel --lambda_amp 1.0 --lambda_vgg_amp 0.1 -t ./train_test_splits/flatfault_b_train.txt -v ./train_test_splits/flatfault_b_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 1

# # CurveFault-A
# python -u train_forward.py -ds curvefault-a -o ./SupervisedExperiment/CurveFault-A/UNetForwardModel_17M/ --tensorboard -eb 150 -m UNetForwardModel --lambda_amp 1.0 --lambda_vgg_amp 0.1 -t ./train_test_splits/curvefault_a_train.txt -v ./train_test_splits/curvefault_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 1

# # CurveFault-B
# python -u train_forward.py -ds curvefault-b -o ./SupervisedExperiment/CurveFault-B/UNetForwardModel_17M/ --tensorboard -eb 150 -m UNetForwardModel --lambda_amp 1.0 --lambda_vgg_amp 0.1 -t ./train_test_splits/curvefault_b_train.txt -v ./train_test_splits/curvefault_b_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 1

# # Style-A
# python -u train_forward.py -ds style-a -o ./SupervisedExperiment/Style-A/UNetForwardModel_17M/ --tensorboard -eb 150 -m UNetForwardModel --lambda_amp 1.0 --lambda_vgg_amp 0.1 -t ./train_test_splits/style_a_train.txt -v ./train_test_splits/style_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 1

# # Style-B
# python -u train_forward.py -ds style-b -o ./SupervisedExperiment/Style-B/UNetForwardModel_17M/ --tensorboard -eb 150 -m UNetForwardModel --lambda_amp 1.0 --lambda_vgg_amp 0.1 -t ./train_test_splits/style_b_train.txt -v ./train_test_splits/style_b_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" --unet_depth 2 --unet_repeat_blocks 1


####################################################
# # IUnetForwardModel (Running execpt  style)
####################################################

# #FlatVel-A
# python -u train_forward.py -ds flatvel-a -o ./SupervisedExperiment/FlatVel-A/IUnetForwardModel/ --tensorboard -eb 150 -m IUnetForwardModel --lambda_amp 1.0 --lambda_vgg_amp 0.1 -t ./train_test_splits/flatvel_a_train.txt -v ./train_test_splits/flatvel_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adamax" --lr 0.002 

# # FlatVel-B
# python -u train_forward.py -ds flatvel-b -o ./SupervisedExperiment/FlatVel-B/IUnetForwardModel/ --tensorboard -eb 150 -m IUnetForwardModel --lambda_amp 1.0 --lambda_vgg_amp 0.1 -t ./train_test_splits/flatvel_b_train.txt -v ./train_test_splits/flatvel_b_val.txt -ap "" --mask_factor 0.0 --optimizer "Adamax" --lr 0.002  

# # CurveVel-A
# python -u train_forward.py -ds curvevel-a -o ./SupervisedExperiment/CurveVel-A/IUnetForwardModel/ --tensorboard -eb 150 -m IUnetForwardModel --lambda_amp 1.0 --lambda_vgg_amp 0.1 -t ./train_test_splits/curvevel_a_train.txt -v ./train_test_splits/curvevel_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adamax" --lr 0.002 

# # CurveVel-B
# python -u train_forward.py -ds curvevel-b -o ./SupervisedExperiment/CurveVel-B/IUnetForwardModel/ --tensorboard -eb 150 -m IUnetForwardModel --lambda_amp 1.0 --lambda_vgg_amp 0.1 -t ./train_test_splits/curvevel_b_train.txt -v ./train_test_splits/curvevel_b_val.txt -ap "" --mask_factor 0.0 --optimizer "Adamax" --lr 0.002 

# # FlatFault-A
# python -u train_forward.py -ds flatfault-a -o ./SupervisedExperiment/FlatFault-A/IUnetForwardModel/ --tensorboard -eb 150 -m IUnetForwardModel --lambda_amp 1.0 --lambda_vgg_amp 0.1 -t ./train_test_splits/flatfault_a_train.txt -v ./train_test_splits/flatfault_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adamax" --lr 0.002 

# # FlatFault-B
# python -u train_forward.py -ds flatfault-b -o ./SupervisedExperiment/FlatFault-B/IUnetForwardModel/ --tensorboard -eb 150 -m IUnetForwardModel --lambda_amp 1.0 --lambda_vgg_amp 0.1 -t ./train_test_splits/flatfault_b_train.txt -v ./train_test_splits/flatfault_b_val.txt -ap "" --mask_factor 0.0 --optimizer "Adamax" --lr 0.002 

# CurveFault-A
python -u train_forward.py -ds curvefault-a -o ./SupervisedExperiment/CurveFault-A/IUnetForwardModel/ --tensorboard -eb 150 -m IUnetForwardModel --lambda_amp 1.0 --lambda_vgg_amp 0.1 -t ./train_test_splits/curvefault_a_train.txt -v ./train_test_splits/curvefault_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adamax" --lr 0.002 

# CurveFault-B
python -u train_forward.py -ds curvefault-b -o ./SupervisedExperiment/CurveFault-B/IUnetForwardModel/ --tensorboard -eb 150 -m IUnetForwardModel --lambda_amp 1.0 --lambda_vgg_amp 0.1 -t ./train_test_splits/curvefault_b_train.txt -v ./train_test_splits/curvefault_b_val.txt -ap "" --mask_factor 0.0 --optimizer "Adamax" --lr 0.002 

# # Style-A
# python -u train_forward.py -ds style-a -o ./SupervisedExperiment/Style-A/IUnetForwardModel/ --tensorboard -eb 150 -m IUnetForwardModel --lambda_amp 1.0 --lambda_vgg_amp 0.1 -t ./train_test_splits/style_a_train.txt -v ./train_test_splits/style_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adamax" --lr 0.002 

# # Style-B
# python -u train_forward.py -ds style-b -o ./SupervisedExperiment/Style-B/IUnetForwardModel/ --tensorboard -eb 150 -m IUnetForwardModel --lambda_amp 1.0 --lambda_vgg_amp 0.1 -t ./train_test_splits/style_b_train.txt -v ./train_test_splits/style_b_val.txt -ap "" --mask_factor 0.0 --optimizer "Adamax" --lr 0.002 


####################################################
# # WaveformNet
####################################################

# FlatVel-A
# python -u train_forward.py -ds flatvel-a -o ./SupervisedExperiment/FlatVel-A/WaveformNet/ --tensorboard -eb 150 -m WaveformNet --lambda_amp 1.0 --lambda_vgg_amp 0.1 -t ./train_test_splits/flatvel_a_train.txt -v ./train_test_splits/flatvel_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" 

# # FlatVel-B
# python -u train_forward.py -ds flatvel-b -o ./SupervisedExperiment/FlatVel-B/WaveformNet/ --tensorboard -eb 150 -m WaveformNet --lambda_amp 1.0 --lambda_vgg_amp 0.1 -t ./train_test_splits/flatvel_b_train.txt -v ./train_test_splits/flatvel_b_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" 

# # CurveVel-A
# python -u train_forward.py -ds curvevel-a -o ./SupervisedExperiment/CurveVel-A/WaveformNet/ --tensorboard -eb 150 -m WaveformNet --lambda_amp 1.0 --lambda_vgg_amp 0.1 -t ./train_test_splits/curvevel_a_train.txt -v ./train_test_splits/curvevel_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam"  

# # CurveVel-B
# python -u train_forward.py -ds curvevel-b -o ./SupervisedExperiment/CurveVel-B/WaveformNet/ --tensorboard -eb 150 -m WaveformNet --lambda_amp 1.0 --lambda_vgg_amp 0.1 -t ./train_test_splits/curvevel_b_train.txt -v ./train_test_splits/curvevel_b_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" 

# # FlatFault-A
# python -u train_forward.py -ds flatfault-a -o ./SupervisedExperiment/FlatFault-A/WaveformNet/ --tensorboard -eb 150 -m WaveformNet --lambda_amp 1.0 --lambda_vgg_amp 0.1 -t ./train_test_splits/flatfault_a_train.txt -v ./train_test_splits/flatfault_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" 

# # FlatFault-B
# python -u train_forward.py -ds flatfault-b -o ./SupervisedExperiment/FlatFault-B/WaveformNet/ --tensorboard -eb 150 -m WaveformNet --lambda_amp 1.0 --lambda_vgg_amp 0.1 -t ./train_test_splits/flatfault_b_train.txt -v ./train_test_splits/flatfault_b_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" 

# # CurveFault-A
# python -u train_forward.py -ds curvefault-a -o ./SupervisedExperiment/CurveFault-A/WaveformNet/ --tensorboard -eb 150 -m WaveformNet --lambda_amp 1.0 --lambda_vgg_amp 0.1 -t ./train_test_splits/curvefault_a_train.txt -v ./train_test_splits/curvefault_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" 

# # CurveFault-B
# python -u train_forward.py -ds curvefault-b -o ./SupervisedExperiment/CurveFault-B/WaveformNet/ --tensorboard -eb 150 -m WaveformNet --lambda_amp 1.0 --lambda_vgg_amp 0.1 -t ./train_test_splits/curvefault_b_train.txt -v ./train_test_splits/curvefault_b_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam"  

# # Style-A
# python -u train_forward.py -ds style-a -o ./SupervisedExperiment/Style-A/WaveformNet/ --tensorboard -eb 150 -m WaveformNet --lambda_amp 1.0 --lambda_vgg_amp 0.1 -t ./train_test_splits/style_a_train.txt -v ./train_test_splits/style_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" 

# # Style-B
# python -u train_forward.py -ds style-b -o ./SupervisedExperiment/Style-B/WaveformNet/ --tensorboard -eb 150 -m WaveformNet --lambda_amp 1.0 --lambda_vgg_amp 0.1 -t ./train_test_splits/style_b_train.txt -v ./train_test_splits/style_b_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" 

####################################################
# # FNO  (Running except style)
####################################################

# #FlatVel-A
# python -u train_forward.py -ds flatvel-a -o ./SupervisedExperiment/FlatVel-A/FNO/ --tensorboard -eb 150 -m FNO --lambda_amp 1.0 --lambda_vgg_amp 0.1 -t ./train_test_splits/flatvel_a_train.txt -v ./train_test_splits/flatvel_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" 

# # FlatVel-B
# python -u train_forward.py -ds flatvel-b -o ./SupervisedExperiment/FlatVel-B/FNO/ --tensorboard -eb 150 -m FNO --lambda_amp 1.0 --lambda_vgg_amp 0.1 -t ./train_test_splits/flatvel_b_train.txt -v ./train_test_splits/flatvel_b_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" 

# # CurveVel-A
# python -u train_forward.py -ds curvevel-a -o ./SupervisedExperiment/CurveVel-A/FNO/ --tensorboard -eb 150 -m FNO --lambda_amp 1.0 --lambda_vgg_amp 0.1 -t ./train_test_splits/curvevel_a_train.txt -v ./train_test_splits/curvevel_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam"  

# # CurveVel-B
# python -u train_forward.py -ds curvevel-b -o ./SupervisedExperiment/CurveVel-B/FNO/ --tensorboard -eb 150 -m FNO --lambda_amp 1.0 --lambda_vgg_amp 0.1 -t ./train_test_splits/curvevel_b_train.txt -v ./train_test_splits/curvevel_b_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" 

# # FlatFault-A
# python -u train_forward.py -ds flatfault-a -o ./SupervisedExperiment/FlatFault-A/FNO/ --tensorboard -eb 150 -m FNO --lambda_amp 1.0 --lambda_vgg_amp 0.1 -t ./train_test_splits/flatfault_a_train.txt -v ./train_test_splits/flatfault_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" 

# # FlatFault-B
# python -u train_forward.py -ds flatfault-b -o ./SupervisedExperiment/FlatFault-B/FNO/ --tensorboard -eb 150 -m FNO --lambda_amp 1.0 --lambda_vgg_amp 0.1 -t ./train_test_splits/flatfault_b_train.txt -v ./train_test_splits/flatfault_b_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" 

# # CurveFault-A
# python -u train_forward.py -ds curvefault-a -o ./SupervisedExperiment/CurveFault-A/FNO/ --tensorboard -eb 150 -m FNO --lambda_amp 1.0 --lambda_vgg_amp 0.1 -t ./train_test_splits/curvefault_a_train.txt -v ./train_test_splits/curvefault_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" 

# # CurveFault-B
# python -u train_forward.py -ds curvefault-b -o ./SupervisedExperiment/CurveFault-B/FNO/ --tensorboard -eb 150 -m FNO --lambda_amp 1.0 --lambda_vgg_amp 0.1 -t ./train_test_splits/curvefault_b_train.txt -v ./train_test_splits/curvefault_b_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam"  

# # Style-A
# python -u train_forward.py -ds style-a -o ./SupervisedExperiment/Style-A/FNO/ --tensorboard -eb 150 -m FNO --lambda_amp 1.0 --lambda_vgg_amp 0.1 -t ./train_test_splits/style_a_train.txt -v ./train_test_splits/style_a_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" 

# # Style-B
# python -u train_forward.py -ds style-b -o ./SupervisedExperiment/Style-B/FNO/ --tensorboard -eb 150 -m FNO --lambda_amp 1.0 --lambda_vgg_amp 0.1 -t ./train_test_splits/style_b_train.txt -v ./train_test_splits/style_b_val.txt -ap "" --mask_factor 0.0 --optimizer "Adam" 

exit;
