#!/bin/bash

#SBATCH --cpus-per-task=16 # this requests 1 node, 16 core. 
#SBATCH --time=4:00:00 
#SBATCH --gres=gpu:1
#SBATCH --partition=a100_normal_q




export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
ulimit -u 65536

module reset
module load Anaconda3/2020.11

# new virtual environment
source activate openfwi_env



############ INVERSE ZERO SHOT #################
# python -u evaluate/evaluate_zero_shot_inverse_models.py --model_save_name "InversionNet" --model_type "InversionNet"
# python -u evaluate/evaluate_zero_shot_inverse_models.py --model_save_name "UNetInverseModel_33M" --model_type "UNetInverseModel"
# python -u evaluate/evaluate_zero_shot_inverse_models.py --model_save_name "UNetInverseModel_17M" --model_type "UNetInverseModel" --unet_repeat_blocks 1



python -u evaluate/evaluate_zero_shot_inverse_models.py --model_save_name "IUnetInverseModel" --model_type "IUnetInverseModel"
# python -u evaluate/evaluate_zero_shot_inverse_models.py --model_save_name "Invertible_XNet" --model_type "IUNET"
# python -u evaluate/evaluate_zero_shot_inverse_models.py --model_save_name "Invertible_XNet_cycle_warmup" --model_type "IUNET"
# python -u evaluate/evaluate_zero_shot_inverse_models.py --model_save_name "Invertible_XNet_Adam" --model_type "IUNET"
# python -u evaluate/evaluate_zero_shot_inverse_models.py --model_save_name "Invertible_XNet_Adam_cycle_warmup" --model_type "IUNET"


############OPEN FWI LOADED CKPTS##############
# python -u evaluate/evaluate_zero_shot_inverse_models_loaded_ckpts_OPENFWI.py --model_save_name "InversionNet_ckpt" --model_type "InversionNet"
# python -u evaluate/evaluate_zero_shot_inverse_models_loaded_ckpts_OPENFWI.py --model_save_name "Velocity_GAN" --model_type "InversionNet"

# ############ FORWARD ZERO SHOT #################
# python -u evaluate/evaluate_zero_shot_forward_models.py --model_save_name "WaveformNet" --model_type "WaveformNet" 
# python -u evaluate/evaluate_zero_shot_forward_models.py --model_save_name "FNO" --model_type "FNO"
# python -u evaluate/evaluate_zero_shot_forward_models.py --model_save_name "IUnetForwardModel" --model_type "IUnetForwardModel"
# python -u evaluate/evaluate_zero_shot_forward_models.py --model_save_name "UNetForwardModel_33M" --model_type "UNetForwardModel"
# python -u evaluate/evaluate_zero_shot_forward_models.py --model_save_name "UNetForwardModel_17M" --model_type "UNetForwardModel" --unet_repeat_blocks 1



############################## UNET EXPERIMENTS ZERO SHOT #############################
################# UNET NO SKIP ZERO SHOT #################

# python -u evaluate/evaluate_zero_shot_inverse_models.py --model_save_name "UNetInverseModel_33M_NoSkip" --model_type "UNetInverseModel" --unet_repeat_blocks 2 --skip 0 
# python -u evaluate/evaluate_zero_shot_inverse_models.py --model_save_name "UNetInverseModel_17M_NoSkip" --model_type "UNetInverseModel" --unet_repeat_blocks 1 --skip 0 


################# LATENT DIM ZERO SHOT #################

# python -u evaluate/evaluate_zero_shot_inverse_models.py --model_save_name "UNetInverseModel_33M_Latent8" --model_type "UNetInverseModel" --unet_repeat_blocks 2 --latent_dim 8
# python -u evaluate/evaluate_zero_shot_inverse_models.py --model_save_name "UNetInverseModel_33M_Latent16" --model_type "UNetInverseModel" --unet_repeat_blocks 2 --latent_dim 16
# python -u evaluate/evaluate_zero_shot_inverse_models.py --model_save_name "UNetInverseModel_33M_Latent32" --model_type "UNetInverseModel" --unet_repeat_blocks 2 --latent_dim 32
# python -u evaluate/evaluate_zero_shot_inverse_models.py --model_save_name "UNetInverseModel_33M_Latent64" --model_type "UNetInverseModel" --unet_repeat_blocks 2 --latent_dim 64


# python -u evaluate/evaluate_zero_shot_inverse_models.py --model_save_name "UNetInverseModel_17M_Latent8" --model_type "UNetInverseModel" --unet_repeat_blocks 1 --latent_dim 8
# python -u evaluate/evaluate_zero_shot_inverse_models.py --model_save_name "UNetInverseModel_17M_Latent16" --model_type "UNetInverseModel" --unet_repeat_blocks 1 --latent_dim 16
# python -u evaluate/evaluate_zero_shot_inverse_models.py --model_save_name "UNetInverseModel_17M_Latent32" --model_type "UNetInverseModel" --unet_repeat_blocks 1 --latent_dim 32
# python -u evaluate/evaluate_zero_shot_inverse_models.py --model_save_name "UNetInverseModel_17M_Latent64" --model_type "UNetInverseModel" --unet_repeat_blocks 1 --latent_dim 64



################# LATENT & NO SKIP ZERO SHOT #################

# python -u evaluate/evaluate_zero_shot_inverse_models.py --model_save_name "UNetInverseModel_33M_Latent8_No_Skip" --model_type "UNetInverseModel" --unet_repeat_blocks 2 --latent_dim 8 --skip 0  
# python -u evaluate/evaluate_zero_shot_inverse_models.py --model_save_name "UNetInverseModel_33M_Latent16_No_Skip" --model_type "UNetInverseModel" --unet_repeat_blocks 2 --latent_dim 16 --skip 0 
# python -u evaluate/evaluate_zero_shot_inverse_models.py --model_save_name "UNetInverseModel_33M_Latent32_No_Skip" --model_type "UNetInverseModel" --unet_repeat_blocks 2 --latent_dim 32 --skip 0 
# python -u evaluate/evaluate_zero_shot_inverse_models.py --model_save_name "UNetInverseModel_33M_Latent64_No_Skip" --model_type "UNetInverseModel" --unet_repeat_blocks 2 --latent_dim 64 --skip 0 


# python -u evaluate/evaluate_zero_shot_inverse_models.py --model_save_name "UNetInverseModel_17M_Latent8_No_Skip" --model_type "UNetInverseModel" --unet_repeat_blocks 1 --latent_dim 8 --skip 0 
# python -u evaluate/evaluate_zero_shot_inverse_models.py --model_save_name "UNetInverseModel_17M_Latent16_No_Skip" --model_type "UNetInverseModel" --unet_repeat_blocks 1 --latent_dim 16 --skip 0 
# python -u evaluate/evaluate_zero_shot_inverse_models.py --model_save_name "UNetInverseModel_17M_Latent32_No_Skip" --model_type "UNetInverseModel" --unet_repeat_blocks 1 --latent_dim 32 --skip 0 
# python -u evaluate/evaluate_zero_shot_inverse_models.py --model_save_name "UNetInverseModel_17M_Latent64_No_Skip" --model_type "UNetInverseModel" --unet_repeat_blocks 1 --latent_dim 64 --skip 0 



############## Supevised Visualizations
# python -u evaluate/evaluate_suprevised_viz.py
#python -u evaluate/evaluate_suprevised_forward_viz.py


################# Supervised Trace Plots
# python -u evaluate/evaluate_suprevised_forward_trace_viz.py
# python -u evaluate/evaluate_suprevised_inverse_trace_viz.py

exit;
