#!/bin/bash


############## Please update base path in the evaluate scripts to the local path of saved models ##############
############OPEN FWI LOADED CKPTS##############
# python -u evaluate/evaluate_zero_shot_inverse_models_loaded_ckpts_OPENFWI.py --model_save_name "InversionNet_ckpt" --model_type "InversionNet"
# python -u evaluate/evaluate_zero_shot_inverse_models_loaded_ckpts_OPENFWI.py --model_save_name "Velocity_GAN" --model_type "InversionNet"

# ############ FORWARD ZERO SHOT #################
# python -u evaluate/evaluate_zero_shot_forward_models.py --model_save_name "WaveformNet" --model_type "WaveformNet" 
# python -u evaluate/evaluate_zero_shot_forward_models.py --model_save_name "FNO" --model_type "FNO"
# python -u evaluate/evaluate_zero_shot_forward_models.py --model_save_name "IUnetForwardModel" --model_type "IUnetForwardModel"
# python -u evaluate/evaluate_zero_shot_forward_models.py --model_save_name "UNetForwardModel_33M" --model_type "UNetForwardModel"
# python -u evaluate/evaluate_zero_shot_forward_models.py --model_save_name "UNetForwardModel_17M" --model_type "UNetForwardModel" --unet_repeat_blocks 1

############ INVERSE ZERO SHOT #################
# python -u evaluate/evaluate_zero_shot_inverse_models.py --model_save_name "InversionNet" --model_type "InversionNet"
# python -u evaluate/evaluate_zero_shot_inverse_models.py --model_save_name "UNetInverseModel_33M" --model_type "UNetInverseModel"
# python -u evaluate/evaluate_zero_shot_inverse_models.py --model_save_name "UNetInverseModel_17M" --model_type "UNetInverseModel" --unet_repeat_blocks 1

############################## UNET EXPERIMENTS ZERO SHOT #############################

# python -u evaluate/evaluate_zero_shot_inverse_models.py --model_save_name "UNetInverseModel_33M" --model_type "UNetInverseModel" --unet_repeat_blocks 2  
# python -u evaluate/evaluate_zero_shot_inverse_models.py --model_save_name "UNetInverseModel_17M" --model_type "UNetInverseModel" --unet_repeat_blocks 1 


# python -u evaluate/evaluate_zero_shot_inverse_models.py --model_save_name "IUnetInverseModel" --model_type "IUnetInverseModel"
# python -u evaluate/evaluate_zero_shot_inverse_models.py --model_save_name "Invertible_XNet" --model_type "IUNET"
# python -u evaluate/evaluate_zero_shot_inverse_models.py --model_save_name "Invertible_XNet_cycle_warmup" --model_type "IUNET"
# python -u evaluate/evaluate_zero_shot_inverse_models.py --model_save_name "Invertible_XNet_Adam" --model_type "IUNET"
# python -u evaluate/evaluate_zero_shot_inverse_models.py --model_save_name "Invertible_XNet_Adam_cycle_warmup" --model_type "IUNET"
