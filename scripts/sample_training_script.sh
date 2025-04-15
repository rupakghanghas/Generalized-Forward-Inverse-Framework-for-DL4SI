####################################################
## Invertible X-Net without cycle loss Adamax
##################################################### 
# FlatVel-A
python -u train_invertible_x_net.py -ds flatvel-a -o ./SupervisedExperiment/FlatVel-A/Invertible_XNet/ --optimizer "Adamax" --lr 0.002 --tensorboard -eb 150 -m IUNET --mask_factor 0.0 --lambda_amp 10.0 --lambda_vel 1.0 --lambda_vgg_vel 0.1 --lambda_vgg_amp 0.3 --lambda_cycle_vel 0.0 --lambda_cycle_amp 0.0 -t ./train_test_splits/flatvel_a_train.txt -v ./train_test_splits/flatvel_a_val.txt -ap ""

# To change dataset change -ds <dataset_name> . Dataset names possible: flatvel-a/curvevel-a/curvelfault-a/style-a and similary for b datasets. Also update the train_test_splits file names accordingly. For example for CurveFault A, the train_test_splits file names will be curvefault_a_train.txt and curvefault_a_val.txt
# To change the output directory, change -o <output_directory> . For example for CurveFault A, the output directory will be ./SupervisedExperiment/CurveFault-A/Invertible_XNet/
# To change the optimizer, change --optimizer <optimizer_name> . Optimizer names possible: Adam/Adamax/SGD
# To change the learning rate, change --lr <learning_rate> . For example for CurveFault A, the learning rate will be 0.002

###To train for a specific latent depth, add --latent_depth <depth> . Latent depths possible: 8/16/32/64/70
## For example for Curvel A with 64 latent depth
python -u train_invertible_x_net.py -ds curvevel-a -o ./SupervisedExperiment/CurveVel-A/Invertible_XNet_Latent64/ --optimizer "Adamax" --lr 0.002 --tensorboard -eb 150 -m IUNET --mask_factor 0.0 --lambda_amp 10.0 --lambda_vel 1.0 --lambda_vgg_vel 0.1 --lambda_vgg_amp 0.3 --lambda_cycle_vel 0.0 --lambda_cycle_amp 0.0 -t ./train_test_splits/curvevel_a_train.txt -v ./train_test_splits/curvevel_a_val.txt -ap "" --latent-dim 64


## For training on a fraction of the dataset, add --mask_factor <fraction> . Fraction possible: 0.0/0.1/0.2/0.3/0.4/0.5/0.6/0.7/0.8/0.9
## Default Mask Factor is 0
# # Style-A
python -u train_invertible_x_net.py -ds style-a -o ./Training_Fraction_Experiment/Style-A/Invertible_XNet/Mask_Factor_90/  --optimizer "Adamax" --lr 0.002 --tensorboard -eb 150 -m IUNET --mask_factor 0.90 --lambda_amp 10.0 --lambda_vel 1.0 --lambda_vgg_vel 0.1 --lambda_vgg_amp 0.3 --lambda_cycle_vel 0.0 --lambda_cycle_amp 0.0  -t ./train_test_splits/style_a_train.txt -v ./train_test_splits/style_a_val.txt -ap ""
