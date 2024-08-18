import json
import os

def get_config_name(latent_dim=70, model_type="velocity"):
    cfg_name = f"{model_type}_config_latent_dim_{latent_dim}.json"
    return cfg_name

def get_latent_dim(cfg_path, amp_cfg_name):
    config_file = os.path.join(cfg_path, amp_cfg_name)
    with open(config_file, 'r') as file:
        cfg = json.load(file)
    final_encoder_id = list(cfg["encoder_blocks"].keys())[-1]
    return cfg["encoder_blocks"][final_encoder_id]["out_channels"]