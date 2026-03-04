from typing import Optional
from policy_sources.visualnav_transformer.deployment.src.utils import load_model as deployment_load_model
from policy_sources.omnivla.inference.run_omnivla_modified import Inference
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

import torch
import yaml

class InferenceConfigOriginal:
    resume: bool = True
    # vla_path: str = "./omnivla-original"
    # resume_step: Optional[int] = 120000
    vla_path: str = "./weights/omnivla-finetuned-cast"   
    resume_step: Optional[int] = 210000
    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_film: bool = False
    num_images_in_input: int = 2
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0

class InferenceConfigFinetuned:
    resume: bool = True
    # vla_path: str = "./omnivla-original"
    # resume_step: Optional[int] = 120000    
    vla_path: str = "./weights/omnivla-finetuned-chop"   
    resume_step: Optional[int] = 222500
    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_film: bool = False
    num_images_in_input: int = 2
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0
    
def load_config(model_name: str, finetuned: bool = True):
    
    if model_name in {"vint", "gnm", "nomad"}:
        config_path = f"configs/chop_{model_name}_vnt.yaml"
        with open("configs/chop_default_vnt.yaml", "r") as f:
            default_config = yaml.safe_load(f)

        config = default_config

        with open(config_path, "r") as f:
            user_config = yaml.safe_load(f)

        config.update(user_config)
    else:
        with open("configs/chop_omnivla.yaml", "r") as f:
            config = yaml.safe_load(f)
    return config

def load_model(config: dict, finetuned: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = None
    noise_scheduler = None
    model_name = config.get("model_type")
    
    if model_name in {"vint", "gnm", "nomad"}:
        ckpt_path = config["chop_finetuned_path"] if finetuned else config["pretrained_model_path"]
        model = deployment_load_model(str(ckpt_path), config, device)

        if model_name == "nomad":
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=config["num_diffusion_iters"],
                beta_schedule='squaredcos_cap_v2',
                clip_sample=True,
                prediction_type='epsilon'
            )

    elif model_name == "omnivla":
        if finetuned:
            vla_config = InferenceConfigFinetuned()
        else:
            vla_config = InferenceConfigOriginal()

        model = Inference(save_dir="./inference",
                        ego_frame_mode=True,
                        save_images=False, 
                        radians=True,
                        vla_config=vla_config)
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    if model is None:
        raise RuntimeError("Model failed to initialize.")

    # Some wrappers (e.g., OmnivLA Inference) are not nn.Modules; guard attribute usage.
    if hasattr(model, "to"):
        model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    if hasattr(model, "requires_grad_"):
        model.requires_grad_(False)
    return model, noise_scheduler
