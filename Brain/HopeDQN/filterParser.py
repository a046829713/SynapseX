from omegaconf import DictConfig, OmegaConf

def unwrap_config(cfg: DictConfig) -> DictConfig:
    if "model" in cfg:
        return cfg
    if "hope" in cfg:
        return cfg.hope
    return cfg