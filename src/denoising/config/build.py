from .schema import Config, RunCfg, DataCfg, MaskCfg, ModelCfg, OptimCfg

def build_config(raw: dict) -> Config:
    return Config(
        run=RunCfg(**raw["run"]),
        data=DataCfg(
            train=raw["data"]["train"],
            val=raw["data"]["val"],
            image_axes=tuple(raw["data"]["image_axes"]),
            fourier_axes=tuple(raw["data"]["fourier_axes"]),
            num_samples=raw["data"]["num_samples"],
            val_samples=raw["data"]["val_samples"],
        ),
        mask=MaskCfg(**raw["masking"]),
        model=ModelCfg(
            in_channels=raw["model"]["in_channels"],
            out_channels=raw["model"]["out_channels"],
            features=tuple(raw["model"]["features"]),
        ),
        optim=OptimCfg(**raw["optim"]),
    )