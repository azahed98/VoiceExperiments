from .SoundStream import SoundStream
from .EnCodec import EnCodec

# TODO
def get_pipeline(pipeline_cfg, opt_cfgs):
    name = pipeline_cfg.pipeline.lower()

    if name == "soundstream":
        return SoundStream(model_cfg, opt_cfgs)
    elif name == "encodec":
        return EnCodec(model_cfg, opt_cfgs)
    
    raise NotImplementedError(f"Model {name} not implemented")