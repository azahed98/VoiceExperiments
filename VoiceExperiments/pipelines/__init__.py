from .SoundStream import SoundStream
from .EnCodec import EnCodec
from .AV2ACodec import AV2ACodec

# TODO
def get_pipeline(pipeline_cfg, opt_cfgs):
    name = pipeline_cfg.pipeline.lower()

    if name == "soundstream":
        return SoundStream(pipeline_cfg, opt_cfgs)
    elif name == "encodec":
        return EnCodec(pipeline_cfg, opt_cfgs)
    elif name == "av2acodec":
        return AV2ACodec(pipeline_cfg, opt_cfgs)
    
    raise NotImplementedError(f"Pipeline {name} not implemented")