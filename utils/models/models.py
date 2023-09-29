from typing import Any

import torch.nn as nn

from utils.models.deepjet import DeepJet
from utils.models.deepjettransformer import DeepJetTransformer
from utils.models.particletransformer import ParticleTransformer


class ModelName:
    DeepJet = "DeepJet"
    ParticleTransformer = "ParticleTransformer"
    DeepJetTransformer = "DeepJetTransformer"


class BTaggingModels(nn.Module):
    def __init__(self, model: str = ModelName.DeepJet, *args, **kwargs):
        super().__init__()
        self.model_str = model
        match model:
            case ModelName.DeepJet:
                self.model = DeepJet(*args, **kwargs)
            case ModelName.ParticleTransformer:
                self.model = ParticleTransformer(
                    *args,
                    **kwargs,
                )
            case ModelName.DeepJetTransformer:
                self.model = DeepJetTransformer(*args, **kwargs)
            case _:
                raise NotImplementedError

    def __call__(self, *args, **kwds):
        return self.model(*args, **kwds)

    def __str__(self):
        return self.model_str
