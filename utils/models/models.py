from utils.models.deepjet import DeepJet
from utils.models.deepjettransformer import DeepJetTransformer
from utils.models.particletransformer import ParticleTransformer


class ModelName:
    DeepJet = "DeepJet"
    ParticleTransformer = "ParticleTransformer"
    DeepJetTransformer = "DeepJetTransformer"


def BTaggingModels(model: str = ModelName.DeepJet, *args, **kwargs):
    match model:
        case ModelName.DeepJet:
            return DeepJet(*args, **kwargs)
        case ModelName.ParticleTransformer:
            return ParticleTransformer(*args, **kwargs)
        case ModelName.DeepJetTransformer:
            return DeepJetTransformer(*args, **kwargs)
        case _:
            raise NotImplementedError
