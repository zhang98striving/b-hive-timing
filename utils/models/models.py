from utils.models.deepjet import DeepJetHLT, DeepJet
from utils.models.deepjettransformer import DeepJetTransformer
from utils.models.particletransformer import ParticleTransformer


class ModelName:
    DeepJet = "DeepJet"
    DeepJetHLT = "DeepJetHLT"
    ParticleTransformer = "ParticleTransformer"
    DeepJetTransformer = "DeepJetTransformer"


def BTaggingModels(model: str = None, *args, **kwargs):
    match model:
        case ModelName.DeepJet:
            return DeepJet(*args, **kwargs)
        case ModelName.DeepJetHLT:
            return DeepJetHLT(*args, **kwargs)
        case ModelName.ParticleTransformer:
            return ParticleTransformer(*args, **kwargs)
        case ModelName.DeepJetTransformer:
            return DeepJetTransformer(*args, **kwargs)
        case _:
            raise NotImplementedError
