from utils.models.deepjet import DeepJetHLT, DeepJet
from utils.models.deepjettransformer import DeepJetTransformer
from utils.models.particletransformer import ParticleTransformer
from utils.models.particlenet_base import ParticleNetTagger
from utils.models.l1t_kerasDeepset import L1TKerasDeepSet


class ModelName:
    DeepJet = "DeepJet"
    DeepJetHLT = "DeepJetHLT"
    ParticleTransformer = "ParticleTransformer"
    DeepJetTransformer = "DeepJetTransformer"
    ParticleNet = "ParticleNet"
    L1TKerasDeepSet = "L1TKerasDeepSet"


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
        case ModelName.ParticleNet:
            return ParticleNetTagger(*args, **kwargs)
        case ModelName.L1TKerasDeepSet:
            return L1TKerasDeepSet(*args, **kwargs)
        case _:
            raise NotImplementedError
