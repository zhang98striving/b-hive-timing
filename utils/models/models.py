from utils.models.particletransformer import ParticleTransformer
from utils.models.deepjettransformer import DeepJetTransformer
from utils.models.particlenet_base import ParticleNetTagger
from utils.models.deepjet import DeepJetHLT, DeepJet, MoDJet
from utils.models.l1t_kerasDeepset import L1TKerasDeepSet
from utils.models.l1t_base import L1TTorchBase
from utils.models.PAIReDTagger import PAIReDTagger


class ModelName:
    DeepJet = "DeepJet"
    DeepJetHLT = "DeepJetHLT"
    MoDJet = "MoDJet"
    ParticleTransformer = "ParticleTransformer"
    DeepJetTransformer = "DeepJetTransformer"
    ParticleNet = "ParticleNet"
    ParticleNetHION = "ParticleNetHION"
    L1TKerasDeepSet = "L1TKerasDeepSet"
    L1TTorchBase = "L1TTorchBase"
    PAIReDTagger = "PAIReDTagger"


def BTaggingModels(model: str = "", *args, **kwargs):
    match model:
        case ModelName.DeepJet:
            return DeepJet(*args, **kwargs)
        case ModelName.DeepJetHLT:
            return DeepJetHLT(*args, **kwargs)
        case ModelName.MoDJet:
            return MoDJet(*args, **kwargs)
        case ModelName.ParticleTransformer:
            return ParticleTransformer(*args, **kwargs)
        case ModelName.DeepJetTransformer:
            return DeepJetTransformer(*args, **kwargs)
        case ModelName.ParticleNet:
            return ParticleNetTagger(*args, **kwargs)
        case ModelName.L1TKerasDeepSet:
            return L1TKerasDeepSet(*args, **kwargs)
        case ModelName.L1TTorchBase:
            return L1TTorchBase(*args, **kwargs)
        case ModelName.PAIReDTagger:
            return PAIReDTagger(*args, **kwargs)
        case _:
            raise NotImplementedError
