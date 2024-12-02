from utils.torch.DeepJetDataset import DeepJetDataset
from utils.torch.LZ4Dataset import LZ4Dataset
from utils.torch.LZ4FP16Dataset import LZ4FP16Dataset
from utils.torch.PAIReDDataset import PAIReDDataset
from utils.torch.LZ4PAIReDDataset import LZ4PAIReDDataset

class DatasetName:
    DeepJetDataset   = "DeepJetDataset"
    LZ4Dataset       = "LZ4Dataset"
    LZ4FP16Dataset   = "LZ4FP16Dataset"
    PAIReDDataset    = "PAIReDDataset"
    LZ4PAIReDDataset = "LZ4PAIReDDataset"
    
def DatasetLoader(dataset_name: str = ""):
    match dataset_name:
        case DatasetName.DeepJetDataset:
            return DeepJetDataset
        case DatasetName.LZ4Dataset:
            return LZ4Dataset
        case DatasetName.LZ4FP16Dataset:
            return LZ4FP16Dataset
        case DatasetName.PAIReDDataset:
            return PAIReDDataset
        case DatasetName.LZ4PAIReDDataset:
            return LZ4PAIReDDataset
        case _:
            raise NotImplementedError