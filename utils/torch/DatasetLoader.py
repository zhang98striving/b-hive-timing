from utils.torch.DeepJetDataset import DeepJetDataset
from utils.torch.LZ4Dataset import LZ4Dataset
from utils.torch.LZ4FP16Dataset import LZ4FP16Dataset

class DatasetName:
    DeepJetDataset = "DeepJetDataset"
    LZ4Dataset     = "LZ4Dataset"
    LZ4FP16Dataset = "LZ4FP16Dataset"


    
def DatasetLoader(dataset_name: str = ""):
    match dataset_name:
        case DatasetName.DeepJetDataset:
            return DeepJetDataset
        case DatasetName.LZ4Dataset:
            return LZ4Dataset
        case DatasetName.LZ4FP16Dataset:
            return LZ4FP16Dataset
        