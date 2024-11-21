from utils.coffea_processors.pf_candidate_and_vertex import PFCandidateAndVertexProcessing
from utils.coffea_processors.l1_processor import L1PFCandidateAndVertexProcessing 
from utils.coffea_processors.custom_selections import PairedTaggerProcessor

class ProcessorClasses:
    PFCandidateAndVertexProcessing = "PFCandidateAndVertexProcessing" 
    L1PFCandidateAndVertexProcessing = "L1PFCandidateAndVertexProcessing"
    PairedTaggerProcessor = "PairedTaggerProcessor"


def ProcessorLoader(model: str = "", *args, **kwargs):
    match model:
        case ProcessorClasses.PFCandidateAndVertexProcessing:
            return PFCandidateAndVertexProcessing(*args, **kwargs)
        case ProcessorClasses.L1PFCandidateAndVertexProcessing:
            return L1PFCandidateAndVertexProcessing(*args, **kwargs)
        case ProcessorClasses.PairedTaggerProcessor:
            return PairedTaggerProcessor(*args, **kwargs)
        case _:
            return PFCandidateAndVertexProcessing(*args, **kwargs) 

