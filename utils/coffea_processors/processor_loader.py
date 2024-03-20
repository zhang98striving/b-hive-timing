from utils.coffea_processors.pf_candidate_and_vertex import PFCandidateAndVertexProcessing
from utils.coffea_processors.l1_processor import L1PFCandidateAndVertexProcessing 

class ProcessorClasses:
    PFCandidateAndVertexProcessing = "PFCandidateAndVertexProcessing" 
    L1PFCandidateAndVertexProcessing = "L1PFCandidateAndVertexProcessing"


def ProcessorLoader(model: str = None, *args, **kwargs):
    match model:
        case ProcessorClasses.PFCandidateAndVertexProcessing:
            return PFCandidateAndVertexProcessing(*args, **kwargs)
        case ProcessorClasses.L1PFCandidateAndVertexProcessing:
            return L1PFCandidateAndVertexProcessing(*args, **kwargs)
        case _:
            return ProcessorClasses.PFCandidateAndVertexProcessing(*args, **kwargs) 

