import law
import luigi

from tasks.plotting import ROCCurveTask
from tasks.working_point import WorkingPointTask
from rich.console import Console

c = Console()


class DeepJetRun(law.WrapperTask):
    """
    This runs a full DeepJet Training on the specified files
    with a specified version
    """

    version = luigi.Parameter()
    training_filelist = luigi.Parameter(
        description="txt file with input root files for training."
    )
    test_filelist = luigi.Parameter(
        description="txt file with input root files for testing."
    )

    def requires(self):
        kwargs = {
            "training_version": self.version,
            "dataset_version": self.version,
            "model_name": "DeepJet",
            "epochs": 10,
        }
        return [
            ROCCurveTask.req(self, **kwargs),
            WorkingPointTask.req(self, **kwargs),
        ]
