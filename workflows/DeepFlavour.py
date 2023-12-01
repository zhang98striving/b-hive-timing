import luigi

from tasks.plotting import ROCCurveTask
from tasks.working_point import WorkingPointTask
from tasks.base import BaseTask
from tasks.parameter_mixins import TrainingDependency, DatasetDependency
from rich.console import Console

c = Console()


class DeepJetRun(TrainingDependency, DatasetDependency, BaseTask):
    training_filelist = luigi.Parameter(
        description="txt file with input root files for training."
    )
    test_filelist = luigi.Parameter(
        description="txt file with input root files for testing."
    )

    def requires(self):
        return [
            ROCCurveTask.req(
                self,
                # this needs to be configured correctly
                # DatasetConstructorTask_training_filelist=self.training_filelist,
                # DatasetConstructorTask_test_filelist=self.test_filelist,
            ),
            WorkingPointTask.req(self),
        ]

    def output(self):
        return self.local_target("deepjetrun.txt")

    def run(self):
        c.print("Everything ready! Well done!")
