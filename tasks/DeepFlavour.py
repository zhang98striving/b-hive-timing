from tasks.plotting import PlottingTask
from tasks.BaseTask import MainBaseTask
from rich.console import Console

c = Console()


class DeepJetRun(MainBaseTask):
    def requires(self):
        return PlottingTask.req(self)

    def output(self):
        return self.local_target("deepjetrun.txt")

    def run(self):
        c.print("Alles ready! Well done!")
