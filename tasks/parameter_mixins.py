import luigi
import law


class DatasetDependency(object):
    dataset_version = luigi.Parameter(
        default="dataset_version_01",
        description="Version Tag for dataset to save file with",
    )
    filelist = luigi.Parameter(
        description="txt file with input root files",
        significant=False,
        default="",
    )

    def store_parts(self):
        parts = super().store_parts()
        # append dataset-version to path
        parts += (self.dataset_version,)
        return parts


class TestDatasetDependency(object):
    test_dataset_version = luigi.Parameter(
        default="dataset_version_01",
        description="Version Tag for dataset to save file with",
    )
    test_filelist = luigi.Parameter(
        description="txt file with input root files",
        significant=False,
        default="",
    )

    def store_parts(self):
        parts = super().store_parts()
        # append dataset-version to path
        parts += (self.test_dataset_version,)
        return parts


class TrainingDependency(object):
    training_version = luigi.Parameter(
        default="training_version_01",
        description="Version Tag for training to save file with",
    )
    epochs = luigi.IntParameter(default=1)
    model_name = luigi.Parameter()
    n_threads = luigi.IntParameter(
        default=4, description="Number of threads to use for dataloader. Default: 4"
    )
    batch_size = luigi.IntParameter(default=1024)
    learning_rate = luigi.FloatParameter(default=1e-3)
    optimizer = luigi.Parameter(
        default="AdamW",
        description="The optimizer to minimize loss. Default: AdamW",
    )
    betas = law.CSVParameter(
        cls=luigi.FloatParameter,
        default=(0.95, 0.999),
        description="The comma-separated list of coefficients betas for optimizer (if applicable). Default: (0.95, 0.999)",
    )
    eps = luigi.FloatParameter(
        default=1e-6,
        description="The epsilon to use in optimizer. Default: 1e-6"
    )
    lr_scheduler = luigi.Parameter(
        default="epoch_lin_decay",
        description="The learning rate scheduler. Default: epoch_lin_decay",
    )
    lr_decay_factor = luigi.FloatParameter(
        default=1e-2,
        description="The factor to decrease the learning rate using scheduler. Default: 1e-2"
    )
    mixed_precision = luigi.BoolParameter(
        default=False,
        description="Decides whether to use Automatic Mixed Precision for training PyTorch models. Default: False",
    )
    use_torch_compile = luigi.BoolParameter(
        default=False,
        description="Decides whether to use torch.compile for acceleration of PyTorch training. Default: False",
    )
    
    def store_parts(self):
        parts = super().store_parts()

        parts += (self.training_version,)
        parts += (self.model_name,)
        parts += ("epochs_{0:d}".format(self.epochs),)

        return parts

class AttackDependency(object):

    attack = luigi.Parameter(
        default="nominal", description="Specify adversarial attack to use."
    )
    attack_magnitude = luigi.FloatParameter(
        default=0.0,
        description="Only use in combination with attack!=nominal. Set the magnitude for choosen attack.",
    )
    attack_iterations = luigi.IntParameter(
        default=1,
        description="Only use in combination with attack!=None and attack_magnitude!=0. Set the number of interations for choosen attack, if applicable.",
    )
    attack_individual_factors = luigi.BoolParameter(
        default=True,
        description="Decides whether individual attack magnitudes should be used per feature or not.",
    )
    attack_reduce = luigi.BoolParameter(
        default=True,
        description="Decides whether default values and integer values should be changed or not.",
    )
    attack_restrict_impact = luigi.FloatParameter(
        default=-1.0,
        description="Sets a maximal l-inf distance that each feature can be changed as a fraction of the nominal one. -1.0 means no restriction.",
    )

    def store_parts(self):
        parts = super().store_parts()

        parts += (self.attack,)
        if self.attack_magnitude > 0.0:
            parts += ("epsilon_{}".format(self.attack_magnitude),)
            parts += ("iterations_{}".format(self.attack_iterations),)

        return parts

class TestAttackDependency(object):

    test_attack = luigi.Parameter(
        default="nominal", description="Specify adversarial attack to use for testing."
    )
    test_attack_magnitude = luigi.FloatParameter(
        default=0.0,
        description="Only use in combination with attack!=nominal. Set the magnitude for choosen attack for testing.",
    )
    test_attack_iterations = luigi.IntParameter(
        default=1,
        description="Only use in combination with attack!=None and attack_magnitude!=0. Set the number of interations for choosen attack, if applicable, for testing.",
    )
    test_attack_individual_factors = luigi.BoolParameter(
        default=True,
        description="Decides whether individual attack magnitudes should be used per feature or not, for testing.",
    )
    test_attack_reduce = luigi.BoolParameter(
        default=True,
        description="Decides whether default values and integer values should be changed or not, for testing.",
    )
    test_attack_restrict_impact = luigi.FloatParameter(
        default=-1.0,
        description="Sets a maximal l-inf distance that each feature can be changed as a fraction of the nominal one. -1.0 means no restriction, for testing.",
    )

    def store_parts(self):
        parts = super().store_parts()

        parts += (f"test_attack_{self.test_attack}",)
        if self.attack_magnitude > 0.0:
            parts += ("test_epsilon_{}".format(self.test_attack_magnitude),)
            parts += ("test_iterations_{}".format(self.test_attack_iterations),)

        return parts
