import luigi


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
        default=4, description="Number of threads to use for dataloader."
    )
    batch_size = luigi.IntParameter(default=1000)
    learning_rate = luigi.FloatParameter(default=0.001)

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
