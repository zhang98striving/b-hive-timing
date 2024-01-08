import luigi


class DatasetDependency(object):
    dataset_version = luigi.Parameter(
        default="dataset_version_01",
        description="Version Tag for dataset to save file with",
    )
    training_filelist = luigi.Parameter(
        description="txt file with input root files for training.",
        significant=False,
        default="",
    )
    test_filelist = luigi.Parameter(
        description="txt file with input root files for testing.",
        default="",
    )

    def store_parts(self):
        parts = super().store_parts()
        # append dataset-version to path
        parts += (self.dataset_version,)
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
    attack = luigi.Parameter(default="nominal", description="Specify adversarial attack to use.")
    attack_magnitude = luigi.FloatParameter(default=0.0, description="Only use in combination with attack!=None. Set the magnitude for choosen attack.")
    attack_iterations = luigi.IntParameter(default=1, description="Only use in combination with attack!=None and attack_magnitude!=0. Set the number of interations for choosen attack, if applicable.")

    def store_parts(self):
        parts = super().store_parts()

        parts += (self.training_version,)
        parts += (self.model_name,)
        parts += ("epochs_{0:d}".format(self.epochs),)
        parts += (self.attack,)
        if self.attack_magnitude > 0.0:
            parts += ("epsilon_{}".format(self.attack_magnitude),)
            parts += ("iterations_{}".format(self.attack_iterations),)

        return parts
