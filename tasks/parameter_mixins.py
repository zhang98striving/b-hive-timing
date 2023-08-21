import luigi
import law


class DatasetDependency(object):
    dataset_version = luigi.Parameter(
        default="dataset_version_01", description="Version Tag for dataset to save file with"
    )

    def store_parts(self):
        parts = super().store_parts()
        # append dataset-version to path
        parts += (self.dataset_version,)
        return parts


class TrainingDependency(object):
    training_version = luigi.Parameter(
        default="training_version_01", description="Version Tag for training to save file with"
    )

    def store_parts(self):
        parts = super().store_parts()
        # append dataset-version to path
        parts += (self.training_version,)
        return parts
