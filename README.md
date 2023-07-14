# Welcome to b-hive (work in progress)
This framework is a modernised version of [DeepJet](https://github.com/DL4Jets/DeepJet) and [DeepJetCore](https://github.com/DL4Jets/DeepJetCore), taking advantage of modern packages like [PyTorch](https://pytorch.org), [numpy](https://numpy.org), [awkward](https://awkward-array.org/doc/main/), [coffea](https://coffeateam.github.io/coffea/), [uproot](https://uproot.readthedocs.io/en/latest/) and [law](https://law.readthedocs.io/en/latest/).
You will be able to read in ROOT files, extract features needed for a training of the DeepJet model, perform a training, make predictions using a trained model and evaluate the output/performance.


## Setup
1) Clone the repository to your machine.

2) All necessary packages and dependencies for a Linux system are provided by the conda environment file `env.txt`. For more information on how to install conda and how to create the provided environment, have a look at the [conda user guide](https://conda.io/projects/conda/en/latest/user-guide/index.html).

## Configuration
1) Everytime you want to use the framework, you need to source `setup.sh` by executing
```
source setup.sh
```
in the shell.

2) When you are using the framework for the first time or to index new tasks in the framework you will need to execute
```
law index
```
in the shell.

3) Adjust the path in [BaseTask.py](https://gitlab.cern.ch/cms-btv/b-hive/-/blob/law/BaseTask.py#L28) to tell the framework where it should store its output. Analogously, you have to adjust the path in [dataset/dataset.py](https://gitlab.cern.ch/cms-btv/b-hive/-/blob/law/dataset/dataset.py#L34). We plan to make this step interactive and more dynamic in the future, but as this framework is still under development, it has to be done by hand.

## Usage
To peform a task simply execute
```
law run <TASK_NAME>
```
in the shell. The currently available tasks are
- `DatasetConstructorTask`: reads in ROOT files and stores the relevant branches in numpy files,
- `TrainingTask`: performes a training with the previously generated numpy files,
- `InferenceTask`: performes a prediction using the previously trained model and
- `PlottingTask`: generates ROC curves using the output of the prediction
  
Due to the usage of law, the framework will check if previous steps in the chain have already been completed and automatically execute them if necessary or fall back on intermediate results to execute the requested task.