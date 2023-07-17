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


## Detailed description of the tasks
### DatasetConstructorTask
This task relies on [dataset/dataset.py](https://gitlab.cern.ch/cms-btv/b-hive/-/blob/law/dataset/dataset.py) to read in [PFNano](https://github.com/cms-jet/PFNano) or [DeepNTuple](https://github.com/CMSDeepFlavour/DeepNTuples) files and store them in the numpy file format.

After specifying in the `sample_dict`what files to process, a coffea processor is started to run over the input files. Losse cuts are applied to the data (10<= p_T <=2000 and -2.5<= eta <=2.5), before the features needed for a training of the DeepJet model are extracted and the truth information is set to
```
0 for b jets,
1 for bb jets,
2 for leptonic b jets,
3 for c jets,
4 for uds jets and 
5 for g jets
```
in the output.

To not exhaust the available RAM of your machine, the files are processed and saved in chunks. The chunk size is a parameter you can adjust to the resources and capabilities of your system. The 
The processed files are stored in the directory defined in the configuration chapter. Furthermore, a text file is stored in the same directory containing all paths to the newly generated files for easer loading later.

For reweighting the inputs of the model later in the training step, one histogram per flavour mentioned above is filed and save in the same directory as well.
The respective binning is
```
p_t = [10, 25, 30, 35, 40, 45, 50, 60, 75, 100, 125, 150, 175, 200, 250, 300, 400, 500, 600, 2000] and 
eta = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.5, 1, 1.5, 2.0, 2.5] 
```
as in the original DeepJet implementation.
