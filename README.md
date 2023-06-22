# b-hive

This is a developement repository for the new b-tagging training framework, which is still heavily work-in-progress. This implies some elements being hardcoded at this point (such as dataset paths, network hyperparameters etc.) and several features are still being worked on actively. Feedback is really apreciated!

## Setup
------------

First you will need to initilize the conda environment. This can be done by 

```
conda env create --name env_name --file env.yml
``` 

with the environment file directly located in the root directory. Once you are in the right environment on the target machine,  you first need to source the provided ```setup.sh``` file and call ```law index --verbose```  to scan the repository for law tasks. Note that you need to adjust the paths to the dataset in the ```DatasetConstructorTask``` class in ```dataset/dataset.py``` manually. Currently PFNano and NTuple dataset formats are supported exclusively; at this point, you have to comment and uncomment the corresponding lines in ```dataset.py``` to switch between the input formats. Expect some more ergonomic features in the future!

Having done the above full training run can then be started via:

    law run DeepJetRun