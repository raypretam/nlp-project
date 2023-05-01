# Table-BERT

## Install

## To run the Experiments

   Run the command (in default config) `python train.py --task [TASK NAME] --data-dir [PATH] --output-dir [PATH]`</br ></br >
  The details of some of the parameters is given below:</br >
  1. `--task` : The type of task. Choices are: `vanilla`(default) and `vertical attention`
  2. `--cpu` : Default value is False. 
  3. `--base-model-name` : Default model is `bert-base-uncased`. Other available models are `bert-large-uncased`, `bert-base-cased`, `bert-base-multilingual` and `bert-base-chinese`.
  4. `--train-batch-size` : Default value is 32. </br ></br >
Please visit the `train.py` file to get details about other parameters.
