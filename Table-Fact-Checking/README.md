# Introduction

## Download the Dataset Split
[Training Set](https://github.com/wenhuchen/Table-Fact-Checking/blob/master/tokenized_data/train_examples.json)|[Validation Set](https://github.com/wenhuchen/Table-Fact-Checking/blob/master/tokenized_data/val_examples.json)|[Test Set](https://github.com/wenhuchen/Table-Fact-Checking/blob/master/tokenized_data/test_examples.json): The data has beeen tokenized and lower cased. You can directly use them to train/evaluation your model.

## Requirements
- Python 3.5
- Ujson 1.35
- Pytorch 1.2.0
- Pytorch_Pretrained_Bert 0.6.2 (Huggingface Implementation)
- Pandas
- tqdm-4.35
- TensorboardX
- unidecode
- nltk: wordnet, averaged_perceptron_tagger

## Direct Running: Without Preprocessing Data
### Latent Program Algorithm
0. Downloading the preprocessed data for LPA
Here we provide the data we obtained after preprocessing through the above pipeline, you can download that by running

```
  sh get_data.sh
```
1. Training the ranking model
Once we have all the training and evaluating data in folder "preprocessed_data_program", we can simply run the following command to evaluate the fact verification accuracy as follows:

```
  cd code/
  python model.py --do_train --do_val
```
2. Evaluating the ranking model
We have put our pre-trained model in code/checkpoints/, the model can reproduce the exact number reported in the paper:
```
  cd code/
  python model.py --do_test --resume
  python model.py --do_simple --resume
  python model.py --do_complex --resume
```
### Table-BERT
1. Training the verification model
```
  cd code/
  python run_BERT.py --do_train --do_eval --scan horizontal --fact [first/second]
```
2. Evaluating the verification model
```
  cd code/
  python run_BERT.py --do_eval --scan horizontal --fact [first/second] --load_dir YOUR_TRAINED_MODEL --eval_batch_size N
  or
  python run_BERT.py --do_eval --scan horizontal --fact first --load_dir outputs_fact-first_horizontal_snapshot/save_step_12500 --eval_batch_size 16
```

## Reference
Please cite the paper in the following format if you use this dataset during your research.
```
@inproceedings{2019TabFactA,
  title={TabFact : A Large-scale Dataset for Table-based Fact Verification},
  author={Wenhu Chen, Hongmin Wang, Jianshu Chen, Yunkai Zhang, Hong Wang, Shiyang Li, Xiyou Zhou and William Yang Wang},
  booktitle = {International Conference on Learning Representations (ICLR)},
  address = {Addis Ababa, Ethiopia},
  month = {April},
  year = {2020}
}
```

