# TaBERT

## Extract/Preprocess Table Corpora from CommonCrawl and Wikipedia

### Prerequisite

The following libraries are used for data extraction:

* [`jnius`](https://pyjnius.readthedocs.io/en/stable/)
* [`info.bliki.wiki`](https://bitbucket.org/axelclk/info.bliki.wiki/wiki/Mediawiki2HTML)
* wikitextparser
* Beautiful Soup 4
* Java Wikipedia code located at `contrib/wiki_extractor`
    * It compiles to a `.jar` file using maven, which is also included in the folder
* `jdk` 12+

### Installation
Fist, you need to install Java JDK. 
Then use the following command to install necessary Python libraries. 

```
pip install -r preprocess/requirements.txt
python -m spacy download en_core_web_sm
```

### Training Table Corpora Extraction

#### CommonCrawl WDC Web Table Corpus 2015

Details of the dataset could be found at [here](http://webdatacommons.org/webtables/2015/downloadInstructions.html).
We used the English relational tables split, which could be downloaded at [here](http://data.dws.informatik.uni-mannheim.de/webtables/2015-07/englishCorpus/compressed/).

The script to preprocess the data is at `scripts/preprocess_commoncrawl_tables.sh`.
The following command pre-processes [a sample](http://data.dws.informatik.uni-mannheim.de/webtables/2015-07/sample.gz) 
of the whole WDC dataset. To preprocess the whole dataset, simply replace 
the `input_file` with the root folder of the downloaded tar ball files.
```shell script
mkdir -p data/datasets
wget http://data.dws.informatik.uni-mannheim.de/webtables/2015-07/sample.gz -P data/datasets
gzip -d < data/datasets/sample.gz > data/datasets/commoncrawl.sample.jsonl

python \
    -m preprocess.common_crawl \
    --worker_num 12 \
    --input_file data/datasets/commoncrawl.sample.jsonl \
    --output_file data/preprocessed_data/common_crawl.preprocessed.jsonl
```

#### Wikipedia Tables

The script to extract Wiki tables is at `scripts/extract_wiki_tables.sh`. It demonstrates
extracting tables from a sampled Wikipedia dump. Again, you may need the full Wikipedida dump
to perform data extraction.

### Notes for Table Extraction

**Extract Tables from Scraped HTML Pages** 
Most code in `preprocess.extract_wiki_data` is for extracting surrounding 
natural language sentences around tables. If you are only interested in 
extracting tables (e.g., from scraped Wiki Web pages), you could just use 
the `extract_table_from_html` function. See the comments for more details. 

## Training Data Generation

This section documents how to generate training data for masked language modeling training 
from extracted and preprocessed tables. 

The scripts to generate training data for our vanilla `TaBERT(K=1)` and vertical attention
`TaBERT(k=3)` models are `utils/generate_vanilla_tabert_training_data.py` and 
`utils/generate_vertical_tabert_training_data.py`. They are heavily optimized for generating 
data in parallel in a distributed compute environment, but could still be used locally. 

The following script assumes you have concatenated
the `.jsonl` files obtained from running the data extraction scripts on Wikipedia and CommonCrawl
corpora and saved to `data/preprocessed_data/tables.jsonl`

```shell script
cd data/preprocessed_data
cat common_crawl.preprocessed.jsonl wiki_tables.jsonl > tables.jsonl
```

The following script generates training data for a vanilla `TaBERT(K=1)` model:
```shell script
output_dir=data/train_data/vanilla_tabert
mkdir -p ${output_dir}

python -m utils.generate_vanilla_tabert_training_data \
    --output_dir ${output_dir} \
    --train_corpus data/preprocessed_data/tables.jsonl \
    --base_model_name bert-base-uncased \
    --do_lower_case \
    --epochs_to_generate 15 \
    --max_context_len 128 \
    --table_mask_strategy column \
    --context_sample_strategy concate_and_enumerate \
    --masked_column_prob 0.2 \
    --masked_context_prob 0.15 \
    --max_predictions_per_seq 200 \
    --cell_input_template 'column|type|value' \
    --column_delimiter "[SEP]"
```

The following script generates training data for a `TaBERT(K=3)` model with 
vertical self-attention:
```shell script
output_dir=data/train_data/vertical_tabert
mkdir -p ${output_dir}

python -m utils.generate_vertical_tabert_training_data \
    --output_dir ${output_dir} \
    --train_corpus data/preprocessed_data/tables.jsonl \
    --base_model_name bert-base-uncased \
    --do_lower_case \
    --epochs_to_generate 15 \
    --max_context_len 128 \
    --table_mask_strategy column \
    --context_sample_strategy concate_and_enumerate \
    --masked_column_prob 0.2 \
    --masked_context_prob 0.15 \
    --max_predictions_per_seq 200 \
    --cell_input_template 'column|type|value' \
    --column_delimiter "[SEP]"
```

**Parallel Data Generation** The script has two additional arguments, `--global_rank` and 
`--world_size`. To generate training data in parallel using `N` processes, just fire up 
`N` processes with the same set of arguments and `--world_size=N`. The argument `--global_rank` 
is set to `[1, 2, ..., N]` for each process.

## Model Training
```shell script
mkdir -p data/runs/vanilla_tabert

python train.py \
    --task vanilla \
    --data-dir data/train_data/vanilla_tabert \
    --output-dir data/runs/vanilla_tabert \
    --table-bert-extra-config '{}' \
    --train-batch-size 8 \
    --gradient-accumulation-steps 32 \
    --learning-rate 2e-5 \
    --max-epoch 10 \
    --adam-eps 1e-08 \
    --weight-decay 0.0 \
    --fp16 \
    --clip-norm 1.0 \
    --empty-cache-freq 128
```

The following script shows training a `TaBERT(k=3)` model with vertical self-attention:
```shell script
mkdir -p data/runs/vertical_tabert

python train.py \
    --task vertical_attention \
    --data-dir data/train_data/vertical_tabert \
    --output-dir data/runs/vertical_tabert \
    --table-bert-extra-config '{"base_model_name": "bert-base-uncased", "num_vertical_attention_heads": 6, "num_vertical_layers": 3, "predict_cell_tokens": true}' \
    --train-batch-size 8 \
    --gradient-accumulation-steps 64 \
    --learning-rate 4e-5 \
    --max-epoch 10 \
    --adam-eps 1e-08 \
    --weight-decay 0.01 \
    --fp16 \
    --clip-norm 1.0 \
    --empty-cache-freq 128
```
Pretrained Model can be found [here](https://drive.google.com/drive/folders/13jzt8Q0FlTRqHfG-mwuTHOqCfsyHamh3?usp=share_link)
