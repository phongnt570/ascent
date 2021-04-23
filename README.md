# ASCENT: Advanced Semantics for Commonsense Knowledge Extraction

## Introduction
ASCENT is a pipeline for extracting and consolidating commonsense
knowledge from the world wide web.
ASCENT is capable of extracting facet-enriched assertions, for
example, `lawyer; represents; clients; [LOCATION] in courts` or
`elephant; uses; its trunk; [PURPOSE] to suck up water`.
A web interface of the ASCENT knowledge base for 10,000 popular
concepts can be found at https://ascent.mpi-inf.mpg.de/.

## Prerequisites
### Setting up environment
You need __python3.7+__ to run the pipeline.

First, create and activate a virtual environment using your
favourite platform, e.g., `python3-venv`:
```shell script
python -m venv .env
source .env/bin/activate
```

Then, install required packages:
```shell script
pip install -r requirements.txt
```

Next, you need to download the following SpaCy model:
```shell script
python -m spacy download en_core_web_md
```

Then, download the `wordnet` corpus for the `nltk` package:
```shell script
python -c 'import nltk; nltk.download("wordnet")'
```

### RoBERTa models
Download our pretrained models for triple clustering and
facet type labeling from https://nextcloud.mpi-klsb.mpg.de/index.php/s/s2ELgPgC5LEGEFp
then extract it to the project's root folder.

### Bing API Key
Edit the file `config.ini` and provide your __Bing API Key__ and
__Bing Search Custom Config__ under the section `[bing_search]`.

## Usage
To run the ASCENT pipeline, navigate to the `src/` folder and execute
the `main.py` script:
```shell script
cd src/
python main.py --config ../config.ini
```

You will be asked to fill in subject(s) which should be __WordNet__
concepts. You can provide a single subject:
```
Enter subjects: lion.n.01
```
or a list of comma-separated subjects:
```
Enter subjects: lion.n.01,lynx.n.02,elephant.n.01
```
or path to a file containing one subject per line:
```
Enter subjects: /path/to/your/subjects.txt
```

Then, enter indices of the modules you want to execute:
```
[0] Bing Search
[1] Crawl articles
[2] Filter irrelevant articles
[3] Extract knowledge
[4] Cluster similar triples
[5] Label facets
[6] Group similar facets
```
For example, to run the complete pipeline:
```
From module: 0
  To module: 6
```

Final results will be written to
`output/kb/<subject>/final.json`.
Intermediate results of every module can be found in the `output` folder as well. 

## Configurations
An example config file is the `config.ini` file.
The missing fields are the Bing API-related ones.
You can find references of the config fields in the following:

- __[default]__
  - `res_dir`: resource folder
  - `output`: output folder
  - `gpu`: list of comma-separated GPUs to be used. `-1` means CPU will
    be used. E.g., `gpu = 0,3` means that we'll use the 0-th and 3-rd GPUs
    of the machine.

- __[bing_search]__
  - `subscription_key`: __Bing API subscription key__ <span style="color:red">(required)</span>
  - `custom_config`: __Bing API custom config__ <span style="color:red">(required)</span>
  - `num_urls`: number of URLs to be fetched by the Bing API
  - `host` = *api.cognitive.microsoft.com*
  - `path` = */bingcustomsearch/v7.0/search*
  - `overwrite`: (true|false) indicates that when result of this module are already
    found in the output folder, overwrite it or not
  - `num_processes`: number of processors for this module

- __[article_grab]__
  - `num_crawlers`: number of parallel crawlers, each crawler works with one subject at a time
  - `processes_per_crawler`: number of processors per crawlers
  - `overwrite`: (true|false) indicates that when result of this module are already
    found in the output folder, overwrite it or not

- __[filter]__
  - `num_processes`: number of processors for this module
  - `overwrite`: (true|false) indicates that when result of this module are already
    found in the output folder, overwrite it or not

- __[extraction]__
  - `doc_threshold`: document cosine-similarity threshold. Documents lower than this threshold
    will be filtered out (default: 0.55)
  - `num_processes`: number of processors for this module
  - `overwrite`: (true|false) indicates that when result of this module are already
    found in the output folder, overwrite it or not

- __[triple_clustering]__
  - `model`: path to the triple clustering model
  - `threshold`: threshold for the HAC algorithm (default: 0.005)
  - `batch_size`: size of triple pair batch to be processed at a time (default: 1024)
  - `overwrite`: (true|false) indicates that when result of this module are already
    found in the output folder, overwrite it or not

- __[facet_labeling]__
  - `model`: path to the facet labeling model
  - `batch_size` = 1024
  - `overwrite`: (true|false) indicates that when result of this module are already
    found in the output folder, overwrite it or not

- __[facet_grouping]__
  - `num_processes`: number of processors for this module
  - `overwrite`: (true|false) indicates that when result of this module are already
    found in the output folder, overwrite it or not
