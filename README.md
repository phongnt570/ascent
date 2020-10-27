# ASCENT: Advanced Semantics for Commonsense Knowledge Extraction

## Introduction
ASCENT is a pipeline for extracting and consolidating commonsense
knowledge from the world wide web.
ASCENT is capable of extracting facet-enriched assertions, for
example, `lawyer; represents; clients; [LOCATION] in courts` or
`elephant; uses; its trunk; [PURPOSE] to suck up water`.
A web interface of the ASCENT knowledge base for 10,000 popular
concepts can be found at https://ascentkb.herokuapp.com/.

## Prerequisites
### Setting up environment
You need python3.7+ to run the pipeline.

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

### RoBERTa models
Download our pretrained models for triple clustering and
facet type labeling:
```shell script
TODO
```

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
Enter subjects: lion.n.01,lynx.n.01,elephant.n.01
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
`ascent/output/kb/<subject>/final.json`.

## Advance configurations
An example config file is `config.ini`. The missing fields are Bing
API-related.

- __[default]__
    - `res_dir`: resource folder
    - `output`: output folder
    - `gpu`: list of comma-separated GPUs to be used. `-1` means CPU will
    be used.
