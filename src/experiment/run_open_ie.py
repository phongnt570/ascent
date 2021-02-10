import argparse
import configparser
import json

import spacy

import time

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str)
parser.add_argument('--cli', action='store_true')
parser.add_argument('--input', type=str)
parser.add_argument('--output', type=str)
parser.add_argument('--gpu', type=int, default=-1)
args = parser.parse_args()

print("Loading config from {}".format(args.config))
config = configparser.ConfigParser()
config.read(args.config)

cuda_device = args.gpu
model_path = config["facet_labeling"]["model"]
batch_size = config["facet_labeling"].getint("batch_size")

# set up the UniversalFilePathHandler
from filepath_handler import UniversalFilePathHandler  # noqa: E402

UniversalFilePathHandler.set_up(config=config)

# the following packages must be loaded AFTER setting the UniversalFilePathHandler
from helper.constants import SPACY_MODEL_NAME  # noqa: E402
from extraction.stuffie import run_stuffie  # noqa: E402
from facet_labeling.facet_labeling_factory import FacetLabelingFactory  # noqa: E402


def main():
    if args.cli:
        cli()
    else:
        if args.input is not None and args.output is not None:
            extract_file(args.input, args.output)
        else:
            print("Please provide input and output files!")


def extract_file(input_path, output_path):
    nlp = load_spacy()
    labeler = load_labeler()

    print("Extracting...")
    with open(input_path) as input_file:
        lines = [line.strip() for line in input_file if line.strip()]

    assertions_per_line = []
    for line in lines:
        assertions_per_line.append(run_stuffie(line, nlp, do_eval=True))

    results_per_line = [[a.to_dict(include_source=True) for a in apl] for apl in assertions_per_line]
    all_assertions = [a for apl in results_per_line for a in apl]
    labeler.label(all_assertions)
    for assertion in all_assertions:
        assertion.pop("source")

    with open(output_path, "w+") as output_file:
        for i, group in enumerate(results_per_line):
            result = {
                "text": lines[i],
                "assertions": group
            }
            output_file.write(json.dumps(result, ensure_ascii=False, sort_keys=False))
            output_file.write("\n")

    print("Finished!")


def cli():
    nlp = load_spacy()
    labeler = load_labeler()

    while True:
        print()
        input_text = input("ENTER TEXT ('q' to quit): ")
        if input_text == 'q':
            return

        assertions = run_stuffie(input_text, nlp, do_eval=True)

        results = [assertion.to_dict(include_source=True) for assertion in assertions]
        labeler.label(results)

        for assertion in results:
            assertion.pop("source")

        print(json.dumps(results, indent=1))

        # print("####################### ORIGINAL ########################")
        # print(json.dumps([assertion.to_dict(simplify=False) for assertion in results], indent=1))

        # print("###################### SIMPLIFIED #######################")
        # print(json.dumps([assertion.to_dict(simplify=True) for assertion in results], indent=1))

        # print("#########################################################")


def load_spacy():
    print("Loading SpaCy model...")
    start = time.time()
    nlp = spacy.load(SPACY_MODEL_NAME)
    end = time.time()
    print("Loading finished [{:.2f}s].".format((end - start)))
    return nlp


def load_labeler():
    print("Loading facet labeling model...")
    start = time.time()
    labeler = FacetLabelingFactory(model_path=model_path,
                                   device=f"cuda:{cuda_device}" if cuda_device >= 0 else "cpu",
                                   batch_size=batch_size)
    end = time.time()
    print("Loading finished [{:.2f}s].".format((end - start)))
    return labeler


if __name__ == '__main__':
    main()
