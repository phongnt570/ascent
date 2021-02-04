import argparse
from os import path

parser = argparse.ArgumentParser()
parser.add_argument("--subject", type=str, required=True)
parser.add_argument("--kb", type=str, default="../output/kb/")
parser.add_argument("--output", type=str, required=True)
args = parser.parse_args()

subject_file = args.subject
kb_dir = args.kb

with open(subject_file) as f:
    subjects = [line.strip() for line in f if line.strip()]

lines = []
for subject in subjects:
    filename = path.join(kb_dir, subject, "final.json")

    with open(filename) as f:
        lines.append(f.readline())

with open(args.output, "w+") as f:
    for line in lines:
        f.write(line)
        f.write("\n")
