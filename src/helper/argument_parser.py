"""Contains functions to parse different command-line arguments."""

import os
import random
from typing import List, Tuple


def get_subject_list(string: str) -> List[str]:
    """Subject argument can be a path to the file which contains one subject per line,
    or a list of subjects separated by commas."""

    if os.path.exists(string) and os.path.isfile(string):
        with open(string) as f:
            subjects = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    else:
        subjects = [name.strip() for name in string.split(',') if name.strip()]

    return subjects


def split_list_into_sublists(flat_list: List, num_sublists: int) -> List[List]:
    flat_list = flat_list.copy()

    sublists = []

    while len(flat_list) > 0:
        for i in range(num_sublists):
            if len(flat_list) == 0:
                break
            if i >= len(sublists):
                sublists.append([flat_list.pop()])
            else:
                sublists[i].append(flat_list.pop())

    return sublists


def split_subjects_to_gpus(subject_list: List, gpus_arg: str) -> Tuple[List[int], List[List]]:
    gpus: List[int] = [int(token) for token in gpus_arg.split(",")]

    subject_list = subject_list.copy()
    random.shuffle(subject_list)
    random.shuffle(subject_list)

    # split subjects into different GPUs
    subject_batches = []
    while len(subject_list) > 0:
        for ind in range(len(gpus)):
            if len(subject_list) == 0:
                break
            if ind >= len(subject_batches):
                subject_batches.append([subject_list.pop()])
            else:
                subject_batches[ind].append(subject_list.pop())

    gpus = [gpu for ind, gpu in enumerate(gpus) if ind < len(subject_batches)]

    return gpus, subject_batches
