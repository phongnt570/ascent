import argparse
import configparser
import logging
import socket

host_name = socket.gethostname()
logging.basicConfig(level=logging.INFO,
                    format='[%(process)d] [%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
                    datefmt='%d-%m %H:%M:%S',
                    # filename='../log_{}.txt'.format(host_name),
                    # filemode='a'
                    )
logger = logging.getLogger(__name__)

# get config file
parser = argparse.ArgumentParser(prog="DESCENT extraction pipeline")
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--subject", type=str)
parser.add_argument("--from_m", type=int)
parser.add_argument("--to_m", type=int)
parser.add_argument("--gpu", type=str)
args = parser.parse_args()

logger.info("Loading config from {}".format(args.config))
config = configparser.ConfigParser()
config.read(args.config)

# use custom gpu if set
if args.gpu is not None:
    config["default"]["gpu"] = args.gpu

# set up the UniversalFilePathHandler
from filepath_handler import UniversalFilePathHandler  # noqa: E402

UniversalFilePathHandler.set_up(config=config)

# the following packages must be loaded AFTER setting the UniversalFilePathHandler
from helper.argument_parser import get_subject_list  # noqa: E402
from pipeline.pipeline import Pipeline  # noqa: E402

# construct pipeline
pipeline = Pipeline(config)

# input subject
if args.subject is not None:
    input_subject = args.subject
elif "subject" in config["default"]:
    input_subject = config["default"]["subject"]
else:
    input_subject = input("Enter subjects: ")
subject_list = get_subject_list(input_subject)

logger.info(
    f"Running pipeline for {len(subject_list)} "
    f"subjects: [{', '.join(subject for subject in subject_list[:5])}{',...' if len(subject_list) > 5 else ''}]")

# run which modules?
if args.from_m is None or args.to_m is None:
    pipeline.print_modules()
    from_module = int(input("From module: "))
    if from_module == len(pipeline) - 1:
        to_module = from_module
    else:
        to_module = int(input("  To module: "))
else:
    from_module = args.from_m
    to_module = args.to_m

# run
pipeline.run(subject_list=subject_list, from_module=from_module, to_module=to_module)
