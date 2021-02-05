import argparse
import logging
import os
import socket
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig

from facet_labeling.facet_data_loader import FacetDataset, label2id, id2label
from filepath_handler import dir_path
from helper.roberta_interface import train, evaluate

host_name = socket.gethostname()
logging.basicConfig(level=logging.INFO,
                    format='[%(process)d] [%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
                    datefmt='%d-%m %H:%M:%S',
                    filename='../logs/log_{}.txt'.format(host_name),
                    )
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--data_dir', type=dir_path, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--max_seq_length', type=int, default=32)
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--eps', type=float, default=1e-8)
parser.add_argument('--print_every', type=int, default=10)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--do_train', action='store_true')
parser.add_argument('--do_eval', action='store_true')
parser.add_argument('--do_predict', action='store_true')
args = parser.parse_args()


def main():
    # Obtain data files
    data_dir = Path(str(args.data_dir))
    train_file = data_dir / 'train.csv'
    dev_file = data_dir / 'dev.csv'
    test_file = data_dir / 'test.csv'

    # Obtain device
    device = 'cpu'
    if 0 <= args.gpu < torch.cuda.device_count():
        device = torch.device('cuda:{}'.format(args.gpu))

    # Create output directory if needed
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_dir = Path(args.output_dir)

    # Load RoBERTa
    logger.info("Loading RoBERTa model...")
    tokenizer = RobertaTokenizer.from_pretrained(args.model)

    if args.do_train:
        config = RobertaConfig.from_pretrained(args.model)

        # change number of labels
        config.num_labels = len(label2id)
        config.label2id = label2id
        config.id2label = id2label

        model = RobertaForSequenceClassification.from_pretrained(args.model, config=config)

        tokenizer.add_special_tokens({'additional_special_tokens': ['[pred]', '[obj]', '[facet]']})
        model.resize_token_embeddings(len(tokenizer))  # Notice: resize_token_embeddings expect to receive
        # the full size of the new vocabulary, i.e. the length of the tokenizer.
    else:
        model = RobertaForSequenceClassification.from_pretrained(args.model)

    model.to(device)

    if args.do_predict:
        logger.info("Creating prediction data loaders")
        test_set = FacetDataset(filename=test_file, tokenizer=tokenizer)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers)

        logger.info("Running classification for {} samples...".format(len(test_set)))
        eval_accuracy = evaluate(model=model,
                                 device=device,
                                 loader=test_loader,
                                 has_label=test_set.has_label,
                                 output_dir=output_dir,
                                 print_real_label=True)

        if test_set.has_label:
            logger.info("Test Accuracy: {:.2f}%".format(eval_accuracy * 100))

    elif args.do_train:
        logger.info("Creating training data loaders")
        train_set = FacetDataset(filename=train_file, tokenizer=tokenizer, maxlen=args.max_seq_length)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

        val_loader = None
        if args.do_eval:
            logger.info("Creating validation data loaders")
            val_set = FacetDataset(filename=dev_file, tokenizer=tokenizer, maxlen=args.max_seq_length)
            val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.num_workers)

        # Train
        train(model=model,
              train_loader=train_loader,
              val_loader=val_loader,
              device=device,
              lr=args.lr,
              eps=args.eps,
              epochs=args.epochs,
              print_every=args.print_every)

        # Save model
        logger.info("Saving model to {}".format(output_dir))
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed training

        model_to_save.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))


if __name__ == '__main__':
    main()
