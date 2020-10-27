import logging
from pathlib import Path
from typing import Union

import numpy as np
import torch
from pandas import DataFrame
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


def flat_accuracy(logits, labels):
    """Function to calculate the accuracy of our predictions vs labels"""

    pred_flat = np.argmax(logits, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def train(model, train_loader: DataLoader, device, lr: float, eps: float, epochs: int, print_every: int,
          val_loader=None):
    logger.info("Let the training begin...")
    optimizer = AdamW(model.parameters(), lr=lr, eps=eps)
    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_loader) * epochs
    # Create the learning rate decay scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)
    last_loss = 100000
    for epoch_i in range(0, epochs):
        # TRAINING
        logger.info('Epoch {:} / {:}'.format(epoch_i + 1, epochs))

        model.train()  # change to 'training' mode
        total_loss = 0
        for step, batch in enumerate(train_loader):
            # Progress update.
            if (step + 1) % print_every == 0 or (step + 1) == len(train_loader):
                logger.info('Batch {:>5,}  of  {:>5,}.'.format(step + 1, len(train_loader)))

            b_input_ids, b_token_type_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch)
            model.zero_grad()
            outputs = model(b_input_ids,
                            token_type_ids=b_token_type_ids,
                            attention_mask=b_input_mask,
                            labels=b_labels)

            loss = outputs[0]  # labels is provided, hence outputs = (loss, logits)
            total_loss += loss.item()
            loss.backward()

            # Clip the norm of the gradients to 1.0. This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Update weights
            optimizer.step()
            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        cur_loss = total_loss / len(train_loader)
        logger.info("Average training loss: {:.2f}".format(cur_loss))

        # VALIDATION if validation set is provided
        if val_loader is not None:
            logger.info("Running Validation...")
            eval_accuracy = evaluate(model=model, device=device, loader=val_loader)
            logger.info("Validation Accuracy: {:.2f}%".format(eval_accuracy * 100))

        if last_loss - cur_loss <= 1e-3:
            break

        last_loss = cur_loss

    logger.info("Training complete!")


def evaluate(model, device, loader, has_label: bool = True, output_dir: Union[str, Path] = None,
             print_real_label: bool = True):
    model.eval()

    predictions = []
    num_true_pred, num_samples = 0, 0

    for batch in loader:
        cuda_batch = tuple(t.to(device) for t in batch)

        b_input_ids = cuda_batch[0]
        b_token_type_ids = cuda_batch[1]
        b_input_mask = cuda_batch[2]

        with torch.no_grad():
            outputs = model(b_input_ids,
                            token_type_ids=b_token_type_ids,
                            attention_mask=b_input_mask)

        logits = outputs[0]  # labels is not provided, hence outputs = (logits, )
        predictions.extend(torch.softmax(logits, dim=1).tolist())

        if has_label:
            logits = logits.detach().cpu().numpy()
            label_ids = batch[-1].to('cpu').numpy()
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            num_true_pred += tmp_eval_accuracy * len(b_input_ids)
            num_samples += len(b_input_ids)

    # Print results to file
    if output_dir is not None:
        outfile = output_dir / 'test_output.csv'
        logger.info('Writing test results to file {}'.format(outfile.absolute()))

        predicted = []
        probability = []
        for probs in predictions:
            predicted.append(model.config.id2label[np.argmax(probs)] if print_real_label else np.argmax(probs))
            probability.append(np.max(probs))

        df: DataFrame = loader.dataset.df
        df["predicted"] = predicted
        df["confidence"] = probability
        df.to_csv(outfile, index=False, float_format="%.3f")

    # Return test accuracy if ground-truth labels are provided
    if has_label:
        return num_true_pred / num_samples
