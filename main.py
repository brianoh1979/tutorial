#!/usr/bin/env python
import wandb
wandb.init(project="firstproject")

import argparse
import os
from train import train_cnn


# let's define some default hyperparameter values
PROJECT_NAME = "fashion_mnist"
BATCH_SIZE = 32
DROPOUT = 0.2
EPOCHS = 20
L1_SIZE = 16
L2_SIZE = 32
HIDDEN_LAYER_SIZE = 128
LEARNING_RATE = 0.01


# parse all args and call the model
if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument(
    "-p",
    "--project_name",
    type=str,
    default=PROJECT_NAME,
    help="Main project name")

  parser.add_argument(
    "-b",
    "--batch_size",
    type=int,
    default=BATCH_SIZE,
    help="batch_size")

  parser.add_argument(
    "--dropout_mask",
    type=float,
    default=DROPOUT,
    help="dropout before dense layers")

  parser.add_argument(
    "-e",
    "--epochs",
    type=int,
    default=EPOCHS,
    help="number of training epochs (passes through full training data)")

  parser.add_argument(
    "--hidden_size",
    type=int,
    default=HIDDEN_LAYER_SIZE,
    help="hidden layer size")

  parser.add_argument(
    "-l1",
    "--L1_conv_size",
    type=int,
    default=L1_SIZE,
    help="layer 1 size")

  parser.add_argument(
    "-l2",
    "--L2_conv_size",
    type=int,
    default=L2_SIZE,
    help="layer 2 size")

  parser.add_argument(
    "-lr",
    "--learning_rate",
    type=float,
    default=LEARNING_RATE,
    help="learning rate")


  parser.add_argument(
    "-q",
    "--dry_run",
    action="store_true",
    help="Dry run (do not log to wandb)")  

  args = parser.parse_args()

  # easier testing--don't log to wandb if dry run is set
  if args.dry_run:
    os.environ['WANDB_MODE'] = 'dryrun'
 
  train_cnn(args)
