#!/usr/bin/env bash

python3 train.py --adadelta --trainRoot './data/train.lmdb' --valRoot './data/val.lmdb' --ngpu 1 --cuda --valInterval 500