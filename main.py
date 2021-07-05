import os
import math
import pickle
from __future__ import print_function, division
from IPython.core.debugger import set_trace

from DeepPurpose import utils
from DeepPurpose import DTI as models

import rdkit
from rdkit import Chem
from rdkit import rdBase
from rdkit.six import iteritems
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import CalcTPSA, CalcNumHBD, CalcNumHBA
from rdkit.Chem.Descriptors import ExactMolWt, MolLogP, MolMR

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pl_bolts.callbacks import PrintTableMetricsCallback

import torch
from torch import nn
import torch.multiprocessing
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

from datasets import load_dataset
from transformers import get_linear_schedule_with_warmup, AdamW, GPT2Tokenizer, GPT2Config, AutoModelWithLMHead
import pandas as pd
import numpy as np
from sklearn import preprocessing
from google.cloud import storage
from six.moves import urllib

from feedback import MoleculeValidationAndFeedback
from sa_score import SyntheticAccessibilityScore
from model import SmilesModel
from data_module import SmilesDataModule
from settings import max_length, learning_rate, batch_size, max_epochs, test, to_grab_checkpoint, data_index

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/htr1-310816-c727b825505c.json"
print('Finished Imports.\nVersion Details:\nPytorch: \t{}\nLightning:\t{}\nRdKit:    \t{}\n'.format(torch.__version__,
                                                                                                    pl.__version__,
                                                                                                    rdkit.__version__))
rdBase.DisableLog('rdApp.error')




"""SETTING LIGHTNING CALLBACKS"""
checkpoint_callback = ModelCheckpoint(dirpath = os.getcwd(), filename = '{epoch}-{val_loss:.2f}')

early_stopping = EarlyStopping(monitor = 'val_loss')

table_metrics = PrintTableMetricsCallback()




"""CREATING DATASET MODULE, MODEL AND TRANSFORMER FEEDBACK CLASSES"""
data = SmilesDataModule(batch_size = batch_size, 
                        max_length = max_length,
                        data_index = data_index,
                        to_grab_checkpoint = to_grab_checkpoint)


feedback = MoleculeValidationAndFeedback(sa_score = SyntheticAccessibilityScore(), 
                                        molecules = data.all_raw_smiles,
                                        tokenizer = data.tokenizer,
                                         max_length = max_length)


model = SmilesTransformerModel(batch_size = batch_size, 
                    max_length = max_length, 
                    learning_rate = learning_rate, 
                    tokenizer = data.tokenizer,
                    smiles_data_module = data,
                    to_grab_checkpoint = to_grab_checkpoint,
                    feedback = feedback)
  



"""SETTING TRAINER AND STARTING PYTORCH LIGHTNING TRAINING"""
trainer = pl.Trainer(
    gpus=torch.cuda.device_count(),
    precision=16, # PRECISION IS SET TO 16 FOR FASTER TRAINING. THE OTHER OPTION IS 32
    profiler='simple', # THE OTHER OPTION IS ADVANCED
    max_epochs=max_epochs, 
    callbacks=[checkpoint_callback, 
                early_stopping, 
                table_metrics], 
    limit_train_batches=0.1 if test else 1.0, # SETTING SMALLER BATCHES ON TEST. FULL FOR NON TEST
    limit_val_batches=0.5 if test else 1.0) #  SAME AS ABOVE COMMENT




trainer.fit(model, data)





