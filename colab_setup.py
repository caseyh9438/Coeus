!pip install --upgrade pip
!pip install transformers
!pip install sentencepiece
!pip install datasets
!pip install pytorch-lightning
!pip install lightning-bolts
!curl -Lo conda_installer.py https://raw.githubusercontent.com/deepchem/deepchem/master/scripts/colab_install.py
import conda_installer
conda_installer.install()
!/root/miniconda/bin/conda info -e
!wget -c https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
!chmod +x Miniconda3-latest-Linux-x86_64.sh
!./Miniconda3-latest-Linux-x86_64.sh -b -f -p /usr/local
!conda config --set always_yes yes --set changeps1 no
!conda install -q -y -c conda-forge python=3.7
!conda install -q -y -c conda-forge rdkit==2020.09.2
!pip install git+https://github.com/bp-kelley/descriptastorus
!pip install deeppurpose
import sys
sys.path.append('/usr/local/lib/python3.7/site-packages/')

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