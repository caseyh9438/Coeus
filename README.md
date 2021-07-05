
# Coeus - Feedback Transformer Optimizing Small Molecule Generation

Coeus is a pipeline for training a model to generate small molecules that exhibit your desired binding, permeability and druglike behaviors. The model is optimized to maximize binding to the target pocket's
amino acid sequence while simultaneusly mimizing the binding to non-desired proteins and pocket sequences. Ceous generates new molecules every epoch, analyes them against the utility function and updates the training
data with the best preformers.

The code currently targets the MLH3 endonuclease domain pocket sequence to generate small molecules to prevent somatic expansion in Huntington's disease and other dynamic mutation tri-nuculeotide disorders.

Coeus is built using Pytorch and Pytorch Lightning.

Google Cloud Storage (GCS) recieves model checkpoints at the end of each epoch and that is where the most recent model is grabbed if instructed. To access this functionality you'll need to have access to the .json key file.

The SMILES data used to train the model is pulled from a GCS url. 


I would recommend using the conda env locally and Docker only when training on cloud servers.


**Local Conda Env Setup:**
```sh
git clone https://github.com/caseyh9438/Coeus
cd Coeus
conda env create -f environment.yml
conda activate coeus
python main.py
```


**Cloud Linux Server Env Setup:**
```sh
git clone https://github.com/caseyh9438/Coeus
cd Coeus
bash docker_install.sh
docker-compose up --build
```


**For Google Colab Env Setup:**
```sh
copy and paste setup code from colab_setup.py and run in the first cell
copy and paste other classes, settings and main py files
```
