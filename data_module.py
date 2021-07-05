import pytorch_lightning as pl



class SmilesDataModule(pl.LightningDataModule):

  def __init__(self, batch_size = None, max_length = None, data_index = None, to_grab_checkpoint = False):
    super().__init__()
    self.batch_size = batch_size
    self.tokenizer = self.get_tokenizer()
    self.max_length = max_length
    self.data_index = data_index
    self.to_grab_checkpoint = to_grab_checkpoint
    
    self.train_smiles = []
    self.train_masks = []

    self.val_smiles = []
    self.val_masks = []
    
    self.t_smiles, self.v_smiles = self.get_data()
    self.all_raw_smiles = self.t_smiles + self.v_smiles

    self.train_tensordataset = None
    self.valid_tensordataset = None

  def setup(self, stage=None):
    if 'train_tensordataset.pt' in os.listdir(os.getcwd()) and 'valid_tensordataset.pt' in os.listdir(os.getcwd()):
      print('Grabbing previous train, val tensor_datasets from current dir')
      self.train_tensordataset = torch.load('train_tensordataset.pt')
      self.valid_tensordataset = torch.load('valid_tensordataset.pt')
    else:
      # compile train data lists
      self.compile_data_lists(smiles=self.t_smiles, smiles_list=self.train_smiles, masks_list=self.train_masks)
      # compile valid data lists
      self.compile_data_lists(smiles=self.v_smiles, smiles_list=self.val_smiles, masks_list=self.val_masks)
      
  def sep(self, x):
    return ' '.join(str(x))

  def convert_token(self, input, max_l):
    return self.tokenizer.encode_plus(self.sep(input), 
                                      truncation=True, 
                                      add_special_tokens=True, 
                                      max_length=max_l, 
                                      padding="max_length", 
                                      return_attention_mask=True,
                                      return_tensors='pt')
    
  def get_tokenizer(self):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer

  def compile_data_lists(self, smiles = None, smiles_list = None, masks_list = None):
    for smile in smiles:
      encoding_dict = self.convert_token(smile, self.max_length)
      smiles_list.append(torch.tensor(encoding_dict['input_ids']))
      masks_list.append(torch.tensor(encoding_dict['attention_mask']))

  def get_data(self):
    train = load_dataset("jglaser/binding_affinity",split="train[:90%]")[:self.data_index]
    validation = load_dataset("jglaser/binding_affinity",split="train[90%:]")[:self.data_index]
    return train['smiles'], validation['smiles']


  def train_dataloader(self, new_molecules = None):
    return self.set_dataloader(tensor_dataset=self.train_tensordataset if new_molecules is None else new_molecules, 
                          smiles=self.train_smiles, masks=self.train_masks,
                          filename='train_tensordataset.pt')
  
  def val_dataloader(self):
    return self.set_dataloader(tensor_dataset=self.valid_tensordataset,
                          smiles=self.val_smiles, masks=self.val_masks,
                          filename='valid_tensordataset.pt')
    
  def set_dataloader(self, tensor_dataset=None, smiles=None, masks=None, filename=None):
    if tensor_dataset != None:
      return DataLoader(tensor_dataset, batch_size=self.batch_size)
    else:
      smiles, masks = (torch.cat(smiles, dim=0), torch.cat(masks, dim=0))
      tensor_dataset = TensorDataset(smiles, masks)
      return DataLoader(tensor_dataset, batch_size=self.batch_size, num_workers = 4, pin_memory = False)

