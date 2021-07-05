import pytorch_lightning as pl




class SmilesTransformerModel(pl.LightningModule):
  def __init__(self, batch_size = None, max_length = None, learning_rate = None, 
               tokenizer = None, smiles_data_module = None, feedback = None,
               to_grab_checkpoint = False):
    
    super().__init__()
    self.batch_size = batch_size
    self.save_checkpoint_name = "SmilesTransformerModel"
    self.checkpoint_name_to_grab = "SmilesTransformerModel"
    self.pretrained_model = 'gpt2'
    self.bucket_name = 'htr1'
    self.tokenizer = tokenizer
    self.max_length = max_length
    self.learning_rate = learning_rate
    self.smiles_data_module = smiles_data_module
    self.configuration = GPT2Config.from_pretrained(self.pretrained_model)
    self.loaded_optimizer = self.grab_checkpoint()[1] if to_grab_checkpoint is True else None
    self.gpt2 = self.grab_checkpoint()[0] if to_grab_checkpoint is True else self.get_model()
    self.epoch = 0
    self.molecules_to_generate = 200
    self.feedback = feedback
                 

  def forward(self, smiles, masks):
    output = self.gpt2(smiles, labels = smiles, 
                  attention_mask = masks, 
                  token_type_ids = None)
    return output.loss

  def grab_checkpoint(self, model=True):
    return self.load_from_checkpoint()

  def get_model(self):
    model = AutoModelWithLMHead.from_pretrained(self.pretrained_model, config = self.configuration)
    model.resize_token_embeddings(len(self.tokenizer))
    return model

  def configure_optimizers(self, model = None):
    return torch.optim.AdamW(self.parameters() if model is None else model.parameters(), lr = self.learning_rate, eps = 4e-5)

  def training_step(self, train_batch, batch_idx):
    return self.make_step(batch = train_batch, index = batch_idx, logging = 'train_loss')

  def validation_step(self, val_batch, val_idx):
    return self.make_step(batch = val_batch, index = val_idx, logging = 'val_loss')

  def make_step(self, batch, index, logging = 'train_loss'):
    smiles, masks = batch
    loss = self(smiles = smiles, masks = masks)
    self.log(logging, loss)
    return loss

  def on_batch_end(self):
    pass
  
  def on_validation_epoch_end(self):
    " UPDATING TRAIN DATALOADER WITH NEW MOLECULES "
    list_generated_druglike_molecules = self.feedback.run_feedback_pipeline(model = self.gpt2, 
                                                                            molecules_to_generate = self.molecules_to_generate)
    if self.epoch == 0: # RUN BINDING ON FIRST EPOCH ONLY
      self.smiles_data_module.all_raw_smiles = [(self.feedback.get_binding_prediction(molecule = smile), smile) for smile in self.smiles_data_module.all_raw_smiles]
    combined_lists = list_generated_druglike_molecules + self.smiles_data_module.all_raw_smiles
    combined_lists.sort(reverse=True)
    combined_lists = combined_lists[:self.smiles_data_module.data_index if len(combined_lists) >= 1000 else None]
    print("New List Details:\nLength: {}\nAverage Dataset Binding Score: {}\n".format(len(combined_lists), sum([smiles[0] for smiles in combined_lists]) / len(combined_lists)))
    self.smiles_data_module.all_raw_smiles = combined_lists
    total_molecule_tokens = [self.smiles_data_module.convert_token(molecule[1], self.max_length) for molecule in combined_lists]
    new_molecules_dataloader = self.smiles_data_module.set_dataloader(smiles = [input['input_ids'] for input in total_molecule_tokens], 
                                                                      masks = [mask['attention_mask'] for mask in total_molecule_tokens])
    self.smiles_data_module.train_dataloader(new_molecules = new_molecules_dataloader)

    " SAVING MODEL CHECKPOINT TO GOOGLE STORAGE "
    self.save_checkpoint()

  def save_checkpoint(self):
    try:
      loss = trainer.logged_metrics['val_loss'][-1].item()
    except:
      loss = 0
    path = '{}_{}_{}'.format(self.save_checkpoint_name, trainer.current_epoch, loss)
    storage_client = storage.Client()
    bucket = storage_client.bucket(self.bucket_name)
    blob = bucket.blob(path)
    print("Saving the model...")
    torch.save({
                'epoch': self.current_epoch,
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizers[0].state_dict(),
                'loss': loss,
                }, path)
    
    blob.upload_from_filename(path)

  def get_most_recent_model(self):
    storage_client = storage.Client()
    bucket = storage_client.bucket(self.bucket_name)
    recent_models = [n for n in bucket.list_blobs() if self.checkpoint_name_to_grab in str(n)]
    time, url, name = max([(str(time).split(' ')[-1].rstrip('>'), time.public_url, str(time).split(' ')[-2].rstrip(',')) for time in recent_models])
    return name, url

  def to_device(self, device, items=[]): 
    for item in items:
      for state in item.state.values():
          for k, v in state.items():
              if torch.is_tensor(v):
                  state[k] = v.cuda()

  def load_from_checkpoint(self, device=torch.device('cuda')):
    checkpoint_name, checkpoint_url = self.get_most_recent_model()
    urllib.request.urlretrieve(checkpoint_url, checkpoint_name)
    # checkpoint = torch.load(checkpoint_name, map_location=device)
    # print('Loaded checkpoint from epoch {} with loss of {}'.format(checkpoint['epoch'], checkpoint['loss']))
    model = self.get_model()
    optimizer = self.configure_optimizers(model = model)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    checkpoint = torch.load(checkpoint_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    self.to_device(device, items=[optimizer])
    return model, optimizer