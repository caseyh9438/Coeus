max_length = 250 # MAX LENGTH FOR SMILE SEQUENCE MODEL INPUT
learning_rate = 3e-5 
batch_size = 16 # NUMBER OF BATCHES
max_epochs = 10 # MAXIMUM NUMBER OF EPOCHS THE TRAINER WILL ALLOW. THIS NUMBER WILL NOT ALWAYS BE REACHED. THE TRAINER SELF TERMINATES WHEN VALIDATION LOSS TRENDS UP.
test = False # THIS REDUCES DATA MODULE SIZE TO TEST AND DEBUG QUICKER
to_grab_checkpoint = False # IF TRUE, THIS GRABS THE MOST RECENT MODEL FROM GOOGLE CLOUD STORAGE STARTS TRAINING FROM WHERE THAT MODEL LEFT OFF
data_index = 1000000 if test is False else 1000 # THIS INDEXS THE TOTAL NUMBER OF MOLECULES ON WHICH TO TRAIN THE MODEL