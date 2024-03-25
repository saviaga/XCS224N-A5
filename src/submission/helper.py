from .model import GPT
from .dataset import NameDataset,CharCorruptionDataset
from .trainer import Trainer, TrainerConfig

import torch
import random
import os
random.seed(0)

def initialize_vanilla_model(mconf):
    ### TODO:
    ### [part c]: Make some model here

    ### START CODE HERE
    mconf.attention_type = 'self'
    attention_model = GPT(mconf,mconf.attention_type)
    ### END CODE HERE
    return attention_model

def initialize_perceiver_model(mconf, bottleneck_dim=32):
    mconf.attention_type = 'cross'
    mconf.bottleneck_dim = bottleneck_dim
    attention_model =  GPT(mconf,mconf.attention_type)
    ### TODO
    ### [part g]: Make some other model here

    ### START CODE HERE
    ### END CODE HERE
    return attention_model

def finetune(reading_params_path, finetune_corpus_path, pretrain_dataset, block_size, model, finetune_lr=6e-4, writer=None):
    ### TODO:
    ### [part c] [part f]:
    ### - Given:
    ###     1. A finetuning corpus specified in finetune_corpus_path
    ###     2. A path reading_params_path containing pretrained model
    ###         parameters, or None if finetuning without a pretrained model
    ### - Goals:
    ###     1. If reading_params_path is specified, load these parameters
    ###         into the model
    ###     2. Finetune the model on this corpus
    ###
    ### - Make sure to use the following hyperparameters:
    ###     Hyperparameters for finetuning WITHOUT a pretrained model:
    ###         max_epochs=75
    ###         batch_size=256
    ###         learning_rate=6e-4
    ###         lr_decay=True
    ###         warmup_tokens=512*20
    ###         final_tokens=200*len(pretrain_dataset)*block_size
    ###         num_workers=4
    ###     Hyperparameters for finetuning WITH a pretrained model:
    ###         max_epochs=10
    ###         batch_size=256
    ###         learning_rate=6e-4
    ###         lr_decay=True
    ###         warmup_tokens=512*20
    ###         final_tokens=200*len(pretrain_dataset)*block_size
    ###         num_workers=4
    ###
    ###
    ### Note: Please use torch.load(reading_params_path, map_location=torch.device('cpu')) to load pretrained model 
    finetune_dataset = NameDataset(finetune_corpus_path, pretrain_dataset)
    
    tconf = TrainerConfig(max_epochs=75, batch_size=256, learning_rate=finetune_lr,
                          lr_decay=True, warmup_tokens=512*20,
                          final_tokens=200*len(pretrain_dataset)*block_size,
                          num_workers=4) #TrainerConfig object (see trainer.py for more details)
    
    trainer_obj = Trainer(model, finetune_dataset, None, tconf) #Trainer object (see trainer.py for more details)
    ### START CODE HERE
    if reading_params_path is not None:

        model.load_state_dict(torch.load(reading_params_path, map_location=torch.device('cpu')))

    finetune_dataset = NameDataset(finetune_corpus_path, pretrain_dataset)

    ### END CODE HERE
    return tconf, trainer_obj

def pretrain(pretrain_dataset, block_size, model, pretrain_lr=6e-3, writer=None):
    ### TODO:
    ### [part f]:
    ### - Given:
    ###     1. A corpus specified in pretrain_dataset
    ### - Goals:
    ###     1. Pretrain the model on this corpus
    ###
    ### - Make sure to use the following hyperparameters for pretraining:
    ###     max_epochs=650
    ###     batch_size=128
    ###     learning_rate=6e-3
    ###     lr_decay=True
    ###     warmup_tokens=512*20
    ###     final_tokens=200*len(pretrain_dataset)*block_size
    ###     num_workers=4
    

    # Now, data is a string containing the contents of your dataset file


    
    tconf = TrainerConfig(max_epochs=650, batch_size=128, learning_rate=pretrain_lr,
                          lr_decay=True, warmup_tokens=512*20,
                          final_tokens=200*len(pretrain_dataset)*block_size,
                          num_workers=4) #TrainerConfig object (see trainer.py for more details)
    trainer_obj =Trainer(model, pretrain_dataset, None, tconf) #Trainer object (see trainer.py for more details)

    ### START CODE HERE


# Get the directory of the current file (__file__ is the path to the current script)


    pretrain_dataset = CharCorruptionDataset(open('./../data/wiki.txt', encoding='utf-8').read(), 128) 
    print(len(pretrain_dataset))

    ### END CODE HERE
    return tconf, trainer_obj

def train(model, writing_params_path, trainer_obj):
    ### TODO:
    ### - Given:
    ###     An output path writing_params_path for the model parameters
    ### [part c]:
    ###
    ### Note: trainer_obj is of type Trainer (see trainer.py for more details)

    ### START CODE HERE
    trainer_obj.train()
    
    # Save the trained model parameters
    torch.save(model.state_dict(), writing_params_path)
    ### END CODE HERE
    return
