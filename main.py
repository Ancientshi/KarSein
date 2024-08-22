
import random
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from utils import *
from models.KarSein import *
import math
import json
import argparse



def parse_args():
    parser = argparse.ArgumentParser(description="Configure the model settings.")
    parser.add_argument('--seed', type=int, default=42, help='Random Seed')
    parser.add_argument('--netname', type=str, default='WDL', help='Network Name')
    parser.add_argument('--use_bit_wise', type=int, default=1, help='Use KarSein_bit')
    parser.add_argument('--use_vec_wise', type=int, default=1, help='Use KarSein_vec')
    parser.add_argument('--pairwise_multiplication', type=int, nargs='+', default=[0,1], help='Pairwise Multiplication')
    parser.add_argument('--emb_dim', type=int, default=16, help='Embedding Dimension')
    parser.add_argument('--bit_width', type=int, nargs='+', default=[], help='Network Width Configuration for KarSein_bit')
    parser.add_argument('--vec_width', type=int, nargs='+', default=[], help='Network Width Configuration for KarSein_vec')
    parser.add_argument('--grid', type=int, default=16, help='B-Spline Grid Size')
    parser.add_argument('--k', type=int, default=3, help='B-Spline Order')
    parser.add_argument('--dataset', type=str, default='ml-1m', help='Dataset')
    parser.add_argument('--task', type=str, default='TopK', choices=['CTR'], help='Task Type (CTR)')
    parser.add_argument('--epochs', type=int, default=5, help='Number of Epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch Size for Training, 10*batch_size for Evaluation')
    #reg 0.01
    parser.add_argument('--reg', type=float, default=0.01, help='Regularization Rate')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight Decay Rate')
    parser.add_argument('--gamma', type=float, default=0.9, help='Gamma Value for Optimizer')
    parser.add_argument('--note', type=str, default='None', help='Extra Notes for The Experiment')
    parser.add_argument('--device_index', type=int, default=1, help='Device Index')
    args = parser.parse_args()
    return args

args = parse_args()
device_index=args.device_index


seed= args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

def count_parameters(model):
    noemb_params = 0
    emb_params = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad: 
            if 'emb' in name:
                emb_params += param.numel()
            else:
                noemb_params += param.numel()
                
    return emb_params, noemb_params


model_classes = {
    'KARSEIN': KARSEIN
}
    
    
def initialize_model_KARSEIN(netname, feature_dict,vocab_size_dict):
    if netname in model_classes.keys():
        model = model_classes[netname](feature_dict,vocab_size_dict,args.bit_width, args.vec_width, args.grid, args.k, use_bit_wise=args.use_bit_wise,use_vec_wise=args.use_vec_wise,emb_dim=args.emb_dim,pairwise_multiplication=args.pairwise_multiplication)
        return model
    else:
        sys.exit(f"No model found with the name {netname}")
    
def init_model(feature_dict,vocab_size_dict):
    if 'KARSEIN' in args.netname:
        model = initialize_model_KARSEIN(args.netname,feature_dict,vocab_size_dict)
    else:
        sys.exit('No model found with the name {args.netname}')
        
    model=model.cuda(device_index)
    
    # Count and print the number of parameters in millions
    emb_param_count, no_emb_param_count = count_parameters(model)
    emb_param_count_millions = emb_param_count / 1e6
    no_emb_param_count_millions = no_emb_param_count / 1e6
    print(f"Embedding parameters: {emb_param_count_millions:.6f} million")
    print(f"No embedding parameters: {no_emb_param_count_millions:.6f} million")
    return model

if args.task=='CTR':
    criterion =  nn.BCELoss()
else:
    sys.exit('Only CTR task is supported')

train_loader, val_loader, test_loader, feature_dict,vocab_size_dict = get_loader(name=args.dataset,batch_size=args.batch_size,seed=args.seed)
print(f'Train Size: {len(train_loader.dataset)}, Val Size: {len(val_loader.dataset)}, CTR Test Size: {len(test_loader.dataset)}')

model=None
def train(model=None):
    best_model = None
    best_loss = float('inf')
    if model is None:
        model=init_model(feature_dict,vocab_size_dict)
        
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

    model.train()
    for epoch in range(args.epochs):
        train_loss = 0
        if args.task=='CTR':
            for inputs, labels in train_loader:
                inputs = [input.cuda(device_index) for input in inputs]
                labels = labels.cuda(device_index)
                outputs = model(inputs)
                optimizer.zero_grad() 
                loss = criterion(outputs.squeeze(), labels) + args.reg*model.regularization_loss()
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
        scheduler.step()
        mean_train_loss = train_loss / len(train_loader)
        print(f"Epoch: {epoch}, Train Loss: {mean_train_loss:.4f}")

        val_loss = 0
        auc_value_sum=0
        with torch.no_grad():
            if args.task=='CTR':
                for inputs, labels in val_loader:
                    inputs = [input.cuda(device_index) for input in inputs]
                    labels = labels.cuda(device_index)
                    outputs = model(inputs)
                    loss = criterion(outputs.squeeze(), labels)
                    auc_value=auc(labels.detach().cpu().numpy(),outputs.detach().cpu().numpy())
                    auc_value_sum+=auc_value
                    val_loss += loss.item()
            
            mean_val_loss = val_loss / len(val_loader)
            mean_auc_value = auc_value_sum / len(val_loader)
            print(f"Epoch: {epoch}, Val Loss: {mean_val_loss:.4f}, Val AUC: {mean_auc_value:.4f}")

            if mean_val_loss < best_loss:
                best_loss = mean_val_loss
                best_model = copy.deepcopy(model)
       
    return best_model

best_model = train()

# Evaluate best model
best_model.eval()
auc_value_sum=0
recall_eva_df=pd.DataFrame(columns=['user_id','movie_id','rating','prediction'],index=None)
with torch.no_grad():
    if args.task=='CTR':
        for inputs, labels in test_loader:
            inputs = [input.cuda(device_index) for input in inputs]
            labels = labels.cuda(device_index)
            outputs = best_model(inputs)
            auc_value=auc(labels.detach().cpu().numpy(),outputs.detach().cpu().numpy())

      
    mean_auc_value = auc_value_sum / len(test_loader)

    AUC_value_str=f'{mean_auc_value:.4f}'
    print(f"Best Model, AUC: {AUC_value_str}")
    
csv_file_path='results/all.csv'
extra_columns=['test_auc']
result_df= pd.DataFrame(columns=list(args.__dict__.keys())+extra_columns)
result_df.loc[0]=list(args.__dict__.values())+[mean_auc_value]

# Check if the file exists
if not os.path.exists(csv_file_path):
    result_df.to_csv(csv_file_path, mode='a', header=True, index=False)
else:
    result_df.to_csv(csv_file_path, mode='a', header=False, index=False)

                 
                
                
                