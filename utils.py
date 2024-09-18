import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,roc_auc_score
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from matplotlib import pyplot as plt
import os
from torch.nn.utils.rnn import pad_sequence
import sys
import math
from itertools import product
import random

seed=42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

def pad_sequences(sequences, maxlen, pad_value=0):
    padded_sequences = []
    for sequence in sequences:
        if len(sequence) >= maxlen:
            padded_sequences.append(sequence[:maxlen])
        else:
            padded_sequences.append(sequence + [pad_value] * (maxlen - len(sequence)))
    return padded_sequences



class MovieLensTool:
    def __init__(self, path):
        self.path = path
        self.key2index = {}


    def load_and_preprocess(self):
        def split(x):
            key_ans = x.split('|')
            for key in key_ans:
                if key not in self.key2index:
                    self.key2index[key] = len(self.key2index) + 1
            return list(map(lambda x: self.key2index[x], key_ans))

        rating_file = os.path.join(self.path, 'ratings.dat')
        user_file = os.path.join(self.path, 'users.dat')
        movie_file = os.path.join(self.path, 'movies.dat')

        ratings = pd.read_csv(rating_file, sep='::', header=None, engine='python', names=['user_id', 'movie_id', 'rating', 'timestamp'])
        
        users = pd.read_csv(user_file, sep='::', engine='python', encoding='latin-1', names=['user_id', 'gender', 'age', 'occupation', 'zip'])
        movies = pd.read_csv(movie_file, sep='::', engine='python', encoding='latin-1', names=['movie_id', 'title', 'genres'])
        data = pd.merge(pd.merge(ratings, users), movies)

        data['user_id'] = data['user_id'].astype('category').cat.codes
        data['movie_id'] = data['movie_id'].astype('category').cat.codes
        data['occupation'] = data['occupation'].astype('category').cat.codes
        data['zip'] = data['zip'].astype('category').cat.codes
        data['gender'] = data['gender'].astype('category').cat.codes

        genres_list = list(map(split, data['genres'].values))
        max_len = max(map(len, genres_list))
        genres_list = pad_sequences(genres_list, maxlen=max_len, pad_value=0)
        data['genres'] = list(genres_list)

        data['age'] = (data['age'] - data['age'].mean()) / data['age'].std()
        data['rating'] = data['rating'].apply(lambda x: 1 if x >= 3 else 0)
        
        user_id_list=data['user_id'].unique()
        movie_id_list=data['movie_id'].unique()
        user_df=pd.DataFrame(user_id_list,columns=['user_id'])
        movie_df=pd.DataFrame(movie_id_list,columns=['movie_id'])
        self.user_side=pd.merge(user_df, data[['user_id','age','occupation', 'zip', 'gender']], on='user_id', how='left')
        self.item_side= pd.merge(movie_df, data[['movie_id','genres']], on='movie_id', how='left')
        
        self.data=data
    
    def get_user_side(self):
        return self.user_side
    
    def get_item_side(self):
        return self.item_side
    
    def get_vocab_sizes(self):
        data=self.data
        vocab_sizes = {
            'user_id': len(data['user_id'].unique()) + 100,
            'movie_id': len(data['movie_id'].unique()) + 100,
            'occupation': len(data['occupation'].unique()) + 100,
            'zip': len(data['zip'].unique()) + 100,
            'gender': len(data['gender'].unique()) + 100,
            'genres': len(self.key2index) + 1 + 100
        }
        return vocab_sizes

    def get_feature_dict(self):
        feature_dict = {
            'dense_feature': ['age'],
            'sparse_feature': ['movie_id', 'user_id', 'occupation', 'zip', 'gender'],
            'varlen_sparse_feature': ['genres']
        }
        return feature_dict

    def get_ctr_data(self):
        return self.data

class DoubanTool:
    def __init__(self, path):
        self.path = path
        self.key2index = {}
    
    def load_and_preprocess(self):
        def split(x):
            key_ans = x.split('|')
            for key in key_ans:
                if key not in self.key2index:
                    self.key2index[key] = len(self.key2index) + 1
            return list(map(lambda x: self.key2index[x], key_ans))

        itemfeat_file=os.path.join(self.path, 'item_feat.csv')
        interaction_file = os.path.join(self.path, 'ratings.csv')
        
        itemfeat = pd.read_csv(itemfeat_file,sep=' ',header=None)
        itemfeat['genres']=itemfeat.apply(lambda x: '|'.join([str(i) for i in x.index if x[i]==1]),axis=1)
        itemfeat=itemfeat[['genres']]
        itemfeat['movie_id']=itemfeat.index
        
        interactions = pd.read_csv(interaction_file,header=None,sep='\t',names=['user_id','movie_id','rating','timestamp','year'])
        data = pd.merge(interactions, itemfeat, on='movie_id')  
            
        data['user_id'] = data['user_id'].astype('category').cat.codes
        data['movie_id'] = data['movie_id'].astype('category').cat.codes
        data['rating'] = data['rating'].apply(lambda x: 1 if x >= 3 else 0)
        genres_list = list(map(split, data['genres'].values))
        max_len = max(map(len, genres_list))
        genres_list = pad_sequences(genres_list, maxlen=max_len, pad_value=0)
        data['genres'] = list(genres_list)
        
        user_id_list=data['user_id'].unique()
        movie_id_list=data['movie_id'].unique()
        user_df=pd.DataFrame(user_id_list,columns=['user_id'])
        movie_df=pd.DataFrame(movie_id_list,columns=['movie_id'])
        self.user_side=user_df
        self.item_side= pd.merge(movie_df, data[['movie_id','genres']], on='movie_id', how='left')
        
        self.data=data
    
    
    def get_user_side(self):
        return self.user_side
    
    def get_item_side(self):
        return self.item_side
    
    def get_vocab_sizes(self):
        data=self.data
        vocab_sizes = {
            'user_id': len(data['user_id'].unique()) + 100,
            'movie_id': len(data['movie_id'].unique()) + 100,
            'genres': len(self.key2index) + 1 + 100
        }
        return vocab_sizes

    def get_feature_dict(self):
        feature_dict = {
            'dense_feature': ['placeholder'],
            'sparse_feature': ['movie_id', 'user_id'],
            'varlen_sparse_feature': ['genres']
        }
        return feature_dict

    def get_ctr_data(self):
        return self.data

class TaobaoTool:
    def __init__(self, path):
        self.path = path
        self.key2index = {}
    
    def load_and_preprocess(self):
        interaction_file = os.path.join(self.path, 'tianchi_mobile_recommend_train_user.csv')
        data = pd.read_csv(interaction_file)
        
        begin=5000
        active_users = data['user_id'].value_counts().index.tolist()
        active_users=active_users[begin:begin+1000]
        data = data[data['user_id'].isin(active_users)]

        data['user_id'] = data['user_id'].astype('category').cat.codes
        data['item_id'] = data['item_id'].astype('category').cat.codes
        data['item_category']=data['item_category'].astype('category').cat.codes
        data['behavior_type'] =  1

        user_id_list=data['user_id'].unique()
        item_id_list=data['item_id'].unique()
        user_df=pd.DataFrame(user_id_list,columns=['user_id'])
        item_df=pd.DataFrame(item_id_list,columns=['item_id'])
        self.user_side=user_df
        self.item_side= pd.merge(item_df, data[['item_id','item_category']], on='item_id', how='left')
        
        self.data=data
    
    
    def get_user_side(self):
        return self.user_side
    
    def get_item_side(self):
        return self.item_side
    
    def get_vocab_sizes(self):
        data=self.data
        vocab_sizes = {
            'user_id': len(data['user_id'].unique()) + 100,
            'item_id': len(data['item_id'].unique()) + 100,
            'item_category': len(data['item_category'].unique()) + 100
        }
        return vocab_sizes

    def get_feature_dict(self):
        feature_dict = {
            'dense_feature': [],
            'sparse_feature': ['item_id', 'user_id','item_category'],
            'varlen_sparse_feature': []
        }
        return feature_dict

    def get_ctr_data(self):
        return self.data
    
    
class MovieLensDataset(Dataset):
    def __init__(self, ratings, compare=False, topk=False):
        self.topk=topk
        self.compare=compare
        
        if compare == False:
            self.users = torch.tensor(ratings['user_id'].values, dtype=torch.int)
            self.movies = torch.tensor(ratings['movie_id'].values, dtype=torch.int)
            self.ratings = torch.tensor(ratings['rating'].values, dtype=torch.float)
        
            if not topk:
                self.genders = torch.tensor (ratings['gender'].values, dtype=torch.int)
                self.ages = torch.tensor(ratings['age'].values, dtype=torch.float)
                self.occupations = torch.tensor(ratings['occupation'].values, dtype=torch.int)
                self.zips = torch.tensor(ratings['zip'].values, dtype=torch.int)
                self.genres = torch.tensor(ratings['genres'].values.tolist(), dtype=torch.int)
             
            
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        if self.compare == False:
            if not self.topk:
                dense = self.ages[idx].unsqueeze(0)
                sparse = torch.stack([
                    self.movies[idx],
                    self.users[idx],
                    self.occupations[idx],
                    self.zips[idx],
                    self.genders[idx]
                ])
                varlen = self.genres[idx].unsqueeze(0)
                target = self.ratings[idx]

            return (dense, sparse, varlen), target
            
class DoubanDataset(Dataset):
    def __init__(self, ratings, compare=False, topk=False):
        self.topk=topk
        self.compare=compare
        
        if compare == False:
            self.users = torch.tensor(ratings['user_id'].values, dtype=torch.int)
            self.movies = torch.tensor(ratings['movie_id'].values, dtype=torch.int)
            self.ratings = torch.tensor(ratings['rating'].values, dtype=torch.float)
        
            if not topk:
                self.genres = torch.tensor(ratings['genres'].values.tolist(), dtype=torch.int)
            
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        if self.compare == False:
            if not self.topk:
                dense = torch.tensor(0,dtype=torch.float).unsqueeze(0)
                sparse = torch.stack([
                    self.movies[idx],
                    self.users[idx],
                ])
                varlen = self.genres[idx].unsqueeze(0)
                target = self.ratings[idx]
                
            return (dense, sparse, varlen), target
        
class TaobaoDataset(Dataset):
    def __init__(self, ratings, compare=False, topk=False):
        self.topk=topk
        self.compare=compare
        
        if compare == False:
            self.users = torch.tensor(ratings['user_id'].values, dtype=torch.int)
            self.items = torch.tensor(ratings['item_id'].values, dtype=torch.int)
            self.ratings = torch.tensor(ratings['behavior_type'].values, dtype=torch.float)
        
            if not topk:
                self.category = torch.tensor(ratings['item_category'].values, dtype=torch.int)
        
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        if self.compare == False:
            if not self.topk:
                dense = torch.tensor(0,dtype=torch.float).unsqueeze(0)
                sparse = torch.stack([
                    self.items[idx],
                    self.users[idx],
                    self.category[idx]
                ])
                varlen = torch.tensor([],dtype=torch.float).unsqueeze(0)
                target = self.ratings[idx]
                
            return (dense, sparse, varlen), target
         

ml_tool=MovieLensTool('~/Data/ml-1m')
douban_tool=DoubanTool('~/Data/douban')
taobao_tool=TaobaoTool('~/Data/taobao2014')

def get_loader(name='ml-1m',batch_size=512,seed=42):
    if name in ['ml-1m','douban']:
        tool_map={'ml-1m':ml_tool,'douban':douban_tool}
        dataset_map={'ml-1m':MovieLensDataset,'douban':DoubanDataset}
        assigned_tool=tool_map[name]
        assigned_dataset= dataset_map[name]
        
        assigned_tool.load_and_preprocess()
        feature_dict=assigned_tool.get_feature_dict()
        vocab_size_dict=assigned_tool.get_vocab_sizes()
        
        train=assigned_tool.get_ctr_data()

        train, val_and_test = train_test_split(train, test_size=0.2, random_state=seed)
        val, test = train_test_split(val_and_test, test_size=0.5, random_state=seed)
        
        train=train.sample(frac=1)
        val=val.sample(frac=1)
        test=test.sample(frac=1)
        
        train_dataset = assigned_dataset(train,compare=False)
        val_dataset = assigned_dataset(val,compare=False)
        test_dataset = assigned_dataset(test,compare=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size*10, shuffle=False,num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size*10, shuffle=False,num_workers=4)
    
    return train_loader, val_loader, test_loader, feature_dict,vocab_size_dict

def auc(labels,outputs):
    return roc_auc_score(labels,outputs)










