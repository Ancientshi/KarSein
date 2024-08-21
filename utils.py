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
        # #按timestamp排序，取前20000条数据
        # ratings= ratings.sort_values(by='timestamp').head(20000)
        
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
        
        # #去掉rating ==3的数据
        # data=data[data['rating']!=3]
        # #rating>3的设置为1
        # data['rating'] = data['rating'].apply(lambda x: 1 if x > 3 else 0)
        
         
        
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
    
    def init_neg_candidate_dict(self):
        all_movie_ids = set(range(self.get_vocab_sizes()['movie_id'] - 100))
        positive_movies = self.data[self.data['rating'] > 0].groupby('user_id')['movie_id'].apply(set)
        self.neg_candidate_dict = {user_id: list(all_movie_ids - movie_ids) for user_id, movie_ids in positive_movies.items()}

    def get_pair_data(self):
        positive_data = self.data[self.data['rating'] > 0][['user_id', 'movie_id']].rename(columns={'movie_id': 'positive_movie_id'})
        pair_data = [
            (user_id, pos_movie_id, random.choice(self.neg_candidate_dict[user_id]))
            for user_id, pos_movie_id in positive_data.values
        ]
        return pd.DataFrame(pair_data, columns=['user_id', 'positive_movie_id', 'negative_movie_id'])

    def get_previous_interactions_all(self, user_id, train):
        user_data = train[train['user_id'] == user_id]
        return list(user_data['positive_movie_id'].unique())
    
    def create_interaction_df_all(self, user_id_list, movie_id_list):
        cartesian_product = list(product(user_id_list, movie_id_list))
        interaction_df = pd.DataFrame(cartesian_product, columns=['user_id', 'movie_id'])
        interaction_df['rating'] = -1

        actual_data = self.data[['user_id', 'movie_id', 'rating']]
        interaction_df = interaction_df.merge(actual_data, on=['user_id', 'movie_id'], how='left', suffixes=('', '_actual'))
        interaction_df['rating'] = interaction_df['rating_actual'].fillna(interaction_df['rating'])
        interaction_df.drop('rating_actual', axis=1, inplace=True)
        return interaction_df

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

    def init_neg_candidate_dict(self):
        all_movie_ids = set(range(self.get_vocab_sizes()['movie_id'] - 100))
        positive_movies = self.data[self.data['rating'] > 0].groupby('user_id')['movie_id'].apply(set)
        self.neg_candidate_dict = {user_id: list(all_movie_ids - movie_ids) for user_id, movie_ids in positive_movies.items()}

    def get_pair_data(self):
        positive_data = self.data[self.data['rating'] > 0][['user_id', 'movie_id']].rename(columns={'movie_id': 'positive_movie_id'})
        pair_data = [
            (user_id, pos_movie_id, random.choice(self.neg_candidate_dict[user_id]))
            for user_id, pos_movie_id in positive_data.values
        ]
        return pd.DataFrame(pair_data, columns=['user_id', 'positive_movie_id', 'negative_movie_id'])

    def get_previous_interactions_all(self, user_id, train):
        user_data = train[train['user_id'] == user_id]
        return list(user_data['positive_movie_id'].unique())
    
    def create_interaction_df_all(self, user_id_list, movie_id_list):
        cartesian_product = list(product(user_id_list, movie_id_list))
        interaction_df = pd.DataFrame(cartesian_product, columns=['user_id', 'movie_id'])
        interaction_df['rating'] = -1

        actual_data = self.data[['user_id', 'movie_id', 'rating']]
        interaction_df = interaction_df.merge(actual_data, on=['user_id', 'movie_id'], how='left', suffixes=('', '_actual'))
        interaction_df['rating'] = interaction_df['rating_actual'].fillna(interaction_df['rating'])
        interaction_df.drop('rating_actual', axis=1, inplace=True)
        return interaction_df

class TaobaoTool:
    def __init__(self, path):
        self.path = path
        self.key2index = {}
    
    def load_and_preprocess(self):
        #user_id,item_id,behavior_type,user_geohash,item_category,time
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

    def init_neg_candidate_dict(self):
        all_item_ids = set(range(self.get_vocab_sizes()['item_id'] - 100))
        positive_movies = self.data[self.data['behavior_type'] > 0].groupby('user_id')['item_id'].apply(set)
        self.neg_candidate_dict = {user_id: list(all_item_ids - item_ids) for user_id, item_ids in positive_movies.items()}

    def get_pair_data(self):
        positive_data = self.data[self.data['behavior_type'] > 0][['user_id', 'item_id']].rename(columns={'item_id': 'positive_item_id'})
        pair_data = [
            (user_id, pos_item_id, random.choice(self.neg_candidate_dict[user_id]))
            for user_id, pos_item_id in positive_data.values
        ]
        return pd.DataFrame(pair_data, columns=['user_id', 'positive_item_id', 'negative_item_id'])

    def get_previous_interactions_all(self, user_id, train):
        user_data = train[train['user_id'] == user_id]
        return list(user_data['positive_item_id'].unique())
    
    def create_interaction_df_all(self, user_id_list, item_id_list):
        cartesian_product = list(product(user_id_list, item_id_list))
        interaction_df = pd.DataFrame(cartesian_product, columns=['user_id', 'item_id'])
        interaction_df['behavior_type'] = -1

        actual_data = self.data[['user_id', 'item_id', 'behavior_type']]
        interaction_df = interaction_df.merge(actual_data, on=['user_id', 'movie_id'], how='left', suffixes=('', '_actual'))
        interaction_df['behavior_type'] = interaction_df['behavior_type_actual'].fillna(interaction_df['behavior_type'])
        interaction_df.drop('behavior_type_actual', axis=1, inplace=True)
        return interaction_df
    
    
    
    
    
        
    
    
    
# Create the MovieLens dataset class
class MovieLensDataset(Dataset):
    # Initialization and data preparation steps are the same as you described
    def __init__(self, ratings, compare=False, topk=False):
        self.topk=topk
        self.compare=compare
        
        if compare == False:
            #"movie_id", "user_id","gender", "age", "occupation", "zip","genres"
            self.users = torch.tensor(ratings['user_id'].values, dtype=torch.int)
            self.movies = torch.tensor(ratings['movie_id'].values, dtype=torch.int)
            self.ratings = torch.tensor(ratings['rating'].values, dtype=torch.float)
        
            if not topk:
                self.genders = torch.tensor (ratings['gender'].values, dtype=torch.int)
                self.ages = torch.tensor(ratings['age'].values, dtype=torch.float)
                self.occupations = torch.tensor(ratings['occupation'].values, dtype=torch.int)
                self.zips = torch.tensor(ratings['zip'].values, dtype=torch.int)
                self.genres = torch.tensor(ratings['genres'].values.tolist(), dtype=torch.int)
            else:
                self.users_list=ratings['user_id'].values
                self.movies_list=ratings['movie_id'].values
                
                user_side=ml_tool.get_user_side()
                item_side=ml_tool.get_item_side()
                
                self.genders_dict={}
                self.ages_dict={}
                self.occupations_dict={}
                self.zips_dict={}
                for user_id in ratings['user_id'].unique():
                    user_info=user_side[user_side['user_id']==user_id]
                    gender=torch.tensor(user_info['gender'].values[0],dtype=torch.int)
                    age=torch.tensor(user_info['age'].values[0],dtype=torch.float)
                    occupation=torch.tensor(user_info['occupation'].values[0],dtype=torch.int)
                    zip=torch.tensor(user_info['zip'].values[0],dtype=torch.int)
                    self.genders_dict[user_id]=gender
                    self.ages_dict[user_id]=age
                    self.occupations_dict[user_id]=occupation
                    self.zips_dict[user_id]=zip
                    
                self.genres_dict={}
                for movie_id in ratings['movie_id'].unique():
                    movie_info=item_side[item_side['movie_id']==movie_id]
                    genre=torch.tensor(movie_info['genres'].values[0],dtype=torch.int)
                    self.genres_dict[movie_id]=genre
        
        else:
            #"movie_id", "user_id","gender", "age", "occupation", "zip","genres"
            self.users = torch.tensor(ratings['user_id'].values, dtype=torch.int)
            self.positive_movies = torch.tensor(ratings['positive_movie_id'].values, dtype=torch.int)
            self.negative_movies = torch.tensor(ratings['negative_movie_id'].values, dtype=torch.int)
            self.ratings = ratings
            
            self.users_list=ratings['user_id'].values
            self.positive_movies_list=ratings['positive_movie_id'].values
            self.negative_movies_list=ratings['negative_movie_id'].values
            
            user_side=ml_tool.get_user_side()
            item_side=ml_tool.get_item_side()
            
            self.genders_dict={}
            self.ages_dict={}
            self.occupations_dict={}
            self.zips_dict={}
            for user_id in ratings['user_id'].unique():
                user_info=user_side[user_side['user_id']==user_id]
                gender=torch.tensor(user_info['gender'].values[0],dtype=torch.int)
                age=torch.tensor(user_info['age'].values[0],dtype=torch.float)
                occupation=torch.tensor(user_info['occupation'].values[0],dtype=torch.int)
                zip=torch.tensor(user_info['zip'].values[0],dtype=torch.int)
                self.genders_dict[user_id]=gender
                self.ages_dict[user_id]=age
                self.occupations_dict[user_id]=occupation
                self.zips_dict[user_id]=zip
                
            self.positive_genres_dict={}
            for movie_id in ratings['positive_movie_id'].unique():
                movie_info=item_side[item_side['movie_id']==movie_id]
                genre=torch.tensor(movie_info['genres'].values[0],dtype=torch.int)
                self.positive_genres_dict[movie_id]=genre
            
            self.negative_genres_dict={}
            for movie_id in ratings['negative_movie_id'].unique():
                movie_info=item_side[item_side['movie_id']==movie_id]
                genre=torch.tensor(movie_info['genres'].values[0],dtype=torch.int)
                self.negative_genres_dict[movie_id]=genre
                
            
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
                
            else:
                userid = self.users_list[idx]
                movieid = self.movies_list[idx]
                dense = self.ages_dict[userid].unsqueeze(0)
                sparse = torch.stack([
                    self.movies[idx],
                    self.users[idx],
                    self.occupations_dict[userid],
                    self.zips_dict[userid],
                    self.genders_dict[userid]
                ])
                varlen = self.genres_dict[movieid].unsqueeze(0)
                target = self.ratings[idx]
            return (dense, sparse, varlen), target
        else:
            userid = self.users_list[idx]
            positive_movieid = self.positive_movies_list[idx]
            negative_movieid = self.negative_movies_list[idx]
            positive_dense = self.ages_dict[userid].unsqueeze(0)
            positive_sparse = torch.stack([
                self.positive_movies[idx],
                self.users[idx],
                self.occupations_dict[userid],
                self.zips_dict[userid],
                self.genders_dict[userid]
            ])
            positive_varlen = self.positive_genres_dict[positive_movieid].unsqueeze(0)
            positive_target=torch.tensor(1,dtype=torch.float)
             
            negative_dense = self.ages_dict[userid].unsqueeze(0)
            negative_sparse = torch.stack([
                self.negative_movies[idx],
                self.users[idx],
                self.occupations_dict[userid],
                self.zips_dict[userid],
                self.genders_dict[userid]
            ])
            negative_varlen = self.negative_genres_dict[negative_movieid].unsqueeze(0)
            negative_target=torch.tensor(0,dtype=torch.float)
            
            return ((positive_dense, positive_sparse, positive_varlen), positive_target), ((negative_dense, negative_sparse, negative_varlen), negative_target)
            

# Create the MovieLens dataset class
class DoubanDataset(Dataset):
    # Initialization and data preparation steps are the same as you described
    def __init__(self, ratings, compare=False, topk=False):
        self.topk=topk
        self.compare=compare
        
        if compare == False:
            #"movie_id", "user_id","gender", "age", "occupation", "zip","genres"
            self.users = torch.tensor(ratings['user_id'].values, dtype=torch.int)
            self.movies = torch.tensor(ratings['movie_id'].values, dtype=torch.int)
            self.ratings = torch.tensor(ratings['rating'].values, dtype=torch.float)
        
            if not topk:
                self.genres = torch.tensor(ratings['genres'].values.tolist(), dtype=torch.int)
            else:
                self.users_list=ratings['user_id'].values
                self.movies_list=ratings['movie_id'].values
                
                user_side=douban_tool.get_user_side()
                item_side=douban_tool.get_item_side()
                     
                self.genres_dict={}
                for movie_id in ratings['movie_id'].unique():
                    movie_info=item_side[item_side['movie_id']==movie_id]
                    genre=torch.tensor(movie_info['genres'].values[0],dtype=torch.int)
                    self.genres_dict[movie_id]=genre
        
        else:
            #"movie_id", "user_id","gender", "age", "occupation", "zip","genres"
            self.users = torch.tensor(ratings['user_id'].values, dtype=torch.int)
            self.positive_movies = torch.tensor(ratings['positive_movie_id'].values, dtype=torch.int)
            self.negative_movies = torch.tensor(ratings['negative_movie_id'].values, dtype=torch.int)
            self.ratings = ratings
            
            self.users_list=ratings['user_id'].values
            self.positive_movies_list=ratings['positive_movie_id'].values
            self.negative_movies_list=ratings['negative_movie_id'].values
            
            user_side=douban_tool.get_user_side()
            item_side=douban_tool.get_item_side()
            
           
                
            self.positive_genres_dict={}
            for movie_id in ratings['positive_movie_id'].unique():
                movie_info=item_side[item_side['movie_id']==movie_id]
                genre=torch.tensor(movie_info['genres'].values[0],dtype=torch.int)
                self.positive_genres_dict[movie_id]=genre
            
            self.negative_genres_dict={}
            for movie_id in ratings['negative_movie_id'].unique():
                movie_info=item_side[item_side['movie_id']==movie_id]
                genre=torch.tensor(movie_info['genres'].values[0],dtype=torch.int)
                self.negative_genres_dict[movie_id]=genre
                
            
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
                
            else:
                userid = self.users_list[idx]
                movieid = self.movies_list[idx]
                dense = torch.tensor(0,dtype=torch.float).unsqueeze(0)
                sparse = torch.stack([
                    self.movies[idx],
                    self.users[idx],
                ])
                varlen = self.genres_dict[movieid].unsqueeze(0)
                target = self.ratings[idx]
            return (dense, sparse, varlen), target
        else:
            userid = self.users_list[idx]
            positive_movieid = self.positive_movies_list[idx]
            negative_movieid = self.negative_movies_list[idx]
            positive_dense = torch.tensor(0,dtype=torch.float).unsqueeze(0)
            positive_sparse = torch.stack([
                self.positive_movies[idx],
                self.users[idx],
            ])
            positive_varlen = self.positive_genres_dict[positive_movieid].unsqueeze(0)
            positive_target=torch.tensor(1,dtype=torch.float)
             
            negative_dense = torch.tensor(0,dtype=torch.float).unsqueeze(0)
            negative_sparse = torch.stack([
                self.negative_movies[idx],
                self.users[idx],
            ])
            negative_varlen = self.negative_genres_dict[negative_movieid].unsqueeze(0)
            negative_target=torch.tensor(0,dtype=torch.float)
            
            return ((positive_dense, positive_sparse, positive_varlen), positive_target), ((negative_dense, negative_sparse, negative_varlen), negative_target)
                
# Create the Taobao dataset class
class TaobaoDataset(Dataset):
    # Initialization and data preparation steps are the same as you described
    def __init__(self, ratings, compare=False, topk=False):
        self.topk=topk
        self.compare=compare
        
        if compare == False:
            #"item_id", "user_id","gender", "age", "occupation", "zip","genres"
            self.users = torch.tensor(ratings['user_id'].values, dtype=torch.int)
            self.items = torch.tensor(ratings['item_id'].values, dtype=torch.int)
            self.ratings = torch.tensor(ratings['behavior_type'].values, dtype=torch.float)
        
            if not topk:
                self.category = torch.tensor(ratings['item_category'].values, dtype=torch.int)
            else:
                self.users_list=ratings['user_id'].values
                self.items_list=ratings['item_id'].values
                
                user_side=taobao_tool.get_user_side()
                item_side=taobao_tool.get_item_side()
                     
                self.category_dict={}
                for item_id,category in item_side[['item_id','item_category']].values:
                    category=torch.tensor(category,dtype=torch.int)
                    self.category_dict[item_id]=category
        
        else:
            #"item_id", "user_id","gender", "age", "occupation", "zip","genres"
            self.users = torch.tensor(ratings['user_id'].values, dtype=torch.int)
            self.positive_items = torch.tensor(ratings['positive_item_id'].values, dtype=torch.int)
            self.negative_items = torch.tensor(ratings['negative_item_id'].values, dtype=torch.int)
            self.ratings = ratings
            
            self.users_list=ratings['user_id'].values
            self.positive_items_list=ratings['positive_item_id'].values
            self.negative_items_list=ratings['negative_item_id'].values
            
            user_side=taobao_tool.get_user_side()
            item_side=taobao_tool.get_item_side()
            
            self.category_dict={}
            for item_id,category in item_side[['item_id','item_category']].values:
                category=torch.tensor(category,dtype=torch.int)
                self.category_dict[item_id]=category
            
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
                
            else:
                userid = self.users_list[idx]
                itemid = self.items_list[idx]
                dense = torch.tensor(0,dtype=torch.float).unsqueeze(0)
                sparse = torch.stack([
                    self.items[idx],
                    self.users[idx],
                    self.category_dict[itemid]
                ])
                varlen = torch.tensor([],dtype=torch.float).unsqueeze(0)
                target = self.ratings[idx]
            return (dense, sparse, varlen), target
        else:
            userid = self.users_list[idx]
            positive_itemid = self.positive_items_list[idx]
            negative_itemid = self.negative_items_list[idx]
            positive_dense = torch.tensor(0,dtype=torch.float).unsqueeze(0)
            positive_sparse = torch.stack([
                self.positive_items[idx],
                self.users[idx],
                self.category_dict[positive_itemid]
            ])
            positive_varlen = torch.tensor([],dtype=torch.float).unsqueeze(0)
            positive_target=torch.tensor(1,dtype=torch.float)
             
            negative_dense = torch.tensor(0,dtype=torch.float).unsqueeze(0)
            negative_sparse = torch.stack([
                self.negative_items[idx],
                self.users[idx],
                self.category_dict[negative_itemid]
            ])
            negative_varlen = torch.tensor([],dtype=torch.float).unsqueeze(0)
            negative_target=torch.tensor(0,dtype=torch.float)
            
            return ((positive_dense, positive_sparse, positive_varlen), positive_target), ((negative_dense, negative_sparse, negative_varlen), negative_target)
        
        
         

ml_tool=MovieLensTool('~/Data/ml-1m')
douban_tool=DoubanTool('~/Data/douban')
taobao_tool=TaobaoTool('~/Data/taobao2014')

def get_loader(name='ml-1m',batch_size=512,seed=42):
    # Load data
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
        #val, test = val_and_test, val_and_test
        
        #对train shuffle
        train=train.sample(frac=1)
        val=val.sample(frac=1)
        test=test.sample(frac=1)
        
        train_dataset = assigned_dataset(train,compare=False)
        val_dataset = assigned_dataset(val,compare=False)
        test_dataset = assigned_dataset(test,compare=False)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size*10, shuffle=False,num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size*10, shuffle=False,num_workers=4)
    
    return train_loader, val_loader, test_loader, feature_dict,vocab_size_dict

def plot_loss(train_loss_list, test_loss_list, title):
    # 创建画布并返回
    fig = plt.figure(figsize=(10, 6))
    plt.plot(train_loss_list, label='train')
    plt.plot(test_loss_list, label='test')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    return fig  # 返回画布对象
    
def auc(labels,outputs):
    return roc_auc_score(labels,outputs)


def evaluate_metrics(test_truth_list, test_prediction_list, topk=[20]):
    results = {
        'recall': [],
        'mrr': [],
        'ndcg': []
    }
    
    for k in topk:
        recall_list = []
        mrr_list = []
        ndcg_list = []
        
        for ind, test_truth in enumerate(test_truth_list):
            test_truth_index = set(test_truth)
            if len(test_truth_index) == 0:
                continue
                
            recall_dem = len(test_truth_index)
            top_sorted_index = test_prediction_list[ind][0:k]
            top_sorted_index_set = set(top_sorted_index)
            
            # Recall Calculation
            hit_num = len(top_sorted_index_set.intersection(test_truth_index))
            recall_list.append(hit_num * 1.0 / (recall_dem + 1e-20))
            
            # MRR Calculation
            mrr = 1.0
            ctr = 1e20
            for index, itemid in enumerate(top_sorted_index):
                if itemid in test_truth_index:
                    ctr = index + 1
                    break
            mrr /= ctr
            mrr_list.append(mrr)
            
            # NDCG Calculation
            dcg = 0
            idcg = 0
            idcg_dem = 0
            for index, itemid in enumerate(top_sorted_index):
                if itemid in test_truth_index:
                    dcg += 1.0 / np.log2(index + 2)
                    idcg += 1.0 / np.log2(idcg_dem + 2)
                    idcg_dem += 1
            ndcg = dcg * 1.0 / (idcg + 1e-20)
            ndcg_list.append(ndcg)
        
        results['recall'].append(np.mean(recall_list))
        results['mrr'].append(np.mean(mrr_list))
        results['ndcg'].append(np.mean(ndcg_list))
    
    return results


def recall(test_truth_list, test_prediction_list, topk=[20]):
    recalls = []
    for k in topk:
        recall_list = []
        for ind, test_truth in enumerate(test_truth_list):
            test_truth_index = set(test_truth)
            if len(test_truth_index) == 0:
                continue
            recall_dem = len(test_truth_index)
            top_sorted_index = set(test_prediction_list[ind][0:k])
            hit_num = len(top_sorted_index.intersection(test_truth_index))
            recall_list.append(hit_num * 1.0 / (recall_dem + 1e-20))
        recall = np.mean(recall_list)
        recalls.append(recall)
    return recalls

def mrr(test_truth_list, test_prediction_list, topk):
    mrrs = []
    for k in topk:
        mrr_list = []
        for ind, test_truth in enumerate(test_truth_list):
            mrr = 1.0
            test_truth_index = set(test_truth)
            if len(test_truth_index) == 0:
                continue
            top_sorted_index = set(test_prediction_list[ind][0:k])
            ctr = 1e20
            for index, itemid in enumerate(top_sorted_index):
                if itemid in test_truth_index:
                    ctr = index + 1
                    break
            mrr /= ctr
            mrr_list.append(mrr)
        mrrs.append(np.mean(mrr_list))
    return mrrs


def ndcg(test_truth_list, test_prediction_list, topk):
    ndcgs = []
    for k in topk:
        ndcg_list = []
        for ind, test_truth in enumerate(test_truth_list):
            dcg = 0
            idcg = 0
            test_truth_index = set(test_truth)
            if len(test_truth_index) == 0:
                continue
            top_sorted_index = set(test_prediction_list[ind][0:k])
            idcg_dem = 0
            for index, itemid in enumerate(top_sorted_index):
                if itemid in test_truth_index:
                    dcg += 1.0 / np.log2(index + 2)
                    idcg += 1.0 / np.log2(idcg_dem + 2)
                    idcg_dem += 1
            ndcg = dcg * 1.0 / (idcg + 1e-20)
            ndcg_list.append(ndcg)
        ndcgs.append(np.mean(ndcg_list))
    return ndcgs


    
def BPR(positive_predictions, negative_predictions):
    """
    Args:
    positive_predictions: 模型对正样本（用户已互动项）的预测分数
    negative_predictions: 模型对负样本（用户未互动项）的预测分数
    """
    loss = -torch.mean(torch.log(torch.sigmoid(positive_predictions - negative_predictions)))
    return loss


# def recall(labels, outputs, k=20):
#     # 获取每个样本的前k个最高得分的索引
#     _, top_k_predictions = outputs.topk(k, dim=1)
#     # 检查实际标签是否在预测的前k个中
#     match = top_k_predictions == labels.view(-1, 1)
#     # 计算recall@k
#     recall = match.any(dim=1).float().mean().item()
#     return recall