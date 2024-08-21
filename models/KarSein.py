import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, constant_
from models.Layer import *

class KARSEIN(nn.Module):
    def __init__(self, feature_dict,vocab_size_dict, bit_width, vec_width, grid=3, k=3, noise_scale=0.1, noise_scale_base=0.1, base_fun=torch.nn.SiLU, grid_eps=1.0, grid_range=[-1,1],scale_spline=1.0,use_bit_wise=True,use_vec_wise=True,emb_dim=16,pairwise_multiplication=[0,1]):
        super(KARSEIN, self).__init__()
        self.bit_width=bit_width
        self.vec_width=vec_width
        self.grid=grid
        self.k=k
        self.noise_scale=noise_scale
        self.noise_scale_base=noise_scale_base
        self.base_fun=base_fun
        self.grid_eps=grid_eps
        self.grid_range=grid_range
        self.scale_spline=scale_spline
        self.emb_dim=emb_dim
        self.pairwise_multiplication=pairwise_multiplication
        self.use_bit_wise=use_bit_wise
        self.use_vec_wise=use_vec_wise
        
        self.dense_feature=feature_dict['dense_feature']
        self.sparse_feature=feature_dict['sparse_feature']
        self.varlen_sparse_feature=feature_dict['varlen_sparse_feature']
        self.vocab_size_dict=vocab_size_dict
        
        
        self.sparse2emb_embeddings = nn.ModuleDict({
            name: nn.Embedding(num_embeddings=self.vocab_size_dict[name], embedding_dim=self.emb_dim)
            for name in self.sparse_feature
        })
        
        self.varsparse2emb_embeddings = nn.ModuleDict({
            name: nn.Embedding(num_embeddings=self.vocab_size_dict[name], embedding_dim=self.emb_dim)
            for name in self.varlen_sparse_feature
        })
        
        self.emb_num=len(self.sparse_feature)+len(self.varlen_sparse_feature)
        self.concatenated_emb_dim=self.emb_dim*self.emb_num
        
        if self.use_bit_wise:
            width_karsein_bit=self.concatenated_emb_dim+len(self.dense_feature)
            self.karsein_bit_width=[width_karsein_bit]+self.bit_width+[1]
            self.karsein_bit=KarSein_Layer(self.karsein_bit_width,grid_size=grid,spline_order=k,scale_noise=noise_scale,scale_base=noise_scale_base,scale_spline=scale_spline,base_activation=base_fun,grid_eps=grid_eps,grid_range=grid_range)
        
        if self.use_vec_wise:
            self.karsein_vec_width=[self.emb_num]+self.vec_width+[1]
            self.karsein_vec_list=nn.ModuleList()
            
            output_dim=self.emb_num
            for i in range(len(self.karsein_vec_width)-1):
                if i in self.pairwise_multiplication:
                    input_dim=self.karsein_vec_width[0]*output_dim+self.karsein_vec_width[0]
                else:
                    input_dim=output_dim
                    
                output_dim=self.karsein_vec_width[i+1]
                kan=KarSein_Layer([input_dim,output_dim],grid_size=grid,spline_order=k,scale_noise=noise_scale,scale_base=noise_scale_base,scale_spline=scale_spline,base_activation=base_fun,grid_eps=grid_eps,grid_range=grid_range,emb_dim=self.emb_dim)
                self.karsein_vec_list.append(kan)
            
            self.vec_pred_layer=nn.Linear(self.karsein_vec_width[-1]*self.emb_dim,1)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)
            
            
    def cross_feature(self, x1, x2):
        """
        Calculate the pairwise cross features between input tensors x1 and x2.
        
        Args:
        x1: torch.Tensor - Input (batch, 64, d) tensor
        x2: torch.Tensor - Input (batch, 64, d) tensor
         
        Returns:
        torch.Tensor - Concatenated tensor of original features and cross features (batch, 64, d + d*d)
        """
        batch_size, x1_dim, emb_dim = x1.shape
        batch_size, x2_dim, emb_dim = x2.shape

        crossed_features = (x1.unsqueeze(2) * x2.unsqueeze(1)).reshape(batch_size, x1_dim * x2_dim, emb_dim)

        return crossed_features
    
    def forward_(self, x):
        dense, sparse, varsparse = x

        # Efficiently process sparse2emb embeddings
        sparse2emb_list = [
            embedding(sparse[:, index].long())
            for index, embedding in enumerate(self.sparse2emb_embeddings.values())
        ]
        sparse2emb = torch.cat(sparse2emb_list, dim=1)
        
        # Reshape the embeddings for further processing
        concatenated_emb = sparse2emb.view(-1, sum(e.embedding_dim for e in self.sparse2emb_embeddings.values()))
        matrix_X1 = concatenated_emb.view(-1, len(self.sparse_feature), self.emb_dim)
        
        # If there are varlen sparse features, process them
        if self.varlen_sparse_feature:
            varsparse2emb_list = [
                embedding(varsparse[:, index].long())
                for index, embedding in enumerate(self.varsparse2emb_embeddings.values())
            ]
            # Mean pooling across variable-length sparse embeddings
            varsparse2emb = torch.cat(varsparse2emb_list, dim=1)
            varsparse2emb_mean = torch.mean(varsparse2emb, dim=1)
            varsparse2emb_reshaped = varsparse2emb_mean.view(-1, sum(e.embedding_dim for e in self.varsparse2emb_embeddings.values()))
            
            # Concatenate the varlen embeddings with the original embeddings
            concatenated_emb = torch.cat([concatenated_emb, varsparse2emb_reshaped], dim=1)
            matrix_X1 = torch.cat([matrix_X1, varsparse2emb_mean.view(-1, 1, self.emb_dim)], dim=1)
        
        return dense, concatenated_emb, matrix_X1
    
    def forward(self,x):
        dense, concatenated_emb, matrix_X1=self.forward_(x)
        
        if self.use_vec_wise==True:
            matrix_output=matrix_X1.clone()
            for i in range(len(self.karsein_vec_list)):
                if i in self.pairwise_multiplication:
                    crossed_features=self.cross_feature(matrix_X1,matrix_output)
                    matrix_input=torch.cat([matrix_X1,crossed_features],dim=1)
                else:
                    matrix_input=matrix_output
                    
                matrix_output=self.karsein_vec_list[i](matrix_input.transpose(1,2)).transpose(1,2)
        if self.use_bit_wise==True and self.use_vec_wise==False:
            bit_input=torch.cat([concatenated_emb,dense],dim=1)
            bit_output=self.karsein_bit(bit_input)
            logit=torch.sigmoid(bit_output)
        elif self.use_bit_wise==True and self.use_vec_wise==True:
            batch_size=matrix_output.shape[0]
            vec_pred_input=matrix_output.view(batch_size,-1)
            vec_logit=self.vec_pred_layer(vec_pred_input)
            bit_input=torch.cat([concatenated_emb,dense],dim=1)
            bit_output=self.karsein_bit(bit_input)
            bit_logit=torch.sigmoid(bit_output)
            logit=torch.sigmoid(vec_logit+bit_logit)
        elif self.use_bit_wise == False and self.use_vec_wise==True:
            batch_size=matrix_output.shape[0]
            pred_input=matrix_output.view(batch_size,-1)
            logit=self.vec_pred_layer(pred_input)
            
            logit=torch.sigmoid(logit)
        
        return logit
            
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        layer_list=[]
        if self.use_bit_wise:
            layer_list.append(self.karsein_bit)
        if self.use_vec_wise:
            layer_list.extend(self.karsein_vec_list)
        
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in layer_list
        )
        
    def plot(self,folder='./img',x=None):
        dense, concatenated_emb, matrix_X1=self.forward_(x)
        
        if self.use_bit_wise:
            
            bit_input=torch.cat([concatenated_emb,dense],dim=1)
            self.karsein_bit.plot(f'{folder}_bit','karsein_bit_network',bit_input)
        
        if self.use_vec_wise:
            matrix_output=matrix_X1.clone()
            for i in range(len(self.karsein_vec_list)):
                if i==0:
                    crossed_features=self.cross_feature(matrix_X1,matrix_output)
                    matrix_input=torch.cat([matrix_X1,crossed_features],dim=1)
                    matrix_output=self.karsein_vec_list[i](matrix_input.transpose(1,2)).transpose(1,2)
                    self.karsein_vec_list[i].plot(f'{folder}_vec_{i}',f'karsein_vec_network_{i}',matrix_input.transpose(1,2))
                else:
                    matrix_input=matrix_output
                    matrix_output=self.karsein_vec_list[i](matrix_input.transpose(1,2)).transpose(1,2)
                    self.karsein_vec_list[i].plot(f'{folder}_vec_{i}',f'karsein_vec_network_{i}',matrix_input.transpose(1,2))
        
        
    def prune(self,x,threshold):
        dense, concatenated_emb, matrix_X1=self.forward_(x) 
        
        if self.use_bit_wise: 
            print('Pruning karsein_bit_network...')
            bit_input=torch.cat([concatenated_emb,dense],dim=1)
            self.karsein_bit.prune(bit_input,threshold)
            
        if self.use_vec_wise:
            matrix_output=matrix_X1.clone()
            for i in range(len(self.karsein_vec_list)):
                print(f'Pruning {i}-th layer karsein_vec_network...')
                if i==0:
                    crossed_features=self.cross_feature(matrix_X1,matrix_output)
                    matrix_input=torch.cat([matrix_X1,crossed_features],dim=1)
                else:
                    matrix_input=matrix_output
                matrix_output=self.karsein_vec_list[i](matrix_input.transpose(1,2)).transpose(1,2)
                self.karsein_vec_list[i].prune(matrix_input.transpose(1,2),threshold)
            
       