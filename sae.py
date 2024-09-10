import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

def r2_per_channel(predicted, actual):
    inner_dim = actual.shape[-1]
    predicted = predicted.view(-1, inner_dim)
    actual = actual.view(-1, inner_dim)
    channel_means = torch.mean(actual, dim=-2)
    avg_squared_error_per_channel = torch.mean((actual - channel_means)**2, dim=-2)
    avg_squared_error_predicted = torch.mean((predicted - actual)**2, dim=-2)
    return 1 - avg_squared_error_predicted / avg_squared_error_per_channel

class TopKSparseAutoencoder(nn.Module):
    def __init__(self, embedding_size, n_features, topk):
        super(TopKSparseAutoencoder, self).__init__()
        self.embedding_size = embedding_size
        self.n_features = n_features
        self.topk = topk
        
        # self.encode = nn.Linear(embedding_size, n_features, bias=False)
        # self.decode = nn.Linear(n_features, embedding_size, bias=False)
        self.encode = nn.Linear(embedding_size, n_features, bias=True) # BADDDDDD
        self.decode = nn.Linear(n_features, embedding_size, bias=True)
        self.bias = nn.Parameter(torch.zeros(embedding_size))
        # set encode to have activations dim to have l2 norm randomly between 0.05 and 0.1
        # set encoder weight to have feature vector length of 1
        # then scale each to be a uniform distribution between 0.05 and 1.0
        direction_lengths = (torch.rand(n_features) * 0.95 + 0.05).unsqueeze(-1)
        self.encode.weight.data = F.normalize(self.encode.weight.data, p=2, dim=-1) * direction_lengths

        # initialize decode weight as transpose of encode weight
        self.decode.weight.data = self.encode.weight.data.t()

        
    def keep_topk(self, tensor, k):
        values, indices = torch.topk(tensor, k)
        mask = torch.zeros_like(tensor)
        mask.scatter_(-1, indices, 1)
        result = tensor * mask
        return result, values, indices
    
    
    def forward(self, x, return_r2=False):
        x = x - self.bias
        encoded = self.encode(x)
        encoded, values, indices = self.keep_topk(encoded, self.topk)
        decoded = self.decode(encoded) + self.bias
        mse = torch.mean((x - decoded)**2)
        
        result_dict = {'encoded': encoded, 'decoded': decoded, 
                       'topk_idxs': indices, 'topk_values': values, 
                       'mse': mse} 
        if return_r2:
            result_dict['mean_r2'] = torch.mean(r2_per_channel(decoded, x))

        return result_dict
# sae = TopKSparseAutoencoder(128, 1024, 24)
# print(sae.decode.weight.shape)