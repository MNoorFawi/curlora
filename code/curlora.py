import numpy as np
from torch import nn
import torch


def compute_selection_probabilities(A):
    column_norms_squared = torch.sum(A**2, axis=0)
    row_norms_squared = torch.sum(A**2, axis=1)
    total_sum_squares = torch.sum(column_norms_squared)
    column_probs = column_norms_squared / total_sum_squares
    row_probs = row_norms_squared / total_sum_squares
    return column_probs, row_probs


def select_indices_with_replacement(probs, k):
    #inverted_P = 1 / (probs + 0.001)
    inverted_P = (1 / (probs + 0.001)).float()

    # Normalize the inverted probabilities
    probs = inverted_P / inverted_P.sum()
    
    if probs.device == "cuda":
        probs = probs.cpu().numpy()
    else:
        probs = probs.numpy()
    
    return np.random.choice(len(probs), size=k, replace=True, p=probs)


def adjust_duplicates(selected_indices, A, axis):
    unique_indices, counts = np.unique(selected_indices, return_counts=True)
    adjusted_matrix = A[:, unique_indices] if axis == 1 else A[unique_indices, :]
    
    for idx, count in enumerate(counts):
        if count > 1:
            scaling_factor = np.sqrt(count)
            if axis == 1:
                adjusted_matrix[:, idx] *= scaling_factor
            else:
                adjusted_matrix[idx, :] *= scaling_factor
    
    return adjusted_matrix, unique_indices


def cur_decomposition(A, c):
    r = c
    column_probs, row_probs = compute_selection_probabilities(A)
    selected_columns = select_indices_with_replacement(column_probs, c)
    selected_rows = select_indices_with_replacement(row_probs, r)
    
    C = A[:, selected_columns]
    R = A[selected_rows, :]
    
    U = torch.empty(C.shape[1], R.shape[0])
    U = torch.zeros_like(U).to("cuda") #* 0.00
    
    return C, U, R


class CURModule(nn.Module):
    def __init__(self, W, rank):
        super(CURModule, self).__init__()
        C, U, R = cur_decomposition(W, rank)
        self.C = C * 1.0
        self.R = R * 1.0
        self.U = nn.Parameter(U)
        #self.d = torch.nn.Dropout(0.05)

    def forward(self, x):
        W_approx = torch.matmul(torch.matmul(self.C, self.U), self.R)
        x = torch.matmul(x, W_approx.t())
        #x = self.d(x)
        return x


class CURLoRAMLP(nn.Module):
    def __init__(self, base_model, rank=8, alpha=1):
        super(CURLoRAMLP, self).__init__()
        self.base_model = base_model
        self.rank = rank
        self.alpha = alpha
        # Identify the layer to adapt (the last layer)
        layer_to_adapt = base_model.layers[-1]
        # Freeze the parameters of the base model
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.cur_module = CURModule(layer_to_adapt.weight, self.rank)
    
    def forward(self, x):
        x = self.base_model.layers[:-1](x)  # Use all layers except the last one
        x_0 = torch.matmul(x, self.base_model.layers[-1].weight.t()) 
        x_adapted = self.cur_module(x)
        x = x_0 + (self.alpha * x_adapted) + self.base_model.layers[-1].bias
        return x


class LinearWithCURLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.curlora = CURModule(linear.weight, rank)
        self.rank = rank
        self.alpha = alpha

    def forward(self, x):
        x_0 = self.linear(x)
        x_adapted = self.curlora(x)
        x = x_0 + (self.alpha * x_adapted) #+ self.linear.bias
        return x
