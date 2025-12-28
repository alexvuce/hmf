import torch
import torch.nn as nn


class LFM(nn.Module):
    def __init__(self, n_rows: int, n_cols: int, rank_k: int):
        super().__init__()
        self.k = rank_k
        self.U = nn.Parameter(torch.randn((n_rows, rank_k)), requires_grad=True)
        self.V = nn.Parameter(torch.randn((n_cols, rank_k)), requires_grad=True)

    def forward(self, i: torch.LongTensor, j: torch.LongTensor):
        return torch.sum(self.U[i] * self.V[j], 1)


class DLFM(nn.Module): 
	def __init__(self, n_rows: int, n_cols: int, rank_k: int, dropout: float = None):
		super().__init__()
		self.k = rank_k        
		self.dropout = dropout
		self.U = nn.Parameter(torch.randn((n_rows, rank_k)), requires_grad=True)
		self.V = nn.Parameter(torch.randn((n_cols, rank_k)), requires_grad=True)
		self.H1 = nn.Linear(2 * rank_k, 8 * rank_k, bias=False)
		self.gelu = nn.GELU()  
		self.H2 = nn.Linear(8 * rank_k, 2 * rank_k, bias=False)
		if self.dropout:   
			self.DO = nn.Dropout(self.dropout)
		self.logit = nn.Linear(2 * rank_k, 1, bias=False)

	def forward(self, i: torch.LongTensor, j: torch.LongTensor):
		x = torch.concat((self.U[i], self.V[j]), 1)
		x = self.H1(x)
		x = self.gelu(x)      
		x = self.H2(x)
		if self.dropout: 
			x = self.DO(x)
		x = self.logit(x)
		return x.squeeze(-1)


class STTLFM(nn.Module): 
	"""
	Shared Two Tower Model: Feedforward layers shared between embeddings
	"""      
	def __init__(self, n_rows: int, n_cols: int, rank_k: int, dropout: float = None):		
		super().__init__()
		self.k = rank_k
		self.dropout = dropout
		self.U = nn.Parameter(torch.randn((n_rows, rank_k)), requires_grad=True)
		self.V = nn.Parameter(torch.randn((n_cols, rank_k)), requires_grad=True)
		self.H1 = nn.Linear(rank_k, 16 * rank_k, bias=False)
		self.gelu = nn.GELU()  
		if self.dropout:   
			self.DO = nn.Dropout(dropout)
		self.H2 = nn.Linear(16 * rank_k,  rank_k, bias=False)

	def forward(self, i: torch.LongTensor, j: torch.LongTensor):
		u, v = self.U[i], self.V[j]
		u, v = self.H1(u), self.H1(v)
		u, v = self.gelu(u), self.gelu(v) 
		if self.dropout: 
			u, v = self.DO(u), self.DO(v)
		u, v = self.H2(u), self.H2(v)
		return torch.sum(u * v, 1)