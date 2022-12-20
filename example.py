import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pandas as pd
import networkx as nx
import os,pickle
from torch_geometric.data import Data
import torch_geometric.datasets
import torch_geometric.transforms as T
import sys
from utils import load_data
from persistence_image import persistence_image
from DNN import CNN
from GCN import GCNTraining
from torch_geometric.utils import to_networkx
from lazywitness import * 


def computeLWfeatures(G,dataset_name, landmarkPerc=0.25,heuristic = 'degree'):
	"""
	 Computes LW persistence pairs / if pre-computed loads&returns them. 
	landmarkPerc = %nodes fo G to be picked for landmarks. [ Currently we are selecting landarmks by degree]
	heuristic = landmark selection heuristics ('degree'/'random') 
	Returns only H0 PD	
	"""
	if os.path.isfile(dataset_name+'.pd.pkl'):
		with open(dataset_name+'.pd.pkl','rb') as f:
			PD = pickle.load(f)
	else:	
		L = int(len(G.nodes)*landmarkPerc) # Take top 25% maximal degree nodes as landmarks
		landmarks,dist_to_cover = getLandmarksbynumL(G, L = L,heuristic='degree')
		DSparse,INF = get_sparse_matrix(G,dist_to_cover,landmarks) # Construct sparse LxL matrix
		resultsparse = ripser(DSparse, distance_matrix=True)
		resultsparse['dgms'][0][resultsparse['dgms'][0] == math.inf] = INF # Replacing INFINITY death with a finite number.
		PD = resultsparse['dgms'][0] # H0
		
		with open(dataset_name+'.pd.pkl','wb') as f:
			pickle.dump(PD,f)
	return PD

# load dataset
dataset_name = 'cora'
dataset = torch_geometric.datasets.Planetoid('./data/'+dataset_name,dataset_name,transform=T.NormalizeFeatures())

# Train a 2 layer GCN with cross-entropy loss
GCNLoss, gcnoutput = GCNTraining(dataset)
print('GCN(G) output shape & output: ',gcnoutput.shape,'\n', gcnoutput)
# Transform pyg graph to networkx for computing persistence
G = to_networkx(dataset[0],to_undirected=True)

# Compute PD
# PD = computeLWfeatures(G,dataset_name,landmarkPerc=0.25,heuristic = 'degree')
PD = computeLWfeatures(G,dataset_name,landmarkPerc=0.25,heuristic = 'random')

# Compute PI from PD
n = len(G.nodes)
PI = persistence_image(PD, resolution = [n, n])
print('PI shape & output: ',PI.shape,'\n', PI)

# deep neural networks on PI
# we first generate simulated PIs (in one batch)
# batch size is 16
#PIs_in_batch = np.random.rand(1, 100 ,100) # (B, resolution, resolution)
#PIs_in_batch = PIs_in_batch
# PIs_in_batch = PI

device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# PIs_in_batch = torch.FloatTensor(PIs_in_batch).unsqueeze(dim=0).unsqueeze(dim = 1).to(device)

# CNN model
# dim_out = 10 # pre-defined output dimension
# CNN_model = CNN(dim_out = dim_out).to(device)
# cnnoutput = CNN_model(PIs_in_batch) # (B, dim_out)
# print('CNN output shape & output: ',cnnoutput.shape, cnnoutput)

lin = torch.nn.Linear(n,16).to(device)
mlpoutput = lin(torch.FloatTensor(PI).to(device))
print('MLP(PD) output shape & output: ',mlpoutput.size(),'\n', mlpoutput)

print('GCN(G) + MLP(PD) concatenated shape: ',torch.hstack((gcnoutput,mlpoutput)).size())
