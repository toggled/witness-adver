import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pandas as pd
import networkx as nx
import os,pickle

from utils import load_data
from persistence_image import persistence_image
from DNN import CNN
from lazywitness import * 


def computeLWfeatures(filename='cora.edgelist'):
	""" Computes LW persistence pairs / if pre-computed loads&returns them. """
	if os.path.isfile('cora.pd.pkl'):
		with open('cora.pd.pkl','rb') as f:
			PD = pickle.load(f)
	else:	
		G = nx.read_edgelist(filename)
		L = int(len(G.nodes)*0.25) # Take top 25% maximal degree nodes as landmarks
		landmarks,dist_to_cover = getLandmarksbynumL(G, L = L,heuristic='degree')
		DSparse,INF = get_sparse_matrix(G,dist_to_cover,landmarks) # Construct sparse LxL matrix
		resultsparse = ripser(DSparse, distance_matrix=True)
		resultsparse['dgms'][0][resultsparse['dgms'][0] == math.inf] = INF # Replacing INFINITY death with a finite number.
		PD = resultsparse['dgms'][0] # H0
		
		with open('cora.pd.pkl','wb') as f:
			pickle.dump(PD,f)
	return PD

# load dataset
dataset_name = 'cora'
G, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(dataset_name, return_nxgraph=True)

# generate persistence image
PD = computeLWfeatures() 

# toy PD
#PD_birth = np.random.rand(5,1)
#PD_death = PD_birth + np.random.rand(5,1)
#PD = np.concatenate([PD_birth, PD_death], axis=1)
# PI with 50 x 50 resolution based on above PD
PI = persistence_image(PD, resolution = [100, 100])

# deep neural networks on PI
# we first generate simulated PIs (in one batch)
# batch size is 16
PIs_in_batch = np.random.rand(16, 100 ,100) # (B, resolution, resolution)
PIs_in_batch = PIs_in_batch

device = 0
PIs_in_batch = torch.FloatTensor(PIs_in_batch).unsqueeze(dim = 1).to(device)

# CNN model
dim_out = 64 # pre-defined output dimension
CNN_model = CNN(dim_out = dim_out)
output = CNN_model(PIs_in_batch) # (B, dim_out)
