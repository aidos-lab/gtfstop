#Import packages 
import os 
import peartree as pt
import matplotlib.pyplot as plt
import osmnx as ox
import networkx as nx
import numpy as np
from ripser import ripser
from persim import plot_diagrams

import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.data import InMemoryDataset, download_url

import warnings
warnings.filterwarnings('ignore')

#Generate graphs for ICE and Regional Data 

#1. For ICE data 
#Set path variable to read ICE data (modify this accordingly based on where you run it)
path = "data/ice.zip"

feed = pt.get_representative_feed(path)
#Just specifying a time period between which you can get the statistics - this can be modified based on the stop_times.txt file
start = 7*60*60  # 7:00 AM
end = 10*60*60  # 10:00 AM

#Load feed as a transit graph using the given start and end times and plot it!
G_ice = pt.load_feed_as_graph(feed, start, end)
pt.generate_plot(G_ice)


#2. For Regional data 
#Set path variable to read regional data (modify this accordingly based on where you run it)
path_reg = "data/regional.zip"

feed_reg = pt.get_representative_feed(path_reg)
#Just specifying a time period between which you can get the statistics - this can be modified based on the stop_times.txt file
start = 7*60*60  # 7:00 AM
end = 10*60*60  # 10:00 AM

#Load feed as a transit graph using the given start and end times and plot it!
G_reg = pt.load_feed_as_graph(feed_reg, start, end)
pt.generate_plot(G_reg)


#Pytorch Geometric Dataset Creation 
def nx_to_pyg_data(graph):
    """
    Convert a networkx graph to a PyTorch Geometric Data object.
    """
    # Create a mapping from nodes to integers
    node_mapping = {node: i for i, node in enumerate(graph.nodes())}

    # Get the adjacency information with remapped node indices
    edge_index = torch.tensor([(node_mapping[u], node_mapping[v]) for u, v in graph.edges()]).t().contiguous()

    # Extract node features with remapped node indices
    x_list = []
    for node in node_mapping:
        node_data = graph.nodes[node]
        boarding_cost = node_data['boarding_cost']
        modes = len(node_data['modes'])  # Transforming 'modes' list to its length
        y = node_data['y']
        x_coord = node_data['x']
        x_list.append([boarding_cost, modes, y, x_coord])

    x = torch.tensor(x_list, dtype=torch.float)

    edge_features = []

    # Extract edge features
    edge_features = []
    for u, v, key in graph.edges(keys=True):
        edge_data = graph[u][v][key]
        length = float(edge_data.get('length', 0))  # convert to float, default to 0 if 'length' is not found
        mode = int(edge_data.get('mode', '0') == 'transit')  # convert 'mode' to binary representation (1 if 'transit', 0 otherwise)
        edge_features.append([length, mode])

    edge_attr = torch.tensor(edge_features, dtype=torch.float)

    # Return as PyTorch Geometric Data
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, nx_graphs, transform=None, pre_transform=None, pre_filter=None):
        self.nx_graphs = nx_graphs
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []  # We're directly passing in NetworkX graphs so we don't need raw file names

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # No downloading needed as we're directly passing in NetworkX graphs
        pass

    def process(self):
        # Convert the NetworkX graphs to PyTorch Geometric Data objects
        data_list = [nx_to_pyg_data(graph) for graph in self.nx_graphs]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
     

dataset = MyOwnDataset(root="data", nx_graphs=[G_ice, G_reg])