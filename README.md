# HetGDN
An implement of our paper "Diffusion-based Graph Convolution Networks for Heterogeneous Graphs using Adaptive-Type Normalization" (Submission for KDD 2023).

Thank you for your interest in our works!  

# Motivation
(1) GNNs leveraging graph diffusion have made success on analyzing graph data thanks to their robustness of structural noise.  
    A relatively few studies have examined diffusion-based GCNs on heterogeneous graphs.  
    
(2) Nodes in heterogeneous graphs have different features corresponding to their types.
    To leverage diffusion in heterogeneous graphs, we need to handle the heterogeneity among node features.

# Methods
To handle the feature discrepancies among nodes, we propose Adaptive Type Normalization (AdaTN) that project features of a node into the feature space of nodes with different types. 
This process enables features of nodes to be compatible regardless of type differences.
Following figure shows a process to project a latent vector of a movie v into the latent space of u.
![AdaTN](https://user-images.githubusercontent.com/37531907/216320923-676fe8b1-35ee-405e-81bf-c156a1e51691.png)

With the proposed AdaTN, features of a node and those of other nodes can be located into the same latent space.
This indicates that we are able to leverage diffusions to smooth higher-order neighbors in heterogeneous graphs.
![overview](https://user-images.githubusercontent.com/37531907/216320862-ce215572-00ec-4be4-8d6d-5d24e53ac7ac.png)


# Dependencies
Recent versions of the following packages for Python 3 are required:

* Anaconda3
* Python 3.7.11  
* Pytorch 1.10.2  
* torch_geometric 2.0.4  
* torch_scatter 2.0.9  

# Easy Run
> python main.py --dataset=DBLP --mode=rw
