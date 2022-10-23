# HetGDCN
An implement of our paper "Diffusion-based Graph Convolution Networks for Heterogeneous Graphs using Type-Adaptive Normalization" (Submission for TheWebConf 2023).

Thank you for your interest in our works!  

# Motivation
(1) Diffusion-based GCNs have made success on analyzing graph data thanks to their robustness of structural noise.  
    A relatively few studies have examined diffusion-based GCNs on heterogeneous graphs.  
    
      
(2) In diffusion-based GCNs, graphs are assumed to be homophilic or heterophilic.  
    However, we find out that heterogeneous graphs are neither homophilic nor heterophilic.   

# Methods
In this paper, we assume two following statements.  
(1) Each node in heteogeneous graphs has its own type- and content- information.  
(2) Connected nodes in heterogeneous graphs have similar content information (not type information).  

Due to the similarity of content information between adjacent nodes, graph diffusions are proper for diffusing content information (not type information).  
Hence, we try to decouple type- and content- information of nodes.  
Our paper present a novel type-adaptive (de)normalization to analyze heterogeneous graphs by decoupling type- and content- information of nodes. 
![type_adaptive](https://user-images.githubusercontent.com/37531907/197388411-d38899a9-9571-4c73-bb49-bf8c909ae32e.PNG)

With the assumption, our HetGDCN diffuses content information with graph diffusions.  
Then, we recombine type- and content- information.  
![overview](https://user-images.githubusercontent.com/37531907/197388403-df37803a-4f06-4c93-867a-7f1d4ec0a761.PNG)

# Dependencies
Recent versions of the following packages for Python 3 are required:

* Anaconda3
* Python 3.7.11  
* Pytorch 1.10.2  
* torch_geometric 2.0.4  
* torch_scatter 2.0.9  

# Easy Run
> python main.py --dataset=DBLP --mode=appnp
