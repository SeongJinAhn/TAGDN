# TAGDN
An implementation of our paper "TAGDN: Type-Aligning Graph Diffusion Network for Heterogeneous Graph Analysis", which is accepted by Neural Computing and Applications.

# Motivation  
Existing metapath-free GNNs (e.g., ie-HGCN, FastGTN, SR-HGN, MHGCN+) face difficulty capturing long-range dependencies between distant nodes because of over-squashing and over-smoothing.

Graph diffusion is a promising direction to mitigate over-squashing and over-smoothing.

However, existing graph diffusion networks (e.g., APPNP, GraphHeat) do not account for the heterogeneity of attributes and relationships.

#### We design a Type-Aligning Graph Diffusion Network that captures long-range dependencies in heterogeneous graphs without suffering over-smoothing and over-squashing.

![overview](https://github.com/SeongJinAhn/TAGDN/blob/main/Figures/overview.png?raw=true)

# Datasets
The used datasets are available.  
DBLP : https://www.dropbox.com/s/yh4grpeks87ugr2/DBLP_processed.zip?e=1&dl=0  
IMDB : https://www.dropbox.com/s/g0btk9ctr1es39x/IMDB_processed.zip?e=1&dl=0  
ACM, Freebase : [https://github.com/meettyj/HGMAE/tree/master/data]  

# Dependencies
Recent versions of the following packages for Python 3 are required:

* Anaconda3
* Python 3.7.11  
* Pytorch 1.10.2  
* torch_geometric 2.0.4  
* torch_scatter 2.0.9  

# Easy Run
> python main.py --dataset=DBLP --diffusion=ppr --num_layers=15 --alpha=0.1
