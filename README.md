# HetGDN
An implementation of our paper "Heterogeneous Graph Diffusion Networks" (Submission for AAAI-2024).

Thank you for your interest in our work!  

# Motivation  
Existing heterogeneous graph neural networks commonly have difficulty capturing high-order relationships.  
- Metapath-based GNNs : Only focus on selected high-order relationships  
- Metapath-free GNNs : Hard to stack deeply because of over-parameterization and over-squashing (because of a large number of relation-aware parameters)
## We aim to capture comprehensive high-order relationships with graph diffusion without metapaths and relation-aware parameters.  

# Challenges    
Existing diffusion-based GNNs defined in homogeneous graphs (e.g., APPNP, GraphHeat, GDC) encourage node representations of connected nodes similar.  
- However, nodes in heterogeneous graphs also contain type information.
- It is undesirable that connected nodes have similar type information.

# Methods  
Our HetGDN introduces two techniques.  
- (1) Type-Adaptive Normalization disentangles type-information and type-independent information contained in hidden representations.   
- (2) Type-Independent Matching Regularization encourages connected nodes to share type-independent information (not type-information)    

With the aid of those techniques, HetGDN successfully captures high-order relationships in heterogeneous graphs without over-parameterization and over-squashing  
![overview](https://github.com/SeongJinAhn/HetGDN/assets/37531907/189708a2-b88f-412c-a65c-9f80d5771912)


# Dependencies
Recent versions of the following packages for Python 3 are required:

* Anaconda3
* Python 3.7.11  
* Pytorch 1.10.2  
* torch_geometric 2.0.4  
* torch_scatter 2.0.9  

# Easy Run
> python main.py --dataset=DBLP --mode=rw --hops=15
