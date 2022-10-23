# HetGDCN
An implement of our paper "Diffusion-based Graph Convolution Networks for Heterogeneous Graphs using Type-Adaptive Normalization" (Submission for TheWebConf 2023).

Thank you for your interest in our works!  

# Motivation
We find out that GAEs make embeddings of isolated nodes (nodes with no-observed links) zero vectors regardless of their feature information.  
Our works try to distinguish embeddings of isolated nodes by reflecting their feature information better.
![overview](https://user-images.githubusercontent.com/37531907/197387805-0bb48489-284c-4fc8-af92-c014dc6f62c0.PNG)
![type_adaptive](https://user-images.githubusercontent.com/37531907/197387808-7bc26a92-6379-4450-8a7b-a39a0cdea4de.PNG)

# Dependencies
Recent versions of the following packages for Python 3 are required:

* Anaconda3
* Python 3.7.11  
* Pytorch 1.10.2  
* torch_geometric 2.0.4  
* torch_scatter 2.0.9  

# Easy Run
> python main.py --dataset=DBLP --mode=appnp
