# Triplet loss on Acoustic Scene Classification(ASC)-PyTorch
PyTorch implementation of triplet networks for learning embeddings.  

Triplet networks are useful to learn mapping from input to a compact 
Euclidean space where distances correspond to a measure of similarity.  

# Installation
Requires pytorch 0.4 with torchvision 0.2.1  

# Code structure
- **data_manager folder**  
    - Please see the data_manager/[README.md](https://github.com/Howe-Young/asc_triplet/blob/master/data_manager/README.md) for details.  

- **networks.py** 
    - some network classes. e.g. vggish_bn. it's VGG-like network architecture.
  
- **losses.py**
    - *OnlineTripletLoss* class -triplet loss for triplets of embeddings.  
  
- **metrics.py**
    - Sample metrics that can be used with fit function from trainer.py 

- **trainer.py** 
    - fit - unified function for training a network with different number of 
    inputs and different types of loss functions.
 
- **utils.py**
    - *FunctionNegativeTripletSelector* class -generating triplets based 
    on embeddings and ground truth class labels.
    - *plot_embeddings*, *extract_embeddings* are function of learned embeddings visualization.
  
- **experiment folder**
    - **classification_baseline.py** - A baseline of classification code.  
    - **hard_triplet_baseline.py** - A random hard selection of triplets baseline code.

- **jupyter_script**
    - some experiment results.
    
# TODO
1. change network architecture.
2. novel tripets selection strategy
3. pickup classifier.
4. verification on embedding metric.
