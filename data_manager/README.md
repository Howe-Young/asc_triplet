## prepare data
Extracting features from raw wav file and doing some transforms.  
**Distribution of various types of data:**  

| Type | Device | numbers |  
|:----: | :----: | :----: |  
| train | a | 6122|
| train | b | 540 |
| train | c | 540 |
| train | p | 540 |
| train | A | 5582|
| train | abc| 7202|
| test | a | 2518|
| test | b | 180 |
| test | c | 180 |
| test | p | 180 |
| test | A | 2338 |
| test | abc| 2878 |

**Note**:  
a indicates training or testing data on all device A  
b indicates training or testing data on all device B  
c indicates training or testing data on all device C  
p indicates training or testing data parallel to device BC in device A  
A indicates training or testing data exclude p in device A 

### 1. Extracting mel-spectrogram feature from raw wav file  
Using librosa library extract the feature of mel-spectrogram, the size is 40x500, and save it to h5 file.
```
[dcase17]
dev_path = /home/songhongwei/data_home/DCASE2017-baseline-system/applications/data/TUT-acoustic-scenes-2017-development/
eva_path = /home/songhongwei/data_home/DCASE2017-baseline-system/applications/data/TUT-acoustic-scenes-2017-evaluation/

[logmel]
sr = 44100
n_fft = 1764
hop_length = 882
n_mels = 40
```
### 2. Encapsulating data feature to Dataset  
Encapsulating the features extracted in the previous step into Dataset class.  
### 3. Triplet Wrapper
Encapsulating the Dataset to DataLoader for next step iteration.  


### Code structure  
- **data_manager.cfg**
    - the path of dataset.
- **data_prepare.py**  
    - *Dcase18TaskbData* class - Extract the mel-spectrogram from wav file and save it to h5 file.  
- **datasets.py**  
    - *DevSet* class - wrapper for a MNIST-like dataset, returning specify mode and device dataset.  
- **datasets_wrapper.py**  
    - *TripletDevSet* class - wrapper for a MNIST-like dataset, returning random triplets(anchor, positive, negative).  
    - *BalancedBatchSampler* class - BatchSampler for DataLoader, randomly chooses n_classes and n_samples from each 
    class of a MNIST like dataset.  
- **mean_variance.py**  
    - *TaskbStandarizer* class - calculating the mean and variance of the specified data, 
    normalized the data with the specified mean and variance.
- **transformer.py**
    - *ToTensor* class - converting numpy to tensor.  
    - *Normalize* class - normalizing data with given mean and variance.  
        
