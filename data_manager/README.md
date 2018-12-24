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
    
### 2. Encapsulating data feature to Dataset  
Encapsulating the features extracted in the previous step into Dataset class.  
### 3. Triplet Wrapper


### Code structure