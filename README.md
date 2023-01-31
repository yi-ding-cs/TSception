# TSception
This is the PyTorch implementation of TSception V2 using [DEAP](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/) dataset in our paper:

Yi Ding, Neethu Robinson, Su Zhang, Qiuhao Zeng, Cuntai Guan, "TSception: Capturing Temporal Dynamics and Spatial Asymmetry from EEG for Emotion Recognition", accepted as a regular paper in _**IEEE Transactions on Affective Computing**_, preprint available at [arXiv](https://arxiv.org/abs/2104.02935)

It is an end-to-end multi-scale convolutional neural network to classify EEG signals. Previous version of TSception(IJCNN'20) can be found at [website](https://github.com/deepBrains/TSception)

# Prepare the python virtual environment

Please go to the working directory by:

> $ cd ./code

Please create an anaconda virtual environment by:

> $ conda create --name TSception python=3.8

Activate the virtual environment by:

> $ conda activate TSception

Install the requirements by:

> $ pip3 install -r requirements.txt
 
# Run the code
Please download the DEAP dataset at [website](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/). Please place the "data_preprocessed_python" folder at the same location as the script (./code/).

To run the code for the arousal dimension, please type the following command in the terminal:

> $ python3 main-DEAP.py --data-path './data_preprocessed_python/' --label-type 'A'

To run the experiments for the valance please set the --label-type 'V'. The results will be saved into "result.txt" located in the same place as the script. 

# Reproduce the results
We highly suggest running the code on a Ubuntu 18.04 or above machine using anaconda with the provided requirements to reproduce the results. 
You can also download the saved model at [website](https://drive.google.com/file/d/1HRr0IuWlvuJgPc6jVvo-QxMxKuugsGTw/view?usp=sharing) to reproduce the results in the paper. After extracting the downloaded "save.zip", please place it at the same location of the scripts, as (./code/save/). Please run the code by:

> $ python3 main-DEAP.py --data-path './data_preprocessed_python/' --label-type 'A' --reproduce

# Apply TSception to other datasets
If you are interested to apply TSception to other datasets, you can use generate_TS_channel_order() in utils.py to generate the suitable channel order for TSception, and reorder your data on channel dimension before feeding the data to TSception as what we did in reorder_channel() in prepare_data_DEAP.py

## Example of Usage
```python
from utils import generate_TS_channel_order
from networks import TSception

original_order = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']   # DEAP is used as an example

TS_order = generate_TS_channel_order(original_order)   # generate proper channel orders for the asymmetric spatial layer in TSception

data = torch.randn(1, 1, 32, 512)   # (batch_size=1, cnn_channel=1, EEG_channel=32, data_points=512)

idx = []
for chan in TS_order:
    idx.append(original_order.index(chan))
data = data[:, :, idx, :]    # (batch_size=1, cnn_channel=1, EEG_channel=28, data_points=512) Some channels are not selected, hence EEG channel becomes 28.

TS = TSception(
    num_classes=2, 
    input_size=(1, 28, 512),
    sampling_rate=128, 
    num_T=15,   # num_T controls the number of temporal filters
    num_S=15,   # num_S controls the size of hidden embedding due to the global average pooling. Please increase it if you need a larger model capacity, e.g., subject-independent case
    hidden=32,   
    dropout_rate=0.5
)

preds = TS(data)
```

# Acknowledgment
The author would like to thank Su Zhang, Quihao Zeng, and Tushar Chouhan for checking the code

# Cite
Please cite our paper if you use our code in your work:

```
@ARTICLE{9762054,
  author={Ding, Yi and Robinson, Neethu and Zhang, Su and Zeng, Qiuhao and Guan, Cuntai},
  journal={IEEE Transactions on Affective Computing}, 
  title={TSception: Capturing Temporal Dynamics and Spatial Asymmetry from EEG for Emotion Recognition}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TAFFC.2022.3169001}}
```
