# TSception
This is the PyTorch implementation of TSception V2 using [DEAP](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/) dataset in our paper:

Yi Ding, Neethu Robinson, Su Zhang, Qiuhao Zeng, Cuntai Guan, "TSception: Capturing Temporal Dynamics and Spatial Asymmetry from EEG for Emotion Recognition", under review of _**IEEE Transactions on Affective Computing**_, preprint available at [arXiv](https://arxiv.org/abs/2104.02935)

It is an end-to-end multi-scale convolutional neural network to do classification from raw EEG signals. Previous version of TSception(IJCNN'20) can be found at [website](https://github.com/deepBrains/TSception)

# Prepare the python virtual environment
Please create an anaconda virtual environment by:

> $ conda create --name TSception

Activate the virtual environment by:

> $ conda activate TSception

Install the requirements by:

> $ pip3 install -r requirements.txt

Note: The requirements.txt is inside the ./code/ folder 
# Run the code
Please download the DEAP dataset at [website](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/). Please place the "data_preprocessed_python" folder at the same location of the script (./code/). Go to the ./code/ folder where the scripts are by typing below command in terminal:

> $ cd ./code/

To run the code for arousal dimension, please type the following command in terminal:

> $ python3 main-DEAP.py --data-path './data_preprocessed_python/' --label-type 'A'

To run the experiments for valance please set the --label-type 'V'. The results will be saved into "result.txt" located at the same place as the script. 

# Reproduce the results
We highly suggest to run the code on a Ubuntu 18.04 or above machine using anaconda with the provided requirements to reproduce the results. 
You can also download the saved model at [website](https://drive.google.com/file/d/1HRr0IuWlvuJgPc6jVvo-QxMxKuugsGTw/view?usp=sharing) to reproduce the results in the paper. After extracting the downloaded "save.zip", please place it at the same location of the scripts, run the code by:

> $ python3 main-DEAP.py --data-path './data_preprocessed_python/' --label-type 'A' --reproduce True

# Acknowledgment
The author would like to thank Su Zhang, Quihao Zeng and Tushar Chouhan for checking the code

# Cite
Please cite our paper if you use our code in your own work:

```
@misc{ding2021tsception,
      title={TSception: Capturing Temporal Dynamics and Spatial Asymmetry from EEG for Emotion Recognition}, 
      author={Yi Ding and Neethu Robinson and Su Zhang and Qiuhao Zeng and Cuntai Guan},
      year={2021},
      eprint={2104.02935},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
OR

```
@INPROCEEDINGS{9206750,
  author={Y. {Ding} and N. {Robinson} and Q. {Zeng} and D. {Chen} and A. A. {Phyo Wai} and T. -S. {Lee} and C. {Guan}},
  booktitle={2020 International Joint Conference on Neural Networks (IJCNN)}, 
  title={TSception:A Deep Learning Framework for Emotion Detection Using EEG}, 
  year={2020},
  volume={},
  number={},
  pages={1-7},
  doi={10.1109/IJCNN48605.2020.9206750}}
```
