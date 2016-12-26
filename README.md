<div style="text-align:center"><img src ="https://sigvoiced.files.wordpress.com/2016/12/logo_name2.png" /></div>
-----------------------------------------

**Air-Script** is a CNN + Sequence to Sequence model for detecting handwriting on air using a Myo-Armband. It is Inspired by ‘Recursive Recurrent Nets with Attention Modeling for OCR in the Wild’ by [Chen-Yu & Simon, 2016](https://arxiv.org/abs/1603.03101). The idea was to use 1D-CNNs as feature extractors and a sequence to sequence model with Attention mechanism introduced by [Bahdanau et al., 2014](https://arxiv.org/abs/1409.0473) using LSTMs for variable length sequence classification.

The Implementation of [Attention-OCR](https://github.com/da03/Attention-OCR) was extremely helpful and Air-Script was built upon it. This project comes does not give results as expected and is currently under development. Probable issues have been tracked a list of further tasks have been mentioned at the end of this document and is being updated regularly too.

# Prerequsites
1. [Tensorflow](https://www.tensorflow.org/) (Version 0.11.0)
2. [Keras](http://keras.io/#installation) (Version 1.1.1)
3. [Distance (Optional)](http://www.cs.cmu.edu/~yuntiand/Distance-0.1.3.tar.gz)
3. Python 2.7

**I have tested it on an Ubuntu 14.04 and 15.04 with NVIDIA GeForce GT 740M and NVIDIA TITAN X Graphics card with Tensorflow running in a virtual environment. It should ideally run smoothly on any other system with the above packages installed in it.**

###Set Keras backend:

```
export KERAS_BACKEND=tensorflow
```

```
echo 'export KERAS_BACKEND=tensorflow' >> ~/.bashrc
```

### Install Distance (Optional):

```
wget http://www.cs.cmu.edu/~yuntiand/Distance-0.1.3.tar.gz
```

```
tar zxf Distance-0.1.3.tar.gz
```

```
cd distance; sudo python setup.py install
```

# The Idea

The idea was to learn features using a 1D Convolutional Neural Network and then align input sequences (raw IMU signals from Myo-Armband) with output sequences (sequence of characters) using a sequence to sequence model [Sutskever et al., 2014](https://arxiv.org/abs/1409.3215) with attention mechanism. Since our dataset had limited amount of data to train this network, an artificially generated datasets [(Appendix 1)](appendix) of various sizes were used to train the network.

# Model Architecture
![Model Architecture](https://sigvoiced.files.wordpress.com/2016/12/model-architecture2.png)
**[Figure 1]** The high level Model Architecture showing the encoder and decoder with attention

### Components

**The model consists of the following components**

1. **Encoder (1D-CNN):** Which like the encoder of an auto-encoder, encodes the input sequence into a feature vector which is decoded into the output sequence.  

2. **Decoder (LSTM Network + Attention):** A stacked LSTM network with attention was used to decode the feature vector into an output sequence. Both the feature vector and the output sequence were padded for alignment.

### CNN Model Architecture and specifications
![CNN Architecture](https://sigvoiced.files.wordpress.com/2016/12/cnn.png)

**[Figure 2]** The CNN architecture used in the experiments

### Bucketing
A bucketing technique has been used for padding variable length input and output sequences. The bucket specs for input and output sequence lengths are as follows,

* 400 : 4
* 800 : 8
* 1200 : 9
* 1800 : 11
* 1900 : 13

These buckets were selected by analyzing the distribution of the lengths of input and output sequences. The bucket sizes that made the input sequence length distribution uniform were selected keeping in mind the distribution of number of timesteps of input sequence per output sequence of a certain length.
The idea was to not only use the CNN to learn to extracting features from the sensor data  but also to fuse sensor data hierarchically.

![Char len hist](https://sigvoiced.files.wordpress.com/2016/12/char_len_hist.png)

**[Figure 3]** Character length histogram

![Char len hist](https://sigvoiced.files.wordpress.com/2016/12/seq_len_hist.png)

**[Figure 4]** Input sequence length histogram

### Different Model hyper parameters that were used for experiments and the corresponding dataset specs

Parameters | Model 1 | Model 2
--- | --- | --- 
**Min. Output Sequence Length** | 1 | 1 |
**Max. Output Sequence Length** | 10 | 10
**Number of training instances** | 100000 | 100000 
**Number of testing instances** | 1000 | 1000
**Batch Size** | 64 | 64
**LSTM Layers** | 3 | 2
**Initial learning rate** | 0.0001 | 0.001
**LSTM Hidden units** | 128 | 128
**Optimizer** | ADADelta | ADAM
**Epochs** | 38 (approx) (55,800 Iterations) | 10 (approx) (15,6000 Iterations)

***Other models were also trained and tested with slight variations in the hyper parameters and a smaller and larger dataset of sizes 1000 and 1000000 instances.***

### Data Preparation
The data used for training and evaluation has been acquired using Myo-Armband and [Pewter](https://github.com/sigvoiced/pewter) and the data creator module has been used to process the data.

# Results
The results are not as expected. The model overfits and can be made better by just a few tweaks. It learns and the perplexity reduces while training but the results are not good.

### Loss
![Loss curve](https://sigvoiced.files.wordpress.com/2016/12/lc.png)

**[Figure 5]** The loss curve of the models that were experimented with

### Perplexity
![Perplexity curve](https://sigvoiced.files.wordpress.com/2016/12/perp.png)

**[Figure 6]** The Perplexity curve of the models that were experimented with

### Output
![Output](https://sigvoiced.files.wordpress.com/2016/12/results.png)

**[Figure 7]** The results are shown in two parts each. The left hand side shows the Input data sequence. The right hand side shows the heatmap of the attention vector over the input sequence and the title shows the predicted output sequence and the Ground Truth sequence

# Conclusion
It is evident from the results thet they are not good. The resons could have been many and I am still working on fixing them. Some issues are obvious and some need a lot of experimentation.

# Probable Solution to Issues

1. Try different CNN architectures by fusing the sensor data at different levels and changing the sizes of the layers and filters.

2. Pre-train the CNN with existing datasets available for gesture classification and then use it in the above model.

3. Replace CNN with MFCC features and directly apply the Seq2Seq model with attention.

4. Preprocess the data before encoding.

5. Replace CNN with BLSTM as an encoder.

6. Visualize CNN features using t-SNE and check if they actually make sense or not.

# Appendix

# 1. Data Generation
Artificial datasets were generated by,

1. Generating random output sequences using the given labels. Eg, “100289”
2. For every label (character) in the generated output sequence, a random data instance (input sequence for a character) was picked from the original dataset corresponding to the same label.
3. These randomly picked data instances were concatenated to form an input sequence.

The artificial datasets consisted of such input and output sequences.

***No preprocessing was done on the generated data and output sequences of minimum length 1 and maximum length 10 were generated.***
    
# References

[Tensorflow](https://www.tensorflow.org/)

[Keras](https://keras.io/)

[Distance](https://pypi.python.org/pypi/Distance/)

[Attention-OCR](https://github.com/da03/Attention-OCR)

[Recursive Recurrent Nets with Attention Modeling for OCR in the Wild](https://arxiv.org/abs/1603.03101)

[Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
