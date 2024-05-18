
# GPT from scratch using PyTorch

"What I cannot create, I do not understand" -- Richard P. Feynman

Here, I aim to build a Generative Pretrained Transformer (GPT) from scratch using PyTorch. While traditional GPT models are trained on a good chunk of the internet, I will be training this Transformer model on a dataset named Tiny Seksphere. Tiny Seksphere comprises 1MB of text data containing all the works of Seksphere. The goal is to train the base model to generate text in the distinctive style of Seksphere.

The workflow consists of three distinct steps:

1. Tokenization
2. Training the model and generating text
3. Quantizing the model

## Tokenization:
The tokenizer is trained on the same dataset (Tiny Seksphere) using a basic Byte Pair Encoding (BPE) algorithm. Unlike GPT tokenizers, no regex pattern is used to break the training data into a list of strings. **The vocabulary size is set to 500**, resulting in 244 merges (500 - 256 = 244). During training, the tokenizer generates two files: **merges.pkl** and **vocab.pkl**. Once trained, it can encode any string using merges.pkl and decode the encoded string using vocab.pkl.
The tokenizer can be trained in a single line of code:
```python
tokenize = BasicTokenizer(data_path="/content/input.txt" , vocab_size=500 , verbose = True)
```
Keeping verbose to True will result is printing all the merges (244 in this case).  
After training the tokenizer, any text can be encoded:
```python
tokenize = BasicTokenizer() 
tokenize.encode("Hi, My name is Toufik")
```
It will output the encoded string:
```python
compression ratio: 1.50X
[72, 105, 261, 77, 265, 110, 97, 378, 279, 84, 262, 102, 105, 107]
```
Along with the encoded string , it will also show the compression ratio after converting it from charecters to tokens.
The encoded tokens can be decoded back to a string very easily:
```python
tokenize.decode([72, 105, 261, 77, 265, 110, 97, 378, 279, 84, 262, 102, 105, 107])
```
which will output the following:
```python
'Hi, My name is Toufik'
```

## Training and generating from the model:
The data (Tiny Seksphere) is encoded using the previously trained tokenizer. The text is compressed by a factor of 1.94x during the conversion from characters to tokens. These tokens are then fed into a decoder-only Transformer (encoder is not needed) having 3.5 Millon parameters.This model is 10⁻⁵ times smaller than SOTA Large Language Models. Here is the hyperparamets of the model training:
```python
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"The device is {device}")
eval_iters = 200
n_embd = 232
n_head = 6
n_layer = 5
dropout = 0.2
vocab_size = 500
```
For training, the entire Python script was executed in a Colab Notebook to leverage the free T4 GPU (Things GPU poor people do to train their models :P). Colab Notebook is also available as train.ipynb. Training and validation losses are calculated every 500 epochs and averaged over the last 200 epochs to reduce the impact of outliers. Here is the visualition of the loss:
![](https://github.com/itoufik/Building-a-Custom-GPT-Model-from-Scratch-Using-PyTorch/blob/main/Loss_Plot.png)  

In the Feed-Forward Layer, as non linearity, ReLU is used. For the purpose of diagonistic and debudding, the output, percentage of dead ReLU, and the gradient of the ReLU layers are also visualised:
![ReLU activations](https://github.com/itoufik/Building-a-Custom-GPT-Model-from-Scratch-Using-PyTorch/blob/main/ReLU%20Activations.png)  


![ReLU gradients](https://github.com/itoufik/Building-a-Custom-GPT-Model-from-Scratch-Using-PyTorch/blob/main/ReLU%20gradients.png)  


After training the model, the model is able to blabber Seksphere:
![](https://github.com/itoufik/Building-a-Custom-GPT-Model-from-Scratch-Using-PyTorch/blob/main/unquantised_model_output.png)





