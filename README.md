"What I cannot create, I do not understand" -- Richard P. Feynman

Here, I aim to build a Generative Pretrained Transformer (GPT) from scratch using PyTorch. While traditional GPT models are trained on a good chunk of the internet, I will be training this Transformer model on a dataset named Tiny Shakespeare. Tiny Shakespeare comprises 1MB of text data containing all the works of Shakespeare. The goal is to train the base model to generate text in the distinctive style of Shakespeare.

The workflow consists of three distinct steps:

1. Tokenization
2. Training the model and generating text
3. Quantizing the model
## Tokenization:
The tokenizer is trained on the same dataset (Tiny Shakespeare) using a basic Byte Pair Encoding (BPE) algorithm. Unlike GPT tokenizers, no regex pattern is used to break the training data into a list of strings. **The vocabulary size is set to 500**, resulting in 244 merges (500 - 256 = 244). During training, the tokenizer generates two files: **merges.pkl** and **vocab.pkl**. Once trained, it can encode any string using merges.pkl and decode the encoded string using vocab.pkl.
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
Along with the encoded string , it will also show the compression ratio after converting it from characters to tokens.
The encoded tokens can be decoded back to a string very easily:
```python
tokenize.decode([72, 105, 261, 77, 265, 110, 97, 378, 279, 84, 262, 102, 105, 107])
```
which will output the following:
```python
'Hi, My name is Toufik'
```  

## Training the model and generating text:
The data (Tiny Shakespeare) is encoded using the previously trained tokenizer. The text is compressed by a factor of 1.94x during the conversion from characters to tokens. These tokens are then fed into a decoder-only Transformer (encoder is not needed) having 3.5 Million parameters.This model is 10⁻⁵ times smaller than SOTA Large Language Models. Here is the hyperparameters of the model training:  
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
For training, the entire Python script was executed in a Colab Notebook to leverage the free T4 GPU (Things GPU poor people do to train their models :P). Colab Notebook is also available as train.ipynb.  
After setting up the Transformer architecture, the model can be trained via a simple training loop like that:  
```python
loss_t = []
loss_v = []
for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        loss_t.append(losses["train"])
        loss_v.append(losses["val"])
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

```  
With the above mentioned hyperparameters it takes around 30 minutes to train the model.  

Training and validation losses are calculated every 500 epochs and averaged over the last 200 epochs to reduce the impact of outliers. Here is the visualization of the loss:  
![](https://github.com/itoufik/Building-a-Custom-GPT-Model-from-Scratch-Using-PyTorch/blob/main/Loss_Plot.png)  

In the Feed-Forward Layer, to introduce non linearity, ReLU is used. For the purpose of diagnostic and debugging the outputs the ReLU layers, percentage of dead ReLUs, and the gradients of the ReLU layers are also visualized.  

After training the model, when a token is fed to the model, it will be able to predict the **next token**. So, given a tensor of shape (B , T) it will convert it into a tensor of (B , T + 1). And it will continually do so for **max_new_tokens**. This can be achieved via following lines of code:  
```python
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(tokenize.decode(model.generate(context, max_new_tokens=250)[0].tolist()))
```  
After training the model, the model is able to blabber Shakespeare:  

![](https://github.com/itoufik/Building-a-Custom-GPT-Model-from-Scratch-Using-PyTorch/blob/main/output%20files/unquantised_model_output.png)  

## Quantization:  
After training the model, linear channel-wise quantisation is implemented on the model (on the embedding dimension). All the parameters and buffers of the model's linear layers were replaced with a custom linear layer which has parameters and buffers of **INT8** dtype. This is a screen shot of the model where linear models are being replaced with custom made layer named Int8LinearLayer:  
![](https://github.com/itoufik/Building-a-Custom-GPT-Model-from-Scratch-Using-PyTorch/blob/main/model%20stats/print_model_quantised.png)  

Quantizing the model has reduced the memory footprint by 1.78x while increasing the inference time slightly.  

![](https://github.com/itoufik/Building-a-Custom-GPT-Model-from-Scratch-Using-PyTorch/blob/main/model%20stats/quantised_stat.png)  

After quantising the model, it is also able to blabber Shakespeare as well  
![](https://github.com/itoufik/Building-a-Custom-GPT-Model-from-Scratch-Using-PyTorch/blob/main/output%20files/quantised_model_output.png)

