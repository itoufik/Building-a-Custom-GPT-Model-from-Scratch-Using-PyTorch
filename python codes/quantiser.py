# PyTorch imports
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import time

# importing the tokenizer and decoder
from decoder import Decoder , get_model_size
from basictokenizer import BasicTokenizer

# Initialising them
tokenize = BasicTokenizer()
model = Decoder()

# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Loading the model
state_dict = torch.load("model_weight_uq.pth")
model.load_state_dict(state_dict, strict=True, assign=True)


class Int8LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, dtype=torch.float32):
        # Dummy attributes, to be changed later
        super().__init__()
        self.register_buffer("int8_weights",torch.randint(-128, 127, (out_features, in_features), dtype=torch.int8))
        self.register_buffer("scales", torch.randn((out_features), dtype=dtype))
        if bias:
            self.register_buffer("bias", torch.randn((1, out_features), dtype=dtype))
        else:
            self.bias = None
    
    @staticmethod
    def w8_forward(weight, input, scales, bias=None):
        # forward pass through the quantised layer
        casted_weights = weight.to(input.dtype) # cast the weights to input dtype // (B , T , hs)
        output = F.linear(input, casted_weights) * scales # forward , weights already in original dtype (B , T , hs)
        if bias is not None:
            output = output + bias
        return output
    
    def quantize(self, weights):
        # Quantise the layers weight in int 8 , and keep the scales 
        w_fp32 = weights.clone().to(torch.float32) # copy of the weights // (B , T , hs)
        scales = w_fp32.abs().max(dim=-1).values / 127 # scale for quantisation along hs dim in (B, T, hs) tensor // dimention of quantisation // (B , T)
        scales = scales.to(weights.dtype) # scales in original data type // (B , T)
        int8_weights = torch.round(weights/scales.unsqueeze(-1)).to(torch.int8) # creating a fake dimention to broadcast // (B , T , hs)/ (B , T , 1) --> (B , T , hs)
        self.int8_weights = int8_weights # int8
        self.scales = scales # original dtype

    def forward(self, input):
        return self.w8_forward(self.int8_weights, input, self.scales, self.bias)
    
def replace_linear_layer(module, target_class, module_name_to_exclude):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and not \
        any([x == name for x in module_name_to_exclude]):
            old_bias = child.bias
            old_weight = child.weight
            new_module = target_class(child.in_features, child.out_features, old_bias is not None, child.weight.dtype)
            setattr(module, name, new_module)
            getattr(module, name).quantize(old_weight)   
            if old_bias is not None:
              getattr(module, name).bias = old_bias
        else:
            # Recursively call the function for nested modules
            replace_linear_layer(child, target_class, module_name_to_exclude)


# print(model)
model_size = get_model_size(model)
print(f"Model size before quantisation: {model_size / 1e6:.2f} MB")
replace_linear_layer(model, Int8LinearLayer, ["lm_head"]) # not quantising the final layer
model.to(device) # move the quantised the model to cuda , if available
# print(model)
model_size_q = get_model_size(model)
print(f"Model size after quantisation: {model_size_q / 1e6:.2f} MB")
print(f"Model has shrinked by {(model_size/model_size_q):.2f}X")


# generate from the quantised model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
start_q = time.time()
open('output_quantised.txt', 'w').write(tokenize.decode(model.generate(context, max_new_tokens=10000)[0].tolist()))
end_q = time.time()
print(f"Time taken to generate 10,000 tokens with the quantised model is {(end_q-start_q)/60}")
time_q = (end_q-start_q)/60