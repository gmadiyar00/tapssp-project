---
license: other
---
# Disclaimer
I do **NOT** own this model. It belongs to its developer (Microsoft). See the license file for more details.

# Overview
This repo contains the parameters of phi-2, which is a large language model developed by Microsoft.

# How to run
This model requires 12.5 GB of vRAM in float32. 

Should take roughly 6.7 GB in float16.

## 1. Setup
install the needed libraries
```bash
pip install sentencepiece transformers accelerate einops
```

## 2. Download the model
```python
from huggingface_hub import snapshot_download
model_path = snapshot_download(repo_id="amgadhasan/phi-2",repo_type="model", local_dir="./phi-2", local_dir_use_symlinks=False)
```

## 3. Load and run the model
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# We need to trust remote code since this hasn't been integrated in transformers as of version 4.35
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)

def generate(prompt: str, generation_params: dict = {"max_length":200})-> str : 
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, **generation_params)
    completion = tokenizer.batch_decode(outputs)[0]
    return completion

result = generate(prompt)
result
```


## float16
To load this model in float16, use the following code:
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# We need to trust remote code since this hasn't been integrated in transformers as of version 4.35
# We need to set the torch dtype globally since this model class doesn't accept dtype as argument
torch.set_default_dtype(torch.float16)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)

def generate(prompt: str, generation_params: dict = {"max_length":200})-> str : 
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, **generation_params)
    completion = tokenizer.batch_decode(outputs)[0]
    return completion

result = generate(prompt)
result
```

# Acknowledgments
Special thanks to Microsoft for developing and releasing this mode. Also, special thanks to the huggingface team for hosting LLMs for free!