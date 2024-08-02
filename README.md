# CURLoRA: Stable LLM Continual Fine-Tuning and Catastrophic Forgetting Mitigation
Code DOI:[![DOI](https://zenodo.org/badge/827041462.svg)](https://zenodo.org/doi/10.5281/zenodo.12729738)

Research Preprint DOI:[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12790229.svg)](https://doi.org/10.5281/zenodo.12790229)

## Overview
This repo contains the code for the CURLoRA research paper, a novel approach to fine-tuning large language models (LLMs) that leverages CUR matrix decomposition in the context of Low-Rank Adaptation (LoRA). Our method addresses two critical challenges in LLM fine-tuning: mitigating catastrophic forgetting during continual learning and reducing the number of trainable parameters. We propose a unique modification to the CUR decomposition process to enable a more efficient and stable way to adapt LLMs to new tasks without compromising any existing knowledge.  We demonstrate through experiments on multiple datasets that CURLoRA outperforms standard LoRA in mitigating catastrophic forgetting. It maintains model stability and performance across tasks while significantly reducing the number of trainable parameters. Our results show that CURLoRA achieves superior accuracy and perplexity scores compared to LoRA, particularly in scenarios with limited data.

## Contents
- `CURLoRA.pdf`: The research paper detailing the CURLoRA approach.
- `code/`: Directory containing the implementation of CURLoRA and the experiments.
	- `code/curlora.py`: Containing CURLoRA classes.
	- `code/utils.py`: Helper functions.
	- `code/lora.py`: LoRA classes.
	- `code/curlora_experiment.ipynb`: CURLoRA experiment with Mistral 7B (Fine-tuning on MRPC, SST-2 and Sentiment140).
	- `code/curlora_experiment-gpt.ipynb`: CURLoRA experiment with GPT2-Large (Fine-tuning on MRPC, SST-2 and Sentiment140).
	- `code/squad_gpt-curlora.ipynb`: Fine-Tuning GPT2-Large with CURLoRA on SQuAD dataset.

## Quick Start
First we install the requirements
```bash
pip3 install -r code/requirements.txt
```

All CURLoRA helper functions and classes are defined in *code/curlora.py* and *code/utils.py*.

Load the model and apply CURLoRA
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import *

model_name = "gpt2-large"
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to("cuda") # this will make all existing layers in CUDA

# turning off grad for all layers
for param in model.parameters():
        param.requires_grad = False


# replace original Q,K,V layers with CURLoRA (GPT2-Large specific)
# refer to utils.py for a more general way
for name, module in model.named_modules():
    if isinstance(module, type(model.transformer.h[0].attn)):
	    # rank = 24, alpha = 1
	    module.c_attn = LinearWithCURLoRA(module.c_attn, 24, 1)


# now look at how many CURLoRA parameters to be trained
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters after: {total_params:,}")
# making sure CURLoRA layers are on CUDA as well
model.to("cuda")
```
Now you have the model with the CURLoRA layers applied to Attention layers (Key, Value and Query) which you can use for either fine-tuning or inference normally.

You may need to know how the layer is called so that you can replace it correctly. For instance, Q, K, V in Mistral can be found via:
```python
for name, module in model.named_children():
    if any(l in name for l in ["q_proj", "v_proj", "k_proj"]):
		setattr(model, name, LinearWithCURLoRA(module, rank, alpha))
```

Please Note:
1. Some variables and values are hardcoded either in code/utils.py or code/curlora.py like the layers to apply to, rank, alpha, device etc.
2. Ongoing work (contributions are welcome) on supporting quantization (QCURLoRA) i.e. so far you load the whole model not quantized.
3. In code/ directory there are notebooks to run the research paper experiments
4. You may need to use a slightly higher learning rate than with LoRA to get better accuracy. Higher learning rate won't cause overfitting due to the "implicit regularization" explained in the paper.

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/MNoorFawi/curlora/blob/main/LICENSE) file for details.

## Citation
If you find CURLoRA research or code helpful, please consider citing them.

#### Code Citation

1. Bibtext
```bibtex
@software{fawi_2024_12738436,
  author       = {Fawi, Muhammad},
  title        = {{CURLoRA: Leveraging CUR Matrix Decomposition for 
                   Stable LLM Continual Fine-Tuning and Catastrophic
                   Forgetting Mitigation}},
  month        = jul,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {v2.0.0},
  doi          = {10.5281/zenodo.12729738},
  url          = {https://zenodo.org/doi/10.5281/zenodo.12729738}
}
```

2. APA
```text
Fawi, M. (2024). CURLoRA: Leveraging CUR Matrix Decomposition for Stable LLM Continual Fine-Tuning and Catastrophic Forgetting Mitigation (v2.0.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.12738436
```

#### Research Paper Citation

1. Bibtext
```bibtex
@misc{fawi_2024_12730055,
  author       = {Fawi, Muhammad},
  title        = {{CURLoRA: Leveraging CUR Matrix Decomposition for 
                   Stable LLM Continual Fine-Tuning and Catastrophic
                   Forgetting Mitigation}},
  month        = jul,
  year         = 2024,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.12730055},
  url          = {https://doi.org/10.5281/zenodo.12730055}
}
```

2. APA
```text
Fawi, M. (2024). CURLoRA: Leveraging CUR Matrix Decomposition for Stable LLM Continual Fine-Tuning and Catastrophic Forgetting Mitigation. Zenodo. https://doi.org/10.5281/zenodo.12730055
```

**Contribution and ideas will be much appreciated**
