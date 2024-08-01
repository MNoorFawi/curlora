# CURLoRA: Stable LLM Continual Fine-Tuning and Catastrophic Forgetting Mitigation
Code DOI:[![DOI](https://zenodo.org/badge/827041462.svg)](https://zenodo.org/doi/10.5281/zenodo.12729738)

Research Preprint DOI:[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12790229.svg)](https://doi.org/10.5281/zenodo.12790229)

## Overview
This repo contains the code for the CURLoRA research paper, a novel approach to fine-tuning large language models (LLMs) that leverages CUR matrix decomposition in the context of Low-Rank Adaptation (LoRA). Our method addresses two critical challenges in LLM fine-tuning: mitigating catastrophic forgetting during continual learning and reducing the number of trainable parameters. We propose a unique modification to the CUR decomposition process to enable a more efficient and stable way to adapt LLMs to new tasks without compromising any existing knowledge.  We demonstrate through experiments on multiple datasets that CURLoRA outperforms standard LoRA in mitigating catastrophic forgetting. It maintains model stability and performance across tasks while significantly reducing the number of trainable parameters. Our results show that CURLoRA achieves superior accuracy and perplexity scores compared to LoRA, particularly in scenarios with limited data.

## Contents
- `CURLoRA.pdf`: The research paper detailing the CURLoRA approach.
- `code/`: Directory containing the implementation of CURLoRA.

## Quick Start
First we install the requirements
```bash
pip3 install -r code/requirements.txt
```

All CURLoRA helper functions and classes are defined in *code/curlora.py* and *code/utils.py*. So we only need to import the modules, load the model normally and apply CURLoRA on the layers we like.

Load the model
```python
from utils import *

model_name = "mistralai/Mistral-7B-v0.3"
model, tokenizer, lm_head = load_model_and_tokenizer(model_name, type = "curlora")
```
Now you have the model with the CURLoRA layers applied to Attention layers (Key, Value and Query) which you can use for either fine-tuning or inference normally.

Please Note:
1. Some variables and values are hardcoded either in code/utils.py or code/curlora.py like the layers to apply to, rank, alpha, device etc.
2. Ongoing work (contributions are welcome) on supporting quantization (QCURLoRA) i.e. so far you load the whole model not quantized.
3. In code/ directory there are notebooks to run the research paper experiments

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
