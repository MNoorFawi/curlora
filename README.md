# CURLoRA: Leveraging CUR Matrix Decomposition for Stable LLM Continual Fine-Tuning and Catastrophic Forgetting Mitigation
[![DOI](https://zenodo.org/badge/827041462.svg)](https://zenodo.org/doi/10.5281/zenodo.12729738)

## Overview
This repo contains the code for the CURLoRA research paper, a novel approach to fine-tuning large language models (LLMs) that leverages CUR matrix decomposition in the context of Low-Rank Adaptation (LoRA). Our method addresses two critical challenges in LLM fine-tuning: mitigating catastrophic forgetting during continual learning and reducing the number of trainable parameters. We propose a unique modification to the CUR decomposition process, utilizing inverted probabilities for column and row selection which acts as an implicit regularization, and initializing the U matrix as a zero matrix, and only fine-tuning it. Through experiments on multiple datasets, we demonstrate that CURLoRA outperforms standard LoRA in mitigating catastrophic forgetting maintaining model stability and performance across tasks while significantly reducing the number of trainable parameters. Our results show that CURLoRA achieves superior accuracy and perplexity scores compared to LoRA, particularly in scenarios with limited fine-tuning data.

## Contents
- `CURLoRA.pdf`: The research paper detailing the CURLoRA approach.
- `code/`: Directory containing the implementation of CURLoRA.

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/MNoorFawi/curlora/blob/main/LICENSE) file for details.
