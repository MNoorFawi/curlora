```text
\title{CURLoRA: Leveraging CUR Matrix Decomposition for Stable LLM Continual Fine-tuning and Catastrophic Forgetting Mitigation}
\author{Muhammad Fawi}
\date{}

\begin{document}

\maketitle

\begin{abstract}
This paper introduces CURLoRA, a novel approach to fine-tuning large language models (LLMs) that leverages CUR matrix decomposition in the context of Low-Rank Adaptation (LoRA). Our method addresses two critical challenges in LLM fine-tuning: mitigating catastrophic forgetting during continual learning and reducing the number of trainable parameters. We propose a unique modification to the CUR decomposition process, utilizing inverted probabilities for column and row selection, which acts as an implicit regularization. Through experiments on multiple datasets, we demonstrate that CURLoRA outperforms standard LoRA in maintaining model stability and performance across tasks while significantly reducing the number of trainable parameters. Our results show that CURLoRA achieves superior accuracy and perplexity scores compared to LoRA, particularly in scenarios with limited fine-tuning data.
\end{abstract}
\end{document}
```
