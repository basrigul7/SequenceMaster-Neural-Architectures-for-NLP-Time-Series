# SequenceMaster: Neural Architectures for NLP & Time-Series

This project focuses on the implementation and comparative analysis of advanced Neural Network architectures for processing sequential data. It covers the evolution of sequence modeling, starting from **LSTMs** to modern **Transformer** and **Attention** mechanisms, applied to Natural Language Processing (NLP) tasks.

## 🎯 Overview
The core objective is to understand how different neural architectures handle long-range dependencies and sequential patterns. The repository explores the limitations of traditional RNNs and demonstrates the efficiency of newer models in real-world scenarios such as language modeling and low-latency streaming.

## 🚀 Key Features
* **LSTM Implementation:** Deep dive into Long Short-Term Memory units to mitigate the vanishing gradient problem.
* **Encoder-Decoder Framework:** Implementation of sequence-to-sequence models for complex NLP mappings.
* **Transformer vs. RNN Analysis:** A technical comparison highlighting the advantages of parallelization and self-attention over recurrence.
* **Edge Deployment Analysis:** Practical discussion on utilizing LSTMs for low-latency, memory-constrained environments (e.g., wake-word detection).



## 📊 Technical Insights
The project provides a detailed evaluation of architectural trade-offs:
* **Memory & Efficiency:** While Transformers dominate in performance, LSTMs remain superior for edge devices due to their lower RAM footprint and ability to process infinite data streams without massive attention matrices.
* **Long-Range Dependencies:** Analysis of how the Attention mechanism solves the "fixed-vector bottleneck" found in traditional Encoder-Decoder models.
* **Hardware Constraints:** Discussion on the practical application of these models in mobile and low-power hardware.



## 🛠️ Installation & Usage

### Prerequisites
You will need Python 3.x and the following libraries:
```bash
pip install torch torchtext numpy matplotlib seaborn
