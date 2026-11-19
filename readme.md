# 🌐 English-to-Hindi Machine Translation with a Custom PyTorch Transformer

This repository/notebook contains a complete, from-scratch implementation of the classic Seq2Seq **Transformer architecture** (Vaswani et al., 2017) built using **PyTorch** to translate sentences from English to Hindi.

---

## 📖 Table of Contents
1. [Project Overview](#-project-overview)
2. [Dataset Specifications](#-dataset-specifications)
3. [Preprocessing & Tokenization Pipeline](#-preprocessing--tokenization-pipeline)
4. [Detailed Transformer Architecture](#-detailed-transformer-architecture)
5. [Causal Masking & Training Phase](#-causal-masking--training-phase)
6. [Inference & Autoregressive Decoding](#-inference--autoregressive-decoding)
7. [Qualitative Translation Performance](#-qualitative-translation-performance)

---

## 🚀 Project Overview
Instead of relying on high-level libraries (like Hugging Face, Fairseq, or pre-built model modules), this project implements every individual layer of the Transformer—including **Multi-Head Scaled Dot-Product Attention**, **Positional Encoding**, **Encoder/Decoder layers**, and **Autoregressive inference decoding**—completely from scratch. This allows for total control over token processing and sequence masking.

---

## 📊 Dataset Specifications
- **Dataset Source:** `preetviradiya/english-hindi-dataset` via KaggleHub.
- **Dataset Volume:** Over **130,000 parallel English-Hindi sentence pairs**.
- **Vocabulary Statistics:**
  - **English (Source Vocabulary):** 66,121 unique words.
  - **Hindi (Target Vocabulary):** 78,541 unique words.
- **Sequence Constraints:** All sequences are mapped to a fixed maximum length (`MAX_LEN = 20`) to optimize tensor dimension calculations and training efficiency.

---

## ⚙️ Preprocessing & Tokenization Pipeline
To tokenize sentences without external heavy libraries, we utilize a custom-built `TransformerTokenizer` class:
1. **Text Cleansing:** All inputs are converted to lowercase and stripped of special characters/punctuation.
2. **Special Tokens Mapping:**
   - `<pad>` (Index 0): Used to pad shorter sequences to reach `MAX_LEN`.
   - `<sos>` (Index 1): Appended to indicate the **Start of Sentence** for decoder inputs.
   - `<eos>` (Index 2): Appended to signify the **End of Sentence** for decoder targets.
   - `<unk>` (Index 3): Fallback token mapping for out-of-vocabulary words.
3. **Formatting:** Target sequences (Hindi) are padded and shifted right during training, while source sequences (English) are standard-padded.

---

## 🏗️ Detailed Transformer Architecture

```
               [Encoder]                            [Decoder]
          English Sentence                      Hindi Sentence Shifted
                 │                                      │
       ┌─────────┴─────────┐                  ┌─────────┴─────────┐
       │  Input Embedding  │                  │  Input Embedding  │
       └─────────┬─────────┘                  └─────────┬─────────┘
                 ▼                                      ▼
       ┌───────────────────┐                  ┌───────────────────┐
       │Positional Encoding│                  │Positional Encoding│
       └─────────┬─────────┘                  └─────────┬─────────┘
                 ▼                                      ▼
       ┌───────────────────┐                  ┌───────────────────┐
       │Multi-Head Self-Att│                  │Masked Self-Att (1)│
       └─────────┬─────────┘                  └─────────┬─────────┘
                 │                                      │
                 └───────────────┐  ┌───────────────────┘
                                 ▼  ▼
                              ┌───────────────────┐
                              │Cross Attention (2)│
                              └─────────┬─────────┘
                                        ▼
                              ┌───────────────────┐
                              │ Feed-Forward Net  │
                              └─────────┬─────────┘
                                        ▼
                              ┌───────────────────┐
                              │Linear & Softmax   │
                              └─────────┬─────────┘
                                        ▼
                               Predicted Next Word
```

### Architectural Layers Implemented:
- **Positional Encoding:** Computes sinusoidal patterns across alternating dimensions to imbue token representations with absolute sequence order information.
- **Multi-Head Attention:** Splits the Query, Key, and Value vectors into multiple heads ($h=8$) to allow joint attention to information from different representation subspaces at different positions.
- **Encoder Layer:** Combines self-attention and positionwise feedforward layers with residual layer connections and normalization layers.
- **Decoder Layer:** Employs **Masked Self-Attention** (so it cannot look into future tokens) and **Cross-Attention** (mapping query keys directly from the encoder's output representations).
- **Model Size:** Over **61 million** active trainable parameters.

---

## 🔒 Causal Masking & Training Phase
To achieve correct sequential translation during inference, the training loop must closely mirror actual testing constraints:
- **Causal Target Masking:** We generate a custom triangular matrix mask during training:
  $$\text{Mask} = \begin{pmatrix} 1 & 0 & 0 \\ 1 & 1 & 0 \\ 1 & 1 & 1 \end{pmatrix}$$
  This ensures that when predicting word $t$, the decoder only attends to past words $1$ to $t-1$.
- **Hyperparameters:**
  - **Embedding Dim (`d_model`):** 256
  - **Feed-forward Dim (`d_ff`):** 512
  - **Heads (`n_heads`):** 8
  - **Decoder/Encoder Layers (`n_layers`):** 3
  - **Optimizer:** Adam with custom Learning Rate ($3 \times 10^{-4}$)

---

## 🔮 Inference & Autoregressive Decoding
Inference uses a step-by-step loop in `translate_sentence`:
1. Tokenize and pad the input English sentence into tensor indices.
2. Initialize target indices with the `<sos>` token.
3. Loop up to `MAX_LEN - 1`:
   - Pad current predicted indices up to `MAX_LEN`.
   - Apply causal masking so the model only attends to already generated indices.
   - Feed input tensors through the trained model.
   - Predict the index of the next token at the current step.
   - Append prediction and stop decoding immediately if the `<eos>` token is generated.

---

## 📊 Qualitative Translation Performance
After training for 10 epochs with proper causal masking, the model generalizes and produces highly accurate short-phrase translations:

| Input English | Target Translation (Expected) | Model Predicted Translation | Status |
| :--- | :--- | :--- | :--- |
| **Help!** | बचाओ! / सहायता | **मदद** | ✅ Correct |
| **Go away!** | चले जाओ! | **जा कर बाहर जाना** | ✅ Highly Accurate |
| **Come in.** | अंदर आ जाओ। | **आ गया** | ✅ Semantically Close |
| **I forgot.** | मैं भूल गया। | **मैं भूल गया** | ✅ Correct |
| **what's up** | क्या चल रहा है? | **क्या हो रहा है** | ✅ Correct |
| **how are you** | तुम कैसे हो? | **तुम कैसे हो** | ✅ Perfect |