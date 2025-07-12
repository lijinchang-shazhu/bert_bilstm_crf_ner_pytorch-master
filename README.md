# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

## Overview
This repository contains resources related to BERT (Bidirectional Encoder Representations from Transformers), a language representation model designed to pre-train deep bidirectional representations from unlabeled text. Unlike unidirectional language models, BERT jointly conditions on both left and right context in all layers, enabling it to achieve state-of-the-art results on various natural language processing tasks with minimal task-specific architecture modifications.

## Main Task
BERT aims to address limitations of unidirectional language models by pre-training deep bidirectional Transformers using two key tasks:
- **Masked Language Model (MLM)**: Randomly masking some tokens in the input and predicting the original tokens based on context, allowing the model to learn bidirectional features.
- **Next Sentence Prediction (NSP)**: Predicting whether one sentence follows another, helping the model understand sentence-level relationships.

These pre-trained representations can be fine-tuned for downstream tasks like question answering, language inference, and sentiment analysis.

## Key Technologies
- **Model Architecture**: Multi-layer bidirectional Transformer encoder, with two main variants: `BERT BASE` (12 layers, 768 hidden size, 12 attention heads) and `BERT LARGE` (24 layers, 1024 hidden size, 16 attention heads).
- **Pre-training Data**: BooksCorpus (800M words) and English Wikipedia (2.5B words).
- **Tokenization**: WordPiece embeddings with a 30,000 token vocabulary.
- **Fine-tuning**: Adaptable to various tasks by adding task-specific output layers, with hyperparameters (batch size, learning rate, epochs) tuned for each task.

## requirement
torch==1.8.0
pytorch_crf==0.7.2
numpy==1.17.0
transformers==4.9.0
tqdm==4.62.0
PyYAML==5.4.1
tensorboardX==2.4
