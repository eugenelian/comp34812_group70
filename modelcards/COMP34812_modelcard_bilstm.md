---
{}
---
language: en
license: cc-by-4.0
tags:
- pairwise-sequence-classification
- natural-language-inference
repo: https://github.com/eugenelian/comp34812_bilstm

---

# Model Card for n46491el-p40327sm-NLI

<!-- Provide a quick summary of what the model is/does. -->

This is a pairwise sequence classification model that was trained to
      detect whether a piece of text is true based on the premise piece of text.


## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This model is based upon a Bidirectional Long Short-Term Memory
      (BiLSTM) model that was fine-tuned on 24K pairs of texts. A further improvement on
      the model is gained by using DeBERTa only for embeddings.

- **Developed by:** Eugene Lian and Steven Moussa
- **Language(s):** English
- **Model type:** Supervised
- **Model architecture:** Deep learning-based approaches that do not employ transformer architectures
- **Finetuned from model [optional]:** BiLSTM

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** https://huggingface.co/microsoft/deberta-v3-base
- **Paper or documentation:** https://arxiv.org/pdf/2006.03654

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

24K pairs of texts drawn from emails, news articles and blog posts.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->


      - learning_rate: 1e-05
      - train_batch_size: 16
      - eval_batch_size: 16
      - seed: 42
      - dropout: 0.1
      - num_epochs: 10

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


      - overall training time: 15 minutes
      - duration per training epoch: 72 seconds
      - model size: 14.16 MB

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->

Full development set provided, amounting to 6K pairs.

#### Metrics

<!-- These are the evaluation metrics being used. -->


      - Precision
      - Recall
      - F1-score
      - Accuracy
      - Matthew's Correlation Coefficient (MCC)
      - Receiver Operating Characteristic Area Under Curve (ROC AUC) score

### Results


      - Precision: 69.79%
      - Recall: 70.26%
      - F1-score: 70.19%
      - Accuracy: 70.26%
      - MCC: 40.43%
      - ROC AUC score: %

## Technical Specifications

### Hardware


      - RAM: at least 8 GB
      - Storage: at least 2GB,
      - GPU: T4

### Software


      - Transformers 4.50.3
      - Tensorflow 2.16.1

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

Any inputs (concatenation of two sequences) longer than
      100 tokens will be truncated by the model.

## Additional Information

<!-- Any other information that would be useful for other people to know. -->

The hyperparameters were determined by a grid search with the below values.
      - learning_rate: [0.0,0.1,0.15]
      - train_batch_size: [8,16,32]
      - dropout: [0.0,0.1,0.15]
