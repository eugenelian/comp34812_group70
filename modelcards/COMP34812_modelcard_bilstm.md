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

This Bidirectional Long Short-Term Memory (BiLSTM) model is based upon a recurrent neural network that was fine-tuned
      on 24K pairs of texts.

- **Developed by:** Eugene Lian and Steven Moussa
- **Language(s):** English
- **Model type:** Deep learning-based approaches that do not employ transformer architectures
- **Model architecture:** BiLSTM
- **Finetuned from model [optional]:** BiLSTM

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** https://huggingface.co/google-bert/bert-base-uncased
- **Paper or documentation:** https://aclanthology.org/N19-1423.pdf

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

24K premise-hypothesis pairs of texts.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->


      - learning_rate: 0.001
      - train_batch_size: 16
      - eval_batch_size: 16
      - seed: 42
      - num_epochs: 10

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


      - overall training time: 15 minutes
      - duration per training epoch: 72 seconds
      - model size: 100MB

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->

A subset of the development set provided, amounting to 2K pairs.

#### Metrics

<!-- These are the evaluation metrics being used. -->


      - Precision
      - Recall
      - F1-score
      - Accuracy

### Results

The model obtained an F1-score of 70% and an accuracy of 70%.

## Technical Specifications

### Hardware


      - RAM: at least 16 GB
      - Storage: at least 2GB,
      - GPU: V100

### Software


      - Transformers 4.18.0
      - Pytorch 1.11.0+cu113

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

Any inputs (concatenation of two sequences) longer than
      512 subwords will be truncated by the model.

## Additional Information

<!-- Any other information that would be useful for other people to know. -->

The hyperparameters were determined by experimentation
      with different values.
