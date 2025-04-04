---
{}
---
language: en
license: cc-by-4.0
tags:
- text-classification
repo: https://github.com/eugenelian/comp34812_group70

---

# Model Card for p40327sm-n46491el-NLI

<!-- Provide a quick summary of what the model is/does. -->

This is a classification model that was trained to
      detect whether a hypothesis is true based on the premise.


## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This model is based upon a DeBERTa model that was fine-tuned
      on over 24K pairs of texts.
      DeBERTa improves upon BERT and RoBERTa by using disentagled attention and an enhanced mask encoder. In DeBERTa v3, a further improvement is gained by using ELECTRA-Style pre-training with Gradient Disentangled Embedding Sharing.
      GlobalAveragePooling is added after the last hidden layer in RoBERTa, preceding Dropout and a fully connected layer for the classification head.

- **Developed by:** Steven Moussa and Eugene Lian
- **Language(s):** English
- **Model type:** Supervised
- **Model architecture:** Transformers
- **Finetuned from model [optional]:** microsoft/deberta-v3-base

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** https://huggingface.co/microsoft/deberta-v3-base
- **Paper or documentation:** https://arxiv.org/abs/2111.09543

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

24K pairs of texts drawn from emails, news articles and blog posts.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->


      - learning_rate: 3e-05
      - train_batch_size: 32
      - eval_batch_size: 32
      - weight_decay: 0.01
      - num_warmup_steps: 100
      - dropout: 0.2
      - seed: 42
      - num_epochs: 5

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


      - overall training time: 1.25 hours
      - duration per training epoch: 15 minutes
      - model size: 701 MB

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
      - MCC (Matthew's Correlation Coefficient)
      - AUC (Area under the ROC curve)

### Results


      - Precision: 91.79%
      - Recall: 93.53%
      - F1-score: 92.65%
      - Accuracy: 92.34%
      - MCC: 84.67%
      - AUC: 92.30%

## Technical Specifications

### Hardware


      - RAM: at least 8 GB
      - Storage: at least 2GB,
      - GPU: T4

### Software


      - Transformers 4.18.0
      - tensorflow 2.18.0

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

Any inputs (concatenation of two sequences) longer than
      100 tokens will be truncated by the model.

## Additional Information

<!-- Any other information that would be useful for other people to know. -->

The hyperparameters were determined by a grid search with the below values.
      - learning_rate: [0.0,0.1,0.15]
      - train_batch_size: [8,16,32]
      - num_warmup_steps: [50,100,500,1000]
      - dropout: [0.0,0.1,0.15,0.2]
