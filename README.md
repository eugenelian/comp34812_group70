# Repository Structure
[1] modelcards: Directory contains the model cards for both models
[2] solution_b_bilstm: Directory contains the test predictions and source code used for the training, evaluation and demo for solution B. For the demo, download the model weights using the link below before running source code. 
[3] solution_c_transformer: Directory contains the test predictions and source code used for the training, evaluation and demo for solution C. For the demo, download the model weights using the link below before running source code. 


# Solution B: Deep learning-based approaches that do not employ transformer architectures
## Saved Model

Download the model weights from below: 

https://smu-my.sharepoint.com/:u:/g/personal/eugene_lian_2022_scis_smu_edu_sg/EXudFtydXftAgGGvM9nMlVEBXahQn46QCknT751VnEXb-g?e=CaGwn9

## Files and instructions to run
The notebooks were ran in Google Colaboratory
- BiLSTM_DeBERTa_Train.ipynb: Source code for the model. Attach the train.csv and dev.csv at runtime. Trained model can be downloaded once completed.
- BiLSTM_DeBERTa_Eval.ipynb: Code to evaluate the model on the validation set. Attach the dev.csv file and the model weights either from training or downloaded below
- BiLSTM_DeBERTa_Test.ipynb: Code to predict new csv files. Attach the test.csv file. Predictions can be downloaded once completed.


# Solution C: Deep learning-based approaches underpinned by transformer architectures
## Model Weights

Download the model weights from below:

https://livemanchesterac-my.sharepoint.com/:u:/g/personal/steven_moussa_student_manchester_ac_uk/EUPyM4KJbp5HjRjL0Nx7mTQBN7dyBhASVVnAyWBj9lB5mQ?e=InX4qg

## Files and instructions to run
The notebooks were ran in Google Colaboratory
- DeBERTa Train.ipynb: Source code for the model. Attach the train.csv and dev.csv at runtime. Trained weights can be downloaded once completed.
- DeBERTa Eval.ipynb: Code to evaluate the model on the validation set. Attach the dev.csv file and the weights either from training or downloaded below
- DeBERTa Test.ipynb: Code to predict new csv files. Attach the test.csv file. Predictions can be downloaded once completed.


# Resources used

[1] https://www.kaggle.com/code/alessandrozanette/natural-language-inference-with-bert-model

[2] https://huggingface.co/

[3] https://paperswithcode.com/task/natural-language-inference