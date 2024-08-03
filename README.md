# This is a quick description on how to use this project

This project contains scripts, data, and results for predicting depdenent changes.

The strcuture of the project:

1. **Changes:** contains the changes of OpenStack.
2. **doc2vec:** trained doc2vec models used to generate the embedding of text.
3. **1_Changed_files_retrieval:** retreives the changed files from OpenDev API.
4. **2_Metrics_collection:** file to collect different metrics to train our models.
5. **3_Split.py:** a script used to consutuct the dataset of the 2nd model.
6. **4_Assign.py:** a script used to assign assign and compute pair-metrics of the 2nd model.
7. **preliminary_study.ipynb:** the Jupyter notebook used to carry out the results of our preliminary study
8. **model_1.ipynb:** this file evaluate the effectiness of our first model.
9. **model_2.ipynb:** this file evaluate the effectiness of our second model.
10. **Results:** contains all results of our appoach regading the performances, ranking features, and the interpretation
of different models
