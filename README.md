# Infect Predict

What a roller coster ride, thanks to the organizers.

Starting with the basic analysis through EDA procedures. We identified the relation between the dataset provided and brainstormed on the possible solutions wherein we developed and trained a Random forest algorithm to check the accuracy and understand the performance of the algorithm with the data. Soon we realised it was a no-no. Wherein we started to think about the possibilites of ensemble learning.

Our task 1 solution
In total, 3 Stack models

Stack after each single prediction so that following predictions are made on more accurate data. We stack with a weighted average based on val MSE and
Robust cross validation. For cross validation, we utilized 3 folds for each CSV. We scaled the CSVs individually.
Using meta labels, such as cumulative case counts and the total vaccinated we got much better MAE with faster convergence, 

Meta features included the infected_unvaccinated infected_vaccinated total_vaccinated days_increasing cumulative_cases.

### Models generated:

- Random Forest algorithm.
- LSTM with meta - MSE - 3650
- GRU with meta - MSE - 3575
- CNN with meta - MSE - 3599

### Usage
`pip install -r requirements.txt`

Once completed, type
```from infect_predict.pipeline.train import model_prediction
model_prediction('./input/observations_1.csv', 100) 
```

### Code explanation
infect_predict is a library with 
- `pipeline` containing the major pipeline components of training the models, and inferencing the ensemble.
- `inference.py` does the inference for the ensemble models supporting models with metadata.
- `train.py` is the important file for our task 2 submission running the LSTM and a GRU models on the feature engineered dataset. 
- `tf_models.py` contains the model architecture code with LSTM, GRU and CNN.
-  preprocessing files are in infect_predict/utils/preprocess.py.

We went beyond the tasks mentioned and created [Case predictor](https://github.com/RutvikJ77/Case-predictor)
