# Loan Default Prediction 

<br>
We are performing Vehicle Loan Prediction task utilizing Data Mining Techniques
### DATASET
The dataset can be found `[here]`(https://www.kaggle.com/datasets/avikpaul4u/vehicle-loan-default-prediction/?select=train.csv).

### OVERVIEW

This project classify vehicle loan to see whether or not a person will default. 
<br><br>First, in the `read_data.py`, we read and load `vehicle_loan.csv` as the input file.
<br><br>Then we preprocessed the data using `preprocessing.py`, to handle misisng values, perform binning, converting data types, dropping outliers values, dropping uneccesary features, one hot encoding, and standardization.
<br><br>Then we have prepared the preprocessed data using `preparation.py` by performing train/test split. After that we have tried applying SMOTE and ADASYN for target balancing; PCA and RFE for reducing and selecting features for the purpose of evaluating the best combination of preprocessing procedures.
<br><br>Finally, in the `modeling.py`, we have tried 11 models, evaluate, and perform hyperparameter-tuning on the best performing model (CatBoost), and threshold tuning using F-1 Scores.
<br><br>Running `main.py` will run the project as a whole. 

### RUN THE PROJECT
To run the project:
- run `pip install -r requirements.txt`
- run `python main.py`
- As a result, you will see evaluation metrics on each of the models.


### Contribution
- Nilisha: Exploratory Data Analysis, Data Cleaning and Preprocessing, Models - KNN, Decision Tree

- Bhavana: Standardization, SMOTE, Feature selection, Modeling (Rest),  Hyper-parameter tuning

- Tam: Adaptive Synthetic, Recursive Feature Elimination, Hyper-parameter tuning, Threshold tuning



