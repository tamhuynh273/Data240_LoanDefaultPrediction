from read_data import DatasetLoader
from preprocessing import Preprocessor
from preparation import Preparation
from modeling import Modeling
from catboost import CatBoostClassifier
import os

def main():
    filename = "vehicle_loan.csv"
    path = os.path.join(os.getcwd(), filename)
    load_data = DatasetLoader(path)
    df = load_data.get_data()
    print("******* Start data preprocessing *******")
    data_preprocess = Preprocessor(df)
    X, y = data_preprocess.preprocess_data(df)
    print("******* Start data engineering *******")
    prep = Preparation(X, y)
    X_train_resampled_preprocessed, y_train_resampled, X_test_preprocessed = prep.apply_pca('smote')
    ## run below code to apply rfe on smote (change parameter for adasyn)
    # X_train_resampled_preprocessed, y_train_resampled, X_test_preprocessed = prep.apply_rfe('smote')
    y_test = prep.y_test
    print("******* Start modeling *******")
    modeling = Modeling()
    trained_models = modeling.train_models(X_train_resampled_preprocessed, y_train_resampled)
    modeling.evaluate_model(trained_models, X_test_preprocessed, y_test)
    param_grid = {
        'learning_rate': [0.01, 0.1, 0.2],
        'depth': [3, 5, 7],
        'iterations': [450, 550, 650, 750, 850],
    }
    print("\n\n********************************************")
    print("******* Start hyper-parameter tuning *******")
    modeling.tune_hyperparameters(CatBoostClassifier(random_state=42, silent=True), param_grid, X_train_resampled_preprocessed, y_train_resampled)
    modeling.evaluate_best_model_with_threshold_tuning(X_test_preprocessed, y_test)

if __name__ == '__main__':
    main()