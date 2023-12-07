from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_curve, auc, precision_recall_curve, roc_auc_score, average_precision_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier, AdaBoostClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import numpy as np
import matplotlib.pyplot as plt

class Modeling:
    def __init__(self, random_state=42):
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=random_state),
            'Decision Tree': DecisionTreeClassifier(random_state=random_state),
            'KNN': KNeighborsClassifier(),
            'Gaussian Naive Bayes': GaussianNB(),
            'Random Forest': RandomForestClassifier(random_state=random_state),
            'Light GBM': LGBMClassifier(random_state=random_state),
            'XGBoost': XGBClassifier(random_state=random_state),
            'CatBoost': CatBoostClassifier(random_state=random_state, silent=True),
            'Ensemble Voting Classifier': VotingClassifier(
                estimators=[
                    ('logistic_regression', LogisticRegression(random_state=random_state)),
                    ('decision_tree', DecisionTreeClassifier(random_state=random_state)),
                    ('knn', KNeighborsClassifier()),
                    ('gaussian_nb', GaussianNB()),
                    ('random_forest', RandomForestClassifier(random_state=random_state)),
                    ('lgbm', LGBMClassifier(random_state=random_state)),
                    ('xgboost', XGBClassifier(random_state=random_state)),
                    ('catboost', CatBoostClassifier(random_state=random_state, silent=True))
                ],
                voting='soft'
            ), 
            'Bagging': BaggingClassifier(random_state=random_state),
            'AdaBoost': AdaBoostClassifier(random_state=random_state)
        }
        self.best_model = None

    def train_models(self, X_train, y_train):
        trained_models = {}
        for name, model in self.models.items():
            print(f"\nStart training {name}")
            model.fit(X_train, y_train)
            trained_models[name] = model
            print(f"Done training")
        return trained_models


    def evaluate_model(self, trained_models, X_test, y_test):
        for name, model in trained_models.items():
            y_test_pred = model.predict(X_test)

            test_accuracy = accuracy_score(y_test, y_test_pred)
            auc_roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            auc_pr = average_precision_score(y_test, model.predict_proba(X_test)[:, 1])
            precision_ = precision_score(y_test, y_test_pred)
            recall_ = recall_score(y_test, y_test_pred)
            f1_ = f1_score(y_test, y_test_pred)

            print(f"\n\n*********** {name} - Model Evaluation Result ***********\n")
            print('Accuracy on Test set:', test_accuracy)
            print('Precision on Test Set: ', precision_)
            print('Recall on Test Set: ', recall_)
            print('F1 Score on Test Set: ',f1_)
            print('AUC-ROC on Test set: ', auc_roc)
            print('AUC-PR on Test set: ', auc_pr)

    def tune_hyperparameters(self, model, param_grid, X_train, y_train):
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        self.best_model = grid_search.best_estimator_

    def adjust_threshold(self, y_pred_proba, threshold=0.5):
        return (y_pred_proba > threshold).astype(int)

    def evaluate_best_model_with_threshold_tuning(self, X_test, y_test):
        y_test_pred_proba = self.best_model.predict_proba(X_test)[:,1]
        #threshold tuning
        _, _, thresholds = precision_recall_curve(y_test, y_test_pred_proba)
        f1_scores = [f1_score(y_test, y_test_pred_proba >= th) for th in thresholds]
        best_threshold = thresholds[np.argmax(f1_scores)]
        y_test_adjusted = self.adjust_threshold(y_test_pred_proba, best_threshold)

        accuracy = accuracy_score(y_test, y_test_adjusted)
        auc_roc = roc_auc_score(y_test, y_test_pred_proba)
        auc_pr = average_precision_score(y_test, y_test_pred_proba)
        precision_ = precision_score(y_test, y_test_adjusted)
        recall_ = recall_score(y_test, y_test_adjusted)
        f1_ = f1_score(y_test, y_test_adjusted)

        print(f"\n\n*********** Hyper-parameter Evaluation Result ***********\n")
        print('Accuracy on Test set: ', accuracy)
        print('Precision on Test Set: ', precision_)
        print('Recall on Test Set: ', recall_)
        print('F1 Score on Test Set: ',f1_)
        print('AUC-ROC on Test set: ', auc_roc)
        print('AUC-PR on Test set: ', auc_pr)

