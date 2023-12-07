from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from imblearn.over_sampling import ADASYN, SMOTE
from sklearn.decomposition import PCA 
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

class Preparation():
    def __init__(self, X, y, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        self.smote = SMOTE(sampling_strategy='auto', random_state=random_state)
        self.adasyn = ADASYN(sampling_strategy=1.0, random_state=random_state)
        self.pca = PCA()
        self.rfe = None
    
    def apply_smote(self):    
        X_train_smote, y_train_smote = self.smote.fit_resample(self.X_train, self.y_train)
        return X_train_smote, y_train_smote

    def apply_adasyn(self):
        X_train_adasyn, y_train_adasyn = self.adasyn.fit_resample(self.X_train, self.y_train)
        return X_train_adasyn, y_train_adasyn

    def apply_pca(self, imbalance='smote'):
        if imbalance=='smote':
            X_train_resampled, y_train_resampled = self.apply_smote()
        elif imbalance=='adasyn':
            X_train_resampled, _ = self.apply_adasyn()
        self.pca.fit(X_train_resampled)
        cumulative_proportion = np.cumsum(self.pca.explained_variance_ratio_)
        n_features_95 = np.argmax(cumulative_proportion > 0.95) + 1
        pca = PCA(n_components=n_features_95)
        X_train_pca_temp = pca.fit_transform(X_train_resampled)
        X_train_resampled_pca = pd.DataFrame(X_train_pca_temp, columns=[f'PC{i+1}' for i in range(X_train_pca_temp.shape[1])])
        X_test_pca_temp = pca.fit_transform(self.X_test)
        X_test_pca = pd.DataFrame(X_test_pca_temp, columns=[f'PC{i+1}' for i in range(X_test_pca_temp.shape[1])])

        return X_train_resampled_pca, y_train_resampled, X_test_pca

    def apply_rfe(self, imbalance='smote'):
        model = RandomForestClassifier()
        n_features_to_select_range = range(1, len(self.X_train)+1)
        scores = []
        if imbalance=='smote':
            X_train_resampled, y_train_resampled = self.apply_smote()
        elif imbalance =='adasyn':
            X_train_resampled, y_train_resampled = self.apply_adasyn()

        for n in n_features_to_select_range:
            rfe = RFE(estimator=model, n_features_to_select=n).fit(X_train_resampled, y_train_resampled)
            cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)
            f1_score = cross_val_score(model, X_train_resampled, y_train_resampled, scoring='f1', cv=cv, n_jobs=-1)
            mean_f1_score = np.mean(f1_score)
            scores.append(mean_f1_score)

        n_features_to_select = scores.index(max(scores))
        final_rfe = RFE(model, n_features_to_select=n_features_to_select).fit(X_train_resampled, y_train_resampled)
        selected_features = final_rfe.get_support(1)
        X_train_resampled_rfe = X_train_resampled[X_train_resampled.columns[selected_features]]
        X_test_rfe = self.X_test[self.X_test.columns[selected_features]]
        return X_train_resampled_rfe, y_train_resampled, X_test_rfe

