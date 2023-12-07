import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class Preprocessor:
    def __init__(self, df):
        self.df = df

    def converting_data(self, df):
        df['AVERAGE_ACCT_AGE_YRS'] = df.apply(lambda row: row['AVERAGE_ACCT_AGE'].split("yrs")[0], axis=1)
        df['AVERAGE_ACCT_AGE_MONTHS'] = df.apply(lambda row: row['AVERAGE_ACCT_AGE'].split("yrs")[1].split("mon")[0], axis=1)
        df['CREDIT_HISTORY_LENGTH_YRS'] = df.apply(lambda row: row['CREDIT_HISTORY_LENGTH'].split("yrs")[0], axis=1)
        df['CREDIT_HISTORY_LENGTH_MONTHS'] = df.apply(lambda row: row['CREDIT_HISTORY_LENGTH'].split("yrs")[1].split("mon")[0], axis=1)
        df['AVERAGE_ACCT_AGE'] = df['AVERAGE_ACCT_AGE_YRS'].astype(int) + (df['AVERAGE_ACCT_AGE_MONTHS'].astype(int)/12)
        df['CREDIT_HISTORY_LENGTH'] = df['CREDIT_HISTORY_LENGTH_YRS'].astype(int) + (df['CREDIT_HISTORY_LENGTH_MONTHS'].astype(int)/12)
        df['DATE_OF_BIRTH'] =  pd.to_datetime(df['DATE_OF_BIRTH'], format='%d-%m-%Y')
        df['DISBURSAL_DATE'] =  pd.to_datetime(df['DISBURSAL_DATE'], format='%d-%m-%Y')
        df['ANYID_FLAG']  = df.apply(lambda row: True if row['AADHAR_FLAG'] + row['PAN_FLAG'] + row['VOTERID_FLAG'] + row['DRIVING_FLAG'] + row['PASSPORT_FLAG'] > 0 else False, axis=1)
        df['Age'] = df['DISBURSAL_DATE'].dt.year - df['DATE_OF_BIRTH'].dt.year
        df['disbursal_month'] = df['DISBURSAL_DATE'].dt.month
        df['disbursal_day'] = df['DISBURSAL_DATE'].dt.day

        conditions = [
            (df['PERFORM_CNS_SCORE_DESCRIPTION'].str.contains("No Bureau|Not Scored")),
            (df['PERFORM_CNS_SCORE_DESCRIPTION'].str.contains("-Very Low Risk")),
            (df['PERFORM_CNS_SCORE_DESCRIPTION'].str.contains("-Low Risk")),
            (df['PERFORM_CNS_SCORE_DESCRIPTION'].str.contains("-Medium Risk")),
            (df['PERFORM_CNS_SCORE_DESCRIPTION'].str.contains("-High Risk")),
            (df['PERFORM_CNS_SCORE_DESCRIPTION'].str.contains("-Very High Risk")),  
        ]
        values = [0,1,2,3,4,5]
        df['PERFORM_CNS_SCORE_bins'] = np.select(conditions, values)  
        return df
    
    def drop_uneccesary_columns(self, df):
        df['EMPLOYMENT_TYPE'] = df['EMPLOYMENT_TYPE'].fillna("Not available")
        threshold_disbursed_amount = df['DISBURSED_AMOUNT'].mean() + 3*df['DISBURSED_AMOUNT'].std()
        threshold_asset_cost = df['ASSET_COST'].mean() + 3*df['ASSET_COST'].std()
        threshold_pri_no_accts = df['PRI_NO_OF_ACCTS'].mean() + 3*df['PRI_NO_OF_ACCTS'].std()
        drop_col = ['MOBILENO_AVL_FLAG', 'AVERAGE_ACCT_AGE_YRS', 'AVERAGE_ACCT_AGE_MONTHS', 'CREDIT_HISTORY_LENGTH_YRS', \
                    'CREDIT_HISTORY_LENGTH_MONTHS', 'AADHAR_FLAG', 'PAN_FLAG', 'VOTERID_FLAG', 'DRIVING_FLAG', 'ANYID_FLAG',\
                    'PASSPORT_FLAG', 'UNIQUEID', 'PERFORM_CNS_SCORE','PERFORM_CNS_SCORE_DESCRIPTION', 'AVERAGE_ACCT_AGE',\
                    'PRI_DISBURSED_AMOUNT', 'SEC_SANCTIONED_AMOUNT', 'SEC_DISBURSED_AMOUNT', 'SEC_NO_OF_ACCTS', 'BRANCH_ID', \
                    'SUPPLIER_ID', 'MANUFACTURER_ID','CURRENT_PINCODE_ID','DATE_OF_BIRTH','STATE_ID','EMPLOYEE_CODE_ID', \
                    'DISBURSAL_DATE']
        df.drop(columns=drop_col, inplace=True)
        df.drop(df[df['DISBURSED_AMOUNT'] > threshold_disbursed_amount].index, inplace=True)
        df.drop(df[df['ASSET_COST'] > threshold_asset_cost].index, inplace=True) 
        df.drop(df[df['PRI_NO_OF_ACCTS'] > threshold_pri_no_accts].index, inplace=True)
        return df

    def standardization(self, df):
        df = self.drop_uneccesary_columns(df)
        y = df['LOAN_DEFAULT']
        X = df.drop(columns=['LOAN_DEFAULT'])
        exclude_columns = ['DATE_OF_BIRTH', 'BRANCH_ID', 'MANUFACTURER_ID', 'CURRENT_PINCODE_ID', 'SUPPLIER_ID', 'STATE_ID', 'EMPLOYEE_CODE_ID', 'PERFORM_CNS_SCORE_bins']
        numerical_col = X.select_dtypes(exclude=['object','datetime64[ns]']).columns
        keep_columns = [col for col in numerical_col if col not in exclude_columns]
        scaler = StandardScaler()
        X[keep_columns] = scaler.fit_transform(X[keep_columns])
        return X, y

    def one_hot_encoding(self, df):
        X, y = self.standardization(df)
        X = pd.get_dummies(X)
        return X, y

    def preprocess_data(self, df):
        df = self.converting_data(df)
        X, y = self.one_hot_encoding(df)
        return X, y
