import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

def load_and_preprocess(path):
    df = pd.read_csv(path)

    # Clean column names (IMPORTANT FIX)
    df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)

    features = [
        'Sex',
        'Age_recode_with_1_year_olds_and_90_',
        'Race_recode_W_B_AI_API_',
        'Derived_AJCC_Stage_Group_6th_ed_2004_2015_',
        'Radiation_recode',
        'Chemotherapy_recode_yes_no_unk_'
    ]

    df = df[features + [
        'Survival_months',
        'Vital_status_recode_study_cutoff_used_'
    ]]

    df.replace(["Unknown", "unknown", "Not Available", "Blank(s)"], np.nan, inplace=True)

    for col in features:
        df[col] = df[col].fillna(df[col].mode()[0])

    for col in features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    df.rename(columns={
        'Survival_months': 'time',
        'Vital_status_recode_study_cutoff_used_': 'event'
    }, inplace=True)

    df = df.dropna()

    return df, features