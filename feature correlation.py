import pandas as pd
import numpy as np
from dython.nominal import associations
from dython.nominal import identify_nominal_columns
from prediction import preprocess

full_data = pd.read_csv("./Mission 2 - Breast Cancer/train.feats.csv")
full_data.rename(columns=lambda x: x.replace('אבחנה-', ''), inplace=True)
labels = pd.read_csv("./Mission 2 - Breast Cancer/train.labels.0.csv")
# classified_labels = np.where(labels == '[]', 0, 1).reshape(-1)
half_processed_data, classified_labels = preprocess(full_data, labels)
half_processed_data['labels'] = classified_labels


categorical_features=identify_nominal_columns(half_processed_data)
print(categorical_features)
# associations(full_data, nominal_columns='auto', numerical_columns=None, mark_columns=False, nom_nom_assoc='cramer', num_num_assoc='pearson', bias_correction=True, nan_strategy=_REPLACE, nan_replace_value=_DEFAULT_REPLACE_VALUE, ax=None, figsize=None, annot=True, fmt='.2f', cmap=None, sv_color='silver', cbar=True, vmax=1.0, vmin=None, plot=True, compute_only=False, clustering=False, title=None, filename=None)
complete_correlation= associations(half_processed_data, filename= 'complete_correlation.png', figsize=(30,30))
df_complete_corr=complete_correlation['corr']
df_complete_corr.dropna(axis=1, how='all').dropna(axis=0, how='all').style.background_gradient(cmap='coolwarm', axis=None).set_precision(2)