import statistics

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier


def evaluate_and_export(estimator, X: np.ndarray, filename: str):
    """
    Export to specified file the prediction results of given estimator on given testset.
    File saved is in csv format with a single column named 'predicted_values' and n_samples rows containing
    predicted values.
    Parameters
    ----------
    estimator: BaseEstimator or any object implementing predict() method as in BaseEstimator (for example sklearn)
        Fitted estimator to use for prediction
    X: ndarray of shape (n_samples, n_features)
        Test design matrix to predict its responses
    filename:
        path to store file at
    """
    pd.DataFrame(estimator.predict(X),
                 columns=["predicted_values"]).to_csv(filename, index=False)


def feature_evaluation(X: pd.DataFrame, y: pd.Series):
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """

    y_axis = []

    y_np = y.to_numpy()
    y_np = y_np.ravel()

    for i, feature in enumerate(X.columns):
        y_axis.append(
            np.cov(X[feature], y_np)[0, 1] / (
                    np.std(X[feature]) * np.std(y_np)))

    for i, feature in enumerate(X):
        create_scatter_for_feature(X[feature], y_np, round(y_axis[i], 3),
                                   feature)


def create_scatter_for_feature(X: pd.DataFrame, y: np.array, title,
                               feature_name: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X, y=y, mode="markers"))
    fig.update_layout(title="Pirson: " + str(title),
                      xaxis_title=feature_name,
                      yaxis_title="y")

    # fig.show()


def is_in_str(s: str, words: set):
    if type(s) != str:
        return False
    for w in words:
        if s.find(w) != -1:
            return True
    return False


def KI67_score(x):
    if (0 < x <= 5):
        return 1
    if (5 < x < 10):
        return 2
    if (10 <= x < 50):
        return 3
    if (50 <= x <= 100):
        return 4
    else:
        return 0


def KI67_pre(x):
    for sub in ['Sc', 'sc']:
        out = x.find(sub)
        if out != -1:
            for i in range(out, len(x)):
                if x[i].isdigit():
                    return int(x[i])
    sep = ['-', ' ', '=']
    x = "".join(filter(lambda c: c in sep or c.isdigit(), x)).replace('-',
                                                                      ' ').replace(
        '=', ' ')
    x = [int(s) for s in x.split(' ') if s.isdigit()]
    if x == []:
        return 0
    x = statistics.mean(x)
    if (0 < x < 100):
        return x
    else:
        return 0


def how_much_per_unique(x, d: dict):
    if x in d:
        d[x] += 1
    else:
        d[x] = 0


def Lymphatic_penetration_pre(x):
    if x[:2] == "L0":
        return 0
    elif x[:2] == "L1" or x[:2] == "LI":
        return 1
    elif x[:2] == "L2":
        return 2
    return None


def find_score(x: str):
    for sub in ['Sc', 'sc']:
        out = x.find(sub)

        if out != -1:
            for i in range(out, len(x)):
                if x[i].isdigit():
                    return x[i]
                elif x[i] == 'I':
                    return 1
    return None


def preprocess(df: pd.DataFrame):
    # Lymphatic penetration
    df["Lymphatic penetration"] = df["Lymphatic penetration"].apply(
        lambda x: Lymphatic_penetration_pre(x))

    # cur_date
    today = datetime.strptime("6/2/2022", '%d/%m/%Y')
    df["Diagnosis date"] = df["Diagnosis date"].apply(
        lambda x: (datetime.strptime(x[:10], '%d/%m/%Y') - today).days * -1)

    # make categorical from Hospital and Form Name
    X = pd.get_dummies(df, columns=[" Hospital",
                                    " Form Name",
                                    "Histopatological degree"])

    # Her2 preprocessing
    set_pos = {"po", "PO", "Po", "os", "2", "3", "+", "חיובי", 'בינוני',
               "Inter",
               "Indeter", "indeter", "inter"}
    set_neg = {"ne", "Ne", "NE", "eg", "no", "0", "1", "-", "שלילי"}

    X["Her2"] = X["Her2"].astype(str)
    # X["Her2"] = X["Her2"].apply(lambda x: 1 if is_in_str(x, set_pos) else x)
    # X["Her2"] = X["Her2"].apply(lambda x: 0 if is_in_str(x, set_neg) else x)
    # X["Her2"] = X["Her2"].apply(lambda x: 0 if type(x) == str else x)

    # more simple but same i think todo chek with elad
    X["Her2"] = X["Her2"].apply(lambda x: 1 if is_in_str(x, set_pos) else 0)

    # Age  preprocessing FIXME buggy, chek what need to do (remove line, get mean)
    X = X[X["Age"] < 120]
    X = X[0 < X["Age"]]

    # Basic stage preprocessing
    X["Basic stage"] = X["Basic stage"].replace(
        {'Null': 0, 'c - Clinical': 1, 'p - Pathological': 2,
         'r - Reccurent': 3})

    # KI67 protein preprocessing
    # print(sum(X["KI67 protein"].apply(lambda x: 1 if validate(x) else 0)))
    X["KI67 protein"] = X["KI67 protein"].astype(str)

    whitelist = ['-', ' ', '=']
    X["KI67 protein"] = X["KI67 protein"].apply(
        lambda x: KI67_score(KI67_pre(x)))
    print(sum(X["KI67 protein"] == 0) / X["KI67 protein"].size)
    print(X["KI67 protein"].unique())

    # margin type
    margin_neg = {'נקיים', 'ללא'}
    margin_pos = {'נגועים'}
    X["Margin Type"] = X["Margin Type"].apply(
        lambda x: 1 if is_in_str(x, margin_pos) else 0)

    return X


if __name__ == '__main__':
    np.random.seed(0)

    # Load data and preprocess
    # data_path, y_location_of_distal, y_tumor_path = sys.argv[1:]

    original_data = pd.read_csv("./Mission 2 - Breast Cancer/train.feats.csv")

    # remove heb prefix
    original_data.rename(columns=lambda x: x.replace('אבחנה-', ''),
                         inplace=True)

    y_tumor = pd.read_csv("./Mission 2 - Breast Cancer/train.labels.1.csv")
    y_tumor.rename(columns=lambda x: x.replace('אבחנה-', ''), inplace=True)

    for f in original_data.columns:
        print(f)

    print({f: original_data[f].unique().size for f in original_data.columns})
    print()

    d = {}
    original_data["Lymphatic penetration"].apply(
        lambda x: how_much_per_unique(x, d))
    print(d)

    X = preprocess(original_data)

    # feature_evaluation(X[["Age", "Her2", "Basic stage"]], y_tumor)
    print("this is me")
