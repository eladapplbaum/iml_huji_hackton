import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

CHOSEN_DEPTH = 20


def predict_classifier(X: pd.DataFrame, fitted_model):
    pred = fitted_model.predict(X)
    # pred.reshape(-1)
    has_cancer_ind = np.argwhere(np.where(pred == 0, 0, 1) == 1)
    has_cancer_ind = has_cancer_ind.reshape(-1)

    return pred, np.take(X, indices=has_cancer_ind, axis=0), has_cancer_ind


# We use this function in order to find the best classifier.
# We found that the max depth should be 20. Explanations can be found in the
# PDF with supported graph
def find_depth_of_tree(max_depth, train_X, classified_train_y, test_X,
                       classified_test_y):
    train_lost = []
    test_lost = []

    range_of_check = np.arange(1, max_depth)
    for i in range_of_check:
        base_estimator = DecisionTreeRegressor(max_depth=i)
        base_estimator.fit(train_X, classified_train_y)

        train_lost.append(
            mean_squared_error(classified_train_y,
                               base_estimator.predict(train_X)))
        prd_clas = base_estimator.predict(test_X)

        test_lost.append(mean_squared_error(classified_test_y, prd_clas))

    fig = go.Figure(data=[
        go.Bar(x=range_of_check, y=train_lost, name="train lost"),
        go.Bar(x=range_of_check, y=test_lost, name="test lost")])
    fig.update_layout(title="loss on train and test as func of depth")
    fig.show()


def fit_classifier(X: pd.DataFrame, y: pd.Series):
    classifier = DecisionTreeRegressor(max_depth=CHOSEN_DEPTH)
    classified_y = np.where(y == 0, 0, 1)

    return classifier.fit(X, classified_y)


def split_data(X: pd.DataFrame, y: pd.DataFrame):
    classified_y = np.where(y == 0, 0, 1)
    has_cancer_ind = np.argwhere((classified_y.reshape(-1)) == 1)
    has_cancer_ind = has_cancer_ind.reshape(
        has_cancer_ind.shape[0])
    have_cancer_X = np.take(X, indices=has_cancer_ind,
                            axis=0)
    have_cancer_y = np.take(y, indices=has_cancer_ind,
                            axis=0)
    return have_cancer_X, have_cancer_y


def run_tumor_size_pred(train_data, labels, test_data):

    fitted_cls = fit_classifier(train_data, labels)
    have_cancer_train_X, have_cancer_train_y = split_data(train_data, labels)

    linear_model = LinearRegression()
    linear_model.fit(have_cancer_train_X, have_cancer_train_y)

    full_pred, cancer_pred, indx = predict_classifier(test_data, fitted_cls)
    linear_pred = linear_model.predict(cancer_pred)
    np.put(full_pred, indx, linear_pred)
    return full_pred

    # lin_pred_ = linear_model.predict(have_cancer_train_X)
    # lin_success_train = np.square(
    #     np.subtract(have_cancer_train_y, lin_pred_).mean())
    # success = np.square(np.subtract(have_cancer_test_y, lin_pred).mean())
