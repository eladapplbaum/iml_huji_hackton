import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import cross_validate, train_test_split
import plotly.graph_objects as go

from prediction import preprocess

if __name__ == '__main__':
    full_data = pd.read_csv("./Mission 2 - Breast Cancer/train.feats.csv")

    labels = pd.read_csv("./Mission 2 - Breast Cancer/train.labels.1.csv")
    # classified_labels = np.where(labels == 0, 0, 1)
    processed_data = preprocess(full_data)
    train_X, test_X, train_y, test_y = train_test_split(processed_data,
                                                        np.array(
                                                            labels).reshape(
                                                            -1))
    classified_train_y = np.where(train_y == 0, 0, 1)
    classified_test_y = np.where(test_y == 0, 0, 1)
    from sklearn.metrics import mean_squared_error

    train_lost = []
    test_lost = []

    range_of_chek = np.arange(3, 50)
    for i in range_of_chek:
        print(i)

        base_estimator = DecisionTreeRegressor(max_depth=i)
        base_estimator.fit(train_X, classified_train_y)

        train_lost.append(
            mean_squared_error(classified_train_y,
                               base_estimator.predict(train_X)))
        test_lost.append(
            mean_squared_error(classified_test_y,
                               base_estimator.predict(test_X)))

    fig = go.Figure(data=[
        go.Bar(x=range_of_chek, y=train_lost, name="train lost"),
        go.Bar(x=range_of_chek, y=test_lost, name="test lost")])
    fig.show()

    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import LogisticRegression

    linear_model = LinearRegression()
    logistic_model = LogisticRegression()

    linear_model.fit(train_X, train_y)
    lin_pred = linear_model.predict(test_X)
    lin_success = linear_model.score(test_X, test_y)

    fig = go.Figure(data=[
        go.Scatter(x=np.arange(train_y.size), y=lin_pred, name="pred",
                   mode="markers"),
        go.Scatter(x=np.arange(train_y.size), y=labels, name="true",
                   mode="markers"),
    ])
    fig.show()

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysi

    train_score = []
    test_scores = []
    validation = []

    for i in range(5, 15):
        # poly = PolynomialFeatures(i)
        # poly.fit(train_X, train_y)
        # poly.transform()
        # mean_squared_error(test_y, )

        print(i)
    ada = AdaBoostRegressor(DecisionTreeRegressor(max_depth=3),
                            n_estimators=i)
    ada.fit(train_X, train_y)
    scoring = cross_validate(ada, train_X, train_y, cv=5)
    validation.append(scoring['test_score'])
    # pred = ada.predict(test_X)

    train_score.append(ada.score(train_X, train_y))
    test_scores.append(ada.score(test_X, test_y))

    print(train_score)
    print(test_scores)
    print(validation)
