from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier


#################################################################
# This function used in order to check what is the best estimator
#################################################################
# We saw that the best estimator is the random forest:
def check_estimators(train_data, labels):
    from sklearn.metrics import accuracy_score
    train_X, test_X, train_y, test_y = train_test_split(train_data, labels)
    stump_tree = DecisionTreeClassifier(max_depth=1)
    max_depth_tree = DecisionTreeClassifier(max_depth=200)
    multi_target_stump = MultiOutputClassifier(stump_tree, n_jobs=2)
    multi_target_stump.fit(train_data, labels)
    stump_multi_score = multi_target_stump.score(test_X, test_y)
    print("Stump multi score ", stump_multi_score, "Accuracy ",
          accuracy_score(multi_target_stump.predict(test_X), test_y,
                         normalize=False))

    multi_target_tree = MultiOutputClassifier(max_depth_tree, n_jobs=2)
    multi_target_tree.fit(train_data, labels)
    tree_multi_score = multi_target_tree.score(test_X, test_y)
    print("Tree multi score ", tree_multi_score, "Accuracy ",
          accuracy_score(multi_target_tree.predict(test_X), test_y,
                         normalize=False))

    forest = RandomForestClassifier(random_state=1)
    multi_target_forest = MultiOutputClassifier(forest, n_jobs=2)
    multi_target_forest.fit(train_X, train_y)
    forest_multi_score = multi_target_forest.score(test_X, test_y)
    print("Forst score ", forest_multi_score, "Accuracy ",
          accuracy_score(multi_target_forest.predict(test_X), test_y,
                         normalize=False))


############################################
######## PREDICTION FUNCTION ###############
###########################################
def run_predicting_metastases(train_data, labels, test_data):

    forest = RandomForestClassifier(random_state=1)
    multi_target_forest = MultiOutputClassifier(forest, n_jobs=2)
    multi_target_forest.fit(train_data, labels)
    return multi_target_forest.predict(test_data)

