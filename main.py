import numpy as np
import copy
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# forward selection
def forward_selection(data, feature_num):
    selected_features = []
    total_best_feature = []
    total_best_accuracy = 0.
    accuracy_flow = []
    # for each loop, select the best feature
    for i in range(feature_num):
        round_best_feature = 0
        round_best_accuracy = 0.
        # each round, select a feature and use for KNN
        for j in range(1, feature_num + 1):
            if j in selected_features:
                continue
            selected_features.append(j)
            X = data[:, selected_features]
            y = data[:, 0]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)
            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            # select the best feature in this loop and store the best accuracy into a list
            if accuracy > round_best_accuracy:
                round_best_feature = j
                round_best_accuracy = accuracy
            if accuracy > total_best_accuracy:
                total_best_feature = copy.deepcopy(selected_features)
                total_best_accuracy = accuracy
            # print(f'    using feature {selected_features}, accuracy: {accuracy}')
            selected_features.pop(-1)
        selected_features.append(round_best_feature)
        accuracy_flow.append(round_best_accuracy)
        print(f'In this round, feature set {selected_features} is the best, accuracy: {round_best_accuracy}')
    print(f'Finish searching. The best feature in total {total_best_feature}, accuracy: {total_best_accuracy}')

    numbers = range(1, feature_num + 1)
    plt.figure(figsize=(10, 3))
    bars = plt.bar(numbers, accuracy_flow, color='gray')
    plt.title('The Accuracy with Different Numbers of Features')
    plt.xlabel('The number of features')
    plt.ylabel('Accuracy')
    plt.xticks(range(1, feature_num + 1, 1))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval}', ha='center', va='bottom')
    plt.show()


# backward elimination
def backward_elimination(data, feature_num):
    selected_features = list(range(1, feature_num + 1))
    total_best_feature = []
    total_best_accuracy = 0.
    accuracy_flow = []
    # use all the features as the start of backward elimination
    X = data[:, selected_features]
    y = data[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_flow.append(accuracy)
    print(f'feature set {selected_features} is the best, accuracy: {accuracy}')
    # for each loop, eliminate the worst feature
    for i in range(feature_num - 1):
        round_worst_feature = 0
        round_best_accuracy = 0.
        # each round, eliminate a feature and use the rest features for KNN
        for j in selected_features:
            elimination = selected_features.pop(0)
            X = data[:, selected_features]
            y = data[:, 0]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)
            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            # eliminate the worst feature in this loop and store the best accuracy into a list
            if accuracy > round_best_accuracy:
                round_worst_feature = elimination
                round_best_accuracy = accuracy
            if accuracy > total_best_accuracy:
                total_best_feature = copy.deepcopy(selected_features)
                total_best_accuracy = accuracy
            # print(f'    using feature {selected_features}: accuracy: {accuracy}')
            selected_features.append(elimination)
        selected_features.remove(round_worst_feature)
        accuracy_flow.append(round_best_accuracy)
        print(f'In this round, feature set {selected_features} is the best, accuracy: {round_best_accuracy}')
    print(f'Finish searching. The best feature in total {total_best_feature}: accuracy: {total_best_accuracy}')

    numbers = range(feature_num, 0, -1)
    plt.figure(figsize=(10, 3))
    bars = plt.bar(numbers, accuracy_flow, color='gray')
    plt.title('The Accuracy with Different Numbers of Features')
    plt.xlabel('The number of features')
    plt.ylabel('Accuracy')
    plt.xticks(range(feature_num, 0, -1))
    plt.gca().invert_xaxis()
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval}', ha='center', va='bottom')
    plt.show()


data_path = input("Welcome to Bertie Woosters Feature Selection Algorithm\n"
                  "Type in the name of the file to test: ")
algorithm = input("Type the number of the algorithm you want to run."
                  " 1)Forward Selection"
                  " 2)Backward Elimination: ")

dataset = np.loadtxt(data_path)
features = dataset.shape[1] - 1
instances = dataset.shape[0]
print(f"This dataset has {features} features (not including the class attribute), with {instances} instances.")

print("Begin searching")
if algorithm == '1':
    forward_selection(dataset, features)
elif algorithm == '2':
    backward_elimination(dataset, features)
else:
    print("Wrong number!")
