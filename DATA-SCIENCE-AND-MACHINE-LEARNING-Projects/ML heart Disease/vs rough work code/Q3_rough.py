
import numpy as np
import pandas as pd

class MyDecisionTree:
    def __init__(self, max_depth):
        self.max_depth = max_depth

    def gini_index(self, groups, classes):
        total_instances = float(sum([len(group) for group in groups]))
        gini = 0.0
        for group in groups:
            group_size = float(len(group))
            if group_size == 0:
                continue
            score = 0.0
            for class_val in classes:
                p = [row[-1] for row in group].count(class_val) / group_size
                score += p * p
            gini += (1.0 - score) * (group_size / total_instances)
        return gini

    def make_split(self, dataset):
        class_values = list(set(row[-1] for row in dataset))
        best_index, best_value, best_score, best_groups = 999, 999, 999, None
        for index in range(len(dataset[0]) - 1):
            for row in dataset:
                groups = self.test_split(index, row[index], dataset)
                gini = self.gini_index(groups, class_values)
                if gini < best_score:
                    best_index, best_value, best_score, best_groups = index, row[index], gini, groups
        return {'index': best_index, 'value': best_value, 'groups': best_groups}

    def test_split(self, index, value, dataset):
        left, right = list(), list()
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    def to_terminal(self, group):
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)

    def split(self, node, max_depth, min_size, depth):
        left, right = node['groups']
        del (node['groups'])
        if not left or not right:
            node['left'] = node['right'] = self.to_terminal(left + right)
            return
        if depth >= max_depth:
            node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
            return
        if len(left) <= min_size:
            node['left'] = self.to_terminal(left)
        else:
            node['left'] = self.make_split(left)
            self.split(node['left'], max_depth, min_size, depth + 1)
        if len(right) <= min_size:
            node['right'] = self.to_terminal(right)
        else:
            node['right'] = self.make_split(right)
            self.split(node['right'], max_depth, min_size, depth + 1)

    def build_tree(self, train, max_depth, min_size):
        root = self.make_split(train)
        self.split(root, max_depth, min_size, 1)
        return root

    
    def pruning(self, tree, depth=0):
        if 'groups' in tree:
            if not tree['left'] or not tree['right']:
                tree['left'] = tree['right'] = self.to_terminal(tree['groups'])
            if depth >= self.max_depth:
                tree['left'] = self.to_terminal(tree['groups'][0])
                tree['right'] = self.to_terminal(tree['groups'][1])
            if isinstance(tree['left'], dict):
                self.pruning(tree['left'], depth + 1)
            if isinstance(tree['right'], dict):
                self.pruning(tree['right'], depth + 1)


    def predict(self, node, row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.predict(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict(node['right'], row)
            else:
                return node['right']

    def fit(self, features, target):
        dataset = np.column_stack((features, target))
        self.tree = self.build_tree(dataset, self.max_depth, 1)

    def evaluate_accuracy(self, features, target):
        predictions = []
        for row in features:
            prediction = self.predict(self.tree, row)
            predictions.append(prediction)
        predictions = np.array(predictions)
        accuracy = np.mean(predictions == target)
        return accuracy

# Load the dataset
# Replace 'your_data.csv' with the actual path to your data file
data = pd.read_csv('/content/Thyroid data - Sheet1.csv')

# Preprocess the data
# Assuming the data is preprocessed and encoded properly

# Mapping the classes to binary values (0 and 1)
data['label'] = data['label'].map({'negative': 0, 'hyperthyroid': 1, 'T3 toxic': 1, 'goitre': 1})

# Extract features and target variable
features = data.drop('label', axis=1).values
target = data['label'].values

# Creating an instance of MyDecisionTree
max_tree_depth = 5  # Example max depth, you can change it to any desired value
decision_tree_model = MyDecisionTree(max_tree_depth)

# Fitting the decision tree on the data
decision_tree_model.fit(features, target)

# Pruning the decision tree
decision_tree_model.pruning(decision_tree_model.tree)

# Evaluating the model
accuracy = decision_tree_model.evaluate_accuracy(features, target)
print(f"Accuracy of the Decision Tree model: {accuracy:.2f}")

