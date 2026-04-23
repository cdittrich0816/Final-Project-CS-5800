import csv
import os
import random

from data.prepare_data import prepare_iris_csv
from visualization.plot import plot_knn
from knn.knn import cal_topk_neighbors, vote_for_label
import numpy as np


"""
parse the dataset- x, y, label, return a list.
"""
def parse_data(filename):
    points_list = []
    # if data file doesn't exist.
    if not os.path.exists(filename):
        print(f" Can not find file: {filename}")
        return points_list

    with open(filename, mode='r', encoding='utf-8') as file:
        # Get a csv object from file
        csv_reader = csv.reader(file)
        i= 0
        for row in csv_reader:
            if row[0] == "x":
                continue
            i += 1
            points_list.append({
                'x': float(row[0]),  # to make the distance calculation more precise, transform it to float.
                'y': float(row[1]),
                'label': row[2]
            })
        print("rows parsed := ", i)
    return points_list

"""
training using train_points and testing with val_points.
"""
def compute_accuracy(train_points, val_points, k):
    right_predictions = 0

    # train and test
    for target_point in val_points:
        target_xy = (target_point['x'], target_point['y'])
        target_label = target_point['label']

        top_k_neighbors = cal_topk_neighbors(train_points, target_xy, k)
        predicted_label = vote_for_label(top_k_neighbors)

        if predicted_label == target_label:
            right_predictions += 1

    # calculate the accuracy.
    return right_predictions / len(val_points)

"""
Implement a k-fold cross-validation in training dataset and return the optimal k.
"""
def k_fold_cross_validation(points_list, k_ranges, num_folds=5):
    # 0. shuffle all points.
    shuffled_points = points_list[:]
    random.seed(42)
    random.shuffle(shuffled_points)

    # 1. get the size of each fold, should be 24.
    fold_size = len(shuffled_points) // num_folds
    folds = []

    # 2. decide where each fold start at and end at.
    for i in range(num_folds):
        start = i * fold_size
        if i == num_folds - 1:
            end = len(shuffled_points)
        else:
            end = (i + 1) * fold_size
        folds.append(shuffled_points[start:end])

    # 3. initialization.
    best_k = None
    best_avg_accuracy = -1
    k_to_avg_accuracy = {}

    # 4. test for the values of k.
    for k in k_ranges:
        fold_accuracies = []

        for i in range(num_folds):
            # use the ith fold as validation set.
            val_points = folds[i]
            train_points = []

            # use any other folds as training sets.
            for j in range(num_folds):
                if j != i:
                    train_points.extend(folds[j])

            # train and test, then return the accuracy.
            accuracy = compute_accuracy(train_points, val_points, k)

            # accuracy using the ith fold as validation set.
            fold_accuracies.append(accuracy)

        # what's the avg accuracy of k for 5 folds?
        avg_accuracy = sum(fold_accuracies) / len(fold_accuracies)
        k_to_avg_accuracy[k] = avg_accuracy

        print(f"k = {k}, fold accuracies = {[round(a, 4) for a in fold_accuracies]}, avg = {avg_accuracy:.4f}")

        # update the best k.
        if avg_accuracy > best_avg_accuracy:
            best_avg_accuracy = avg_accuracy
            best_k = k

    return best_k, best_avg_accuracy, k_to_avg_accuracy


def main():
    # 0. prepare Iris data from UCI repository.
    prepare_iris_csv()

    # 1. parse the training data file.
    train_filename = "data/train_dataset.csv"
    points_list = parse_data(train_filename)

    # 2. parse the testing data file.
    test_filename = "data/test_dataset.csv"
    target_points_list = parse_data(test_filename)

    # 3. use k-fold cross-validation to select best k
    k_ranges = list(range(1, 16))
    best_k, best_avg_accuracy, k_to_avg_accuracy = k_fold_cross_validation(points_list, k_ranges, num_folds=5)

    print(f"Best k = {best_k}")
    print(f"Best average validation accuracy = {best_avg_accuracy * 100:.2f}%")

    k = best_k

    # 4. Generate decision boundary
    x_values = [point["x"] for point in points_list]
    y_values = [point["y"] for point in points_list]

    x_min, x_max = min(x_values) - 0.5, max(x_values) + 0.5
    y_min, y_max = min(y_values) - 0.5, max(y_values) + 0.5

    step = 0.05
    x_range = np.arange(x_min, x_max, step)
    y_range = np.arange(y_min, y_max, step)

    with open("data/decision_boundary.csv", mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["x", "y", "label"])

        for y in y_range:
            for x in x_range:
                target_xy = (round(float(x), 1), round(float(y), 1))

                top_k_neighbors = cal_topk_neighbors(points_list, target_xy, k)
                predicted_label = vote_for_label(top_k_neighbors)

                writer.writerow([target_xy[0], target_xy[1], predicted_label])

    # 5. Evaluate on test dataset
    right_predictions = 0

    for idx, target_point in enumerate(target_points_list):
        target_xy = (target_point['x'], target_point['y'])
        target_label = target_point['label']

        top_k_neighbors = cal_topk_neighbors(points_list, target_xy, k)

        print("\nTop K Neighbors: ")
        for i, item in enumerate(top_k_neighbors):
            point = item['point']
            dist = item['distance']
            print(
                f"Top {i + 1}: point({point['x']}, {point['y']}) | "
                f"label: {point['label']} | distance: {dist:.4f}"
            )

        predicted_label = vote_for_label(top_k_neighbors)

        if predicted_label == target_label:
            right_predictions += 1

        print(f"Predicted_Label: {predicted_label}, Original_Label: {target_label}")

    print(f"\nTest correct rate: {float(right_predictions / len(target_points_list)) * 100:.2f}%")

    # 6. visualize one target point
    target_xy = (7.4, 6.2)

    top_k_neighbors = cal_topk_neighbors(points_list, target_xy, k)
    predicted_label = vote_for_label(top_k_neighbors)

    plot_knn(points_list, target_xy, top_k_neighbors, predicted_label)


if __name__ == "__main__":
    main()
