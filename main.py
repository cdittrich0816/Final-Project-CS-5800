import csv
import os

from data.prepare_data import prepare_iris_csv
from visualization.plot import plot_knn
from knn.knn import cal_topk_neighbors, vote_for_label
import numpy as np

def parse_data(filename):
    points_list = []
    # if data file doesn't exist.
    if not os.path.exists(filename):
        print(f" Can not find file: {filename}")
        return points_list

    with open(filename, mode='r', encoding='utf-8') as file:
        # read the first row of the data file as header(keys of our dictionary).
        csv_reader = csv.reader(file)
        for row in csv_reader:
            points_list.append({
                'x': float(row[0]),  # to make the distance calculation more precise, transform it to float.
                'y': float(row[1]),
                'label': row[2]
            })
    return points_list

def main():
    # 0. prepare Iris data from UCI repository.
    # prepare_iris_csv()

    # 1. parse the training data file.
    filename = "data/train_dataset.csv"
    points_list = parse_data(filename)

    # 2. parse the testing data file.
    filename = "data/test_dataset.csv"
    target_points_list = parse_data(filename)
    right_predictions = 0
    k = 6  # k

    # 3. Generate decision boundary
    step = 0.1
    x_range = np.arange(0, 10.1, step)
    y_range = np.arange(0, 10.1, step)

    with open("./data/decision_boundary.csv", mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["x", "y", "label"])

        for y in y_range:
            for x in x_range:
                target_xy = (round(float(x), 1), round(float(y), 1))

                top_k_neighbors = cal_topk_neighbors(points_list, target_xy, k)
                predicted_label = vote_for_label(top_k_neighbors)

                writer.writerow([target_xy[0], target_xy[1], predicted_label])

    for idx, target_point in enumerate(target_points_list):
        # 3. extract x and y for target_point
        target_xy = (target_point['x'], target_point['y'])
        target_label = (target_point['label'])

        # 4. get top k nearest neighbors for target_node.
        top_k_neighbors = cal_topk_neighbors(points_list, target_xy, k)

        # 5. print tok k neighbors.
        print("\nTop K Neighbors: ")
        for i, item in enumerate(top_k_neighbors):
            point = item['point']
            dist = item['distance']
            print(f"Top {i + 1}: point({point['x']}, {point['y']}) | label: {point['label']} | distance: {dist:.4f}")

        # 6. Vote for label.
        predicted_label = vote_for_label(top_k_neighbors)

        if predicted_label == target_label:
            right_predictions += 1

        # 7. Print the predicted label.
        print(f"Predicted_Label: {predicted_label}, Original_Label: {target_label}")

    print(f"\nprediction correct rate: {float(right_predictions / 30)}")

    target_xy = (5.4, 1.9)

    top_k_neighbors = cal_topk_neighbors(points_list, target_xy, k)
    predicted_label = vote_for_label(top_k_neighbors)

    plot_knn(points_list, target_xy, top_k_neighbors, predicted_label, k)

if __name__ == "__main__":
    main()
