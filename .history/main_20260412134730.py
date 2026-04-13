import csv
import os

from visualization.plot import plot_knn
from knn.knn import cal_topk_neighbors, vote_for_label

def parse_data(filename):
    points_list = []
    # if data file doesn't exist.
    if not os.path.exists(filename):
        print(f" Can not find file: {filename}")
        return points_list

    with open(filename, mode='r', encoding='utf-8') as file:
        # read the first row of the data file as header(keys of our dictionary).
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            points_list.append({
                'x': float(row['x']),  # to make the distance calculation more precise, transform it to float.
                'y': float(row['y']),
                'label': row['label']
            })
    return points_list

def main():
    # 1. parse the csv file.
    filename = "data/dataset.csv"
    points_list = parse_data(filename)

    # 2. variables to declare.
    target_point = (8.5, 5.5)  # the point we want to classify.
    k = 6  # k

    # 3. get top k nearest neighbors for target_node.
    top_k_neighbors = cal_topk_neighbors(points_list, target_point, k)

    # 4. print tok k neighbors.
    print("\nTop K Neighbors: ")
    for i, item in enumerate(top_k_neighbors):
        point = item['point']
        dist = item['distance']
        print(f"Top {i + 1}: point({point['x']}, {point['y']}) | label: {point['label']} | distance: {dist:.4f}")

    # 5. vote_for_label
    predicted_label = vote_for_label(top_k_neighbors)
print(f"Label: {predicted_label}")


if __name__ == "__main__":
    main()
