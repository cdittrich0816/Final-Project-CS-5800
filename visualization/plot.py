import matplotlib.pyplot as plt
import numpy as np

from knn.knn import cal_topk_neighbors, vote_for_label


def plot_knn(points_list, target_point, top_k_neighbors, predicted_label, k):
    """
    Visualize the KNN classification result with decision boundaries.

    points_list: the list of all the data points
    target_point: the point we want to classify
    top_k_neighbors: the k nearest neighbors for the target point
    predicted_label: the predicted class label for the target point
    k: the number of neighbors used in KNN
    """

    # Color mapping for Iris labels
    label_colors = {
        "Iris-setosa": "blue",
        "Iris-versicolor": "green",
        "Iris-virginica": "orange"
    }

    # Numeric mapping for the boundary regions
    label_to_num = {
        "Iris-setosa": 0,
        "Iris-versicolor": 1,
        "Iris-virginica": 2
    }

    # Extract x and y values from the dataset
    x_values = [point["x"] for point in points_list]
    y_values = [point["y"] for point in points_list]

    # Add padding around the plotted area
    x_min, x_max = min(x_values) - 0.5, max(x_values) + 0.5
    y_min, y_max = min(y_values) - 0.5, max(y_values) + 0.5

    # Create a mesh grid of points across the plot
    step_size = 0.05
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, step_size),
        np.arange(y_min, y_max, step_size)
    )

    # Predict a label for every point in the grid
    z = []
    for grid_x, grid_y in zip(xx.ravel(), yy.ravel()):
        neighbors = cal_topk_neighbors(points_list, (grid_x, grid_y), k)
        label = vote_for_label(neighbors)
        z.append(label_to_num[label])

    z = np.array(z).reshape(xx.shape)

    # Draw the decision boundary background
    plt.figure(figsize=(9, 7))
    plt.contourf(xx, yy, z, alpha=0.25, levels=[-0.5, 0.5, 1.5, 2.5])

    used_labels = set()

    # Plot all dataset points
    for point in points_list:
        x = point["x"]
        y = point["y"]
        label = point["label"]
        color = label_colors.get(label, "gray")
        display_label = label.replace("Iris-", "")

        if display_label not in used_labels:
            plt.scatter(x, y, color=color, s=60, label=display_label)
            used_labels.add(display_label)
        else:
            plt.scatter(x, y, color=color, s=60)

    # Highlight the nearest neighbors and connect them to the target point
    for neighbor in top_k_neighbors:
        point = neighbor["point"]
        x = point["x"]
        y = point["y"]

        plt.plot(
            [target_point[0], x],
            [target_point[1], y],
            linestyle="dashed",
            linewidth=1,
            color="black"
        )

        plt.scatter(
            x,
            y,
            s=140,
            facecolors="none",
            edgecolors="black",
            linewidths=2
        )

    # Plot the target point
    plt.scatter(
        target_point[0],
        target_point[1],
        color="red",
        marker="x",
        s=150,
        linewidths=3,
        label="Target Point"
    )

    plt.xlabel("Sepal Length")
    plt.ylabel("Sepal Width")
    plt.title(
        f"KNN Classification with Decision Boundaries "
        f"(Predicted Label: {predicted_label.replace('Iris-', '')})"
    )
    plt.legend()
    plt.grid(True)
    plt.show()
