import csv
import matplotlib.pyplot as plt


def plot_knn(points_list, target_point, top_k_neighbors, predicted_label):
    """
    This function is used to visualize the KNN classification result with decision boundaries included.

    points_list: the list of all the data points
    target_point: the point we want to classify
    top_k_neighbors: the k nearest neighbors for the target point
    predicted_label: the predicted class label for the target point
    """

    if len(points_list) == 0:
        raise ValueError("points_list is empty")

    if len(top_k_neighbors) == 0:
        raise ValueError("top_k_neighbors is empty")

    if len(target_point) != 2:
        raise ValueError("target_point has to contain exactly two values")

    # This maps the Iris labels to colors
    colors_of_labels = {
        "Iris-setosa": "blue",
        "Iris-versicolor": "green",
        "Iris-virginica": "orange"
    }

    # This maps the class labels to numeric values for plotting
    label_to_num = {
        "Iris-setosa": 0,
        "Iris-versicolor": 1,
        "Iris-virginica": 2
    }

    # In order to display the boundaries, we need to first read the decision boundary data from the CSV file
    boundary_file = "./data/decision_boundary.csv"
    boundary_rows = []

    with open(boundary_file, mode="r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            boundary_rows.append({
                "x": float(row["x"]),
                "y": float(row["y"]),
                "z": label_to_num[row["label"]]
            })

    if len(boundary_rows) == 0:
        raise ValueError("decision_boundary.csv is empty")

    # Next, we get the sorted unique x and y values from the CSV file
    unique_x_vals = sorted({row["x"] for row in boundary_rows})
    unique_y_vals = sorted({row["y"] for row in boundary_rows})

    # Then we build a lookup table from (x, y) -> class number
    lookup = {(row["x"], row["y"]): row["z"] for row in boundary_rows}

    # Finally, we reconstruct the grid
    grid = []
    for y in unique_y_vals:
        row_vals = []
        for x in unique_x_vals:
            row_vals.append(lookup[(x, y)])
        grid.append(row_vals)

    # This plots the actual decision boundary regions. The boundaries appear
    # where the colors change.
    plt.figure(figsize=(9, 7))
    plt.contourf(unique_x_vals, unique_y_vals, grid, levels=[-0.5, 0.5, 1.5, 2.5], alpha=0.25)


    used_labels = set()

    # This plots all the points in the dataset
    for point in points_list:
        x = point["x"]
        y = point["y"]
        label = point["label"]
        color = colors_of_labels.get(label, "gray")
        display_label = label.replace("Iris-", "")

        if display_label not in used_labels:
            plt.scatter(x, y, color=color, s=60, label=display_label)
            used_labels.add(display_label)
        else:
            plt.scatter(x, y, color=color, s=60)

    # This highlights the nearest neighbors and connects them to the target point
    for neighbor in top_k_neighbors:
        point = neighbor["point"]
        x = point["x"]
        y = point["y"]


        plt.plot(
            [target_point[0], x], [target_point[1], y],
            linestyle="dashed", linewidth=1,
            color="black"
        )

        plt.scatter(
            x, y, s=140,
            facecolors="none",
            edgecolors="black",
            linewidths=2
        )

    # This plots the target point on the graph
    plt.scatter(
        target_point[0], target_point[1],
        color="red", marker="x",
        s=150, linewidths=3,
        label="Target Point"
    )

    # This plots the axes labels, as well as the graph title, legend, and grid.
    plt.xlabel("Sepal Length + Sepal Width")
    plt.ylabel("Petal Length + Petal Width")
    plt.title(
        f"KNN Classification Result with Decision Boundaries "
        f"(Predicted Label: {predicted_label.replace('Iris-', '')})"
    )
    plt.legend()
    plt.grid(True)


    # This adds axes limits based on the data points and adds a small margin so that
    # the graph focuses on the relevant data.
    x_values = [point["x"] for point in points_list]
    y_values = [point["y"] for point in points_list]
    plt.xlim(min(x_values) - 0.5, max(x_values) + 0.5)
    plt.ylim(min(y_values) - 0.5, max(y_values) + 0.5)


    plt.show()
