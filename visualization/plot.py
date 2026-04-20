import csv
import matplotlib.pyplot as plt


def plot_knn(points_list, target_point, top_k_neighbors, predicted_label):
    """
    Visualize the KNN classification result with decision boundaries.

    points_list: the list of all the data points
    target_point: the point we want to classify
    top_k_neighbors: the k nearest neighbors for the target point
    predicted_label: the predicted class label for the target point
    """

    # Map the Iris labels to colors
    label_colors = {
        "Iris-setosa": "blue",
        "Iris-versicolor": "green",
        "Iris-virginica": "orange"
    }

    # Map the class labels to numeric values for plotting
    label_to_num = {
        "Iris-setosa": 0,
        "Iris-versicolor": 1,
        "Iris-virginica": 2
    }

    # Read the decision boundary data from the CSV file
    boundary_rows = []

    with open("./data/decision_boundary.csv", mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            boundary_rows.append({
                "x": float(row["x"]),
                "y": float(row["y"]),
                "z": label_to_num[row["label"]]
            })

    # Get sorted unique x and y values from the CSV file
    x_unique = sorted({row["x"] for row in boundary_rows})
    y_unique = sorted({row["y"] for row in boundary_rows})

    # Build a lookup table from (x, y) -> class number
    z_lookup = {(row["x"], row["y"]): row["z"] for row in boundary_rows}

    # Reconstruct the grid
    z_grid = []
    for y in y_unique:
        row_vals = []
        for x in x_unique:
            row_vals.append(z_lookup[(x, y)])
        z_grid.append(row_vals)

    plt.figure(figsize=(9, 7))
    plt.contourf(x_unique, y_unique, z_grid, levels=[-0.5, 0.5, 1.5, 2.5], alpha=0.25)


    used_labels = set()

    # Plot all the points in the dataset
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

    x_values = [point["x"] for point in points_list]
    y_values = [point["y"] for point in points_list]

    plt.xlim(min(x_values) - 0.5, max(x_values) + 0.5)
    plt.ylim(min(y_values) - 0.5, max(y_values) + 0.5)

    plt.show()
