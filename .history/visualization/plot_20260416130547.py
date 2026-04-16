import matplotlib.pyplot as plt


def plot_knn(points_list, target_point, top_k_neighbors, predicted_label):
    """
    This function exists to visualize the KNN classification result.

    points_list: all classified points from the dataset
    target_point: the point we want to classify, e.g. (8.5, 5.5)
    top_k_neighbors: the k nearest neighbors returned by the cal_topk_neighbors() function
    predicted_label: final label returned by the vote_for_label() function
    """

    # Colors are assigned to each class label.
    label_colors = {
        "A": "blue",
        "B": "green",
        "C": "orange"
    }

    # We create a new figure.
    plt.figure(figsize=(8, 6))

    # We eep track of which labels have already been added to the legend,
    # so the legend does not repeat the same label over and over.
    used_labels = set()

    # Plot all the dataset points.
    for point in points_list:
        x = point["x"]
        y = point["y"]
        label = point["label"]
        color = label_colors.get(label, "gray")  # fallback color if label is unknown

        # Only show each class label once in the legend.
        if label not in used_labels:
            plt.scatter(x, y, color=color, s=60, label=f"Class {label}")
            used_labels.add(label)
        else:
            plt.scatter(x, y, color=color, s=60)

    # We highlight the top-k neighbors with a black outline
    # and draw a dashed line from the target point to each one.
    for neighbor in top_k_neighbors:
        point = neighbor["point"]
        x = point["x"]
        y = point["y"]

        # Draw a dashed line from the target point to this neighbor.
        plt.plot(
            [target_point[0], x],
            [target_point[1], y],
            linestyle="dashed",
            linewidth=1
        )

        # Draw the neighbor again on top with a larger marker and black outline.
        plt.scatter(
            x,
            y,
            s=140,
            facecolors="none",
            edgecolors="black",
            linewidths=2
        )

    # Plot the target point as a large red X.
    plt.scatter(
        target_point[0],
        target_point[1],
        color="red",
        marker="x",
        s=150,
        linewidths=3,
        label="Target Point"
    )

    # We add the labels and title.
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"KNN Classification Result (Predicted Label: {predicted_label})")

    # Show legend and grid on the plot.
    plt.legend()
    plt.grid(True)

    # Display the plot.
    plt.show()