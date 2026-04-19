from ucimlrepo import fetch_ucirepo

"""
prepare_iris_csv prepares training data and test data from UCI repository.
"""
def prepare_iris_csv():
    # 1. Fetch the dataset from UCI Repository
    iris = fetch_ucirepo(id=53)
    df = iris.data.original

    # 2. Select the first two features and iris label.
    # iloc[:, [0, 1, -1]]: select Sepal Length, Sepal Width, and iris label
    selected_df = df.iloc[:, [0, 1, -1]].copy()

    # 3. rename the columns to make it aligned with our knn logic.
    selected_df.columns = ['x', 'y', 'label']

    # 4. Shuffle the dataset to get a random sequence.
    # frac=1: shuffle all of the data.
    # random_state=42: make the random seed a fixed one to ensure every time we run it, we would get the same result.
    shuffled_df = selected_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 5. 80% for training, 20% for testing
    train_df = shuffled_df.iloc[:120]
    test_df = shuffled_df.iloc[120:]

    # 6. Save to CSV files
    train_df.to_csv('data/train_dataset.csv', index=False)
    test_df.to_csv('data/test_dataset.csv', index=False)

if __name__ == "__main__":
    prepare_iris_csv()
