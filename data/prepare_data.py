from ucimlrepo import fetch_ucirepo

"""
prepare_iris_csv prepares training data and testing data from UCI repository.
"""
def prepare_iris_csv():
    # 1. Fetch dataset
    iris = fetch_ucirepo(id=53)
    df = iris.data.original

    # 2. build new combined features
    result_df = df.copy()

    # x = sepal length + sepal width
    result_df['x'] = df.iloc[:, 0] + df.iloc[:, 1]
    # y = petal length + petal width
    result_df['y'] = df.iloc[:, 2] + df.iloc[:, 3]
    # get label
    result_df['label'] = df.iloc[:, 4]

    # we only need 3 columns- x, y, label
    selected_df = result_df[['x', 'y', 'label']].copy()

    # 3. shuffle
    shuffled_df = selected_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 4. split
    train_df = shuffled_df.iloc[:120]
    test_df = shuffled_df.iloc[120:]

    # 5. save
    train_df.to_csv('data/train_dataset.csv', index=False)
    test_df.to_csv('data/test_dataset.csv', index=False)


if __name__ == "__main__":
    prepare_iris_csv()
