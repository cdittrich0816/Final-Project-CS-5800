import operator

from .distance import cal_euclidean_distance

"""
cal_topk_neighbors will return top k neighbors of target_point.
steps.
    1. Calculate the Euclidean Distance between the unclassified point and all other classified points.
    2. Sort by distance in ascending order.
    3. Return top k points.
"""
def cal_topk_neighbors(classified_points, target_point, k):
    distances = []

    # step 1. traverse the classified_points and calculate the distances between them and the target_point.
    for point in classified_points:
        # extract the coordinate of point
        current_point = (point['x'], point['y'])

        # call cal_euclidean_distance()
        dist = cal_euclidean_distance(target_point, current_point)

        # should put the distance and corresponding point(x, y, label) together for future voting.
        distances.append({
            'distance': dist,
            'point': point
        })

    # step 2. sort distances by ascending order.
    distances.sort(key=operator.itemgetter('distance'))

    # step 3. get topk points.
    top_k_neighbors = distances[:k]

    return top_k_neighbors


# vote_for_label will return the lexicographically smallest label.
def vote_for_label(top_k_neighbors):
    # step 1. Use a hash table to store the frequency of each label.
    label_votes = {}
    for neighbor in top_k_neighbors:
        if neighbor['point']['label'] not in label_votes:
            label_votes[neighbor['point']['label']] = 1
        else:
            label_votes[neighbor['point']['label']] += 1

    # step 2. simple vote: order by frequency and return the lexicographically smallest label.
    # 1. find the max frequency
    max_votes = 0
    for votes in label_votes.values():
        if votes > max_votes:
            max_votes = votes

    # 2. get all labels who have a max frequency
    max_labels = []
    for label, votes in label_votes.items():
        if votes == max_votes:
            max_labels.append(label)

    # 3. return the lexicographically smallest label
    max_labels.sort()
    return max_labels[0]
