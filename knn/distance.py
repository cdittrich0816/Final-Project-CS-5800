import math

"""
cal_euclidean_distance calculates the Euclidean Distance between two points
    point1: the coordinate of the first point(x1, y1)
    point2: the coordinate of the second point(x2, y2)
    return: the Euclidean Distance between two points
"""
def cal_euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


# Local test
if __name__ == "__main__":
    p1 = (0, 0)
    p2 = (3, 4)
    print(f"Euclidean Distance: {cal_euclidean_distance(p1, p2)}")  # should be 5.0
