import cv2
import numpy as np


def bezier_curve(cp, t):
    """
    Get point on bezier curve at t [0~1]
    :param cp: Nx2 Control points
    :param t: point ratio on the curve [0, 1]
    :return: point
    """
    n = len(cp)
    p = np.zeros((1, 2), dtype=np.float32)
    for i in range(n):
        a = n - 1
        if i == 0 or i == n-1:
            a = 1
        p += a * t**i * (1-t)**(n-1-i) * cp[i]
    return p


def get_bezier_coeff(idx, n, t):
    """
    Compute coefficients of bezier curve
    :param idx: index of control point
    :param n: number of control point
    :param t: t of bezier curve
    :return: coefficient
    """
    a = n - 1
    if idx == 0 or idx == 3:
        a = 1
    return a * t**idx * (1-t)**(n-1-idx)


def bezier_curve_get_cp(pts, t, ncp):
    """
    Get contol point of bezier curve from points on the curve
    :param pts: points on the curve (Nx2)
    :param t: t on the curve (N)
    :param ncp: number of control points of a bezier curve
    :return: control points of bezier curve (ncp, 2)
    """
    n = len(pts)
    assert n == len(t), "number of pts and t should be same"
    assert n >= ncp, "number of pts should be same or greater than number of control point"
    p = pts[1:-1]
    coeff = np.zeros((n-2, ncp), dtype=np.float32)
    for i in range(n-2):
        for j in range(ncp):
            coeff[i, j] = get_bezier_coeff(j, ncp, t[i + 1])

    return np.matmul(np.linalg.pinv(coeff), p)


def get_sequential_distance(pts):
    result = []
    for i in range(len(pts)-1):
        distance = pts[i+1] - pts[i]
        norm = np.linalg.norm(distance)
        result.append(norm)
    return result


def cumulate(sequence):
    result = [sequence[0]]
    for i in range(1, len(sequence)):
        a = result[i-1] + sequence[i]
        result.append(a)
    return result


def simulation():
    img_dim = (300, 500, 3)  # image size
    # cp = np.array([[10, 250], [50, 100], [100, 100], [490, 250]], dtype=np.float32)    # control points
    # cp = np.array([[10, 250], [150, 100], [350, 100], [490, 250]], dtype=np.float32)  # control points
    # cp = np.array([[10, 250], [20, 50], [50, 50], [490, 250]], dtype=np.float32)  # control points
    cp = np.array([[10, 50], [150, 100], [350, 100], [490, 50]], dtype=np.float32)  # control points

    r = np.arange(0, 1, 0.01)
    n = 10

    img = np.zeros(img_dim, dtype=np.float32)
    for t in r:
        p = bezier_curve(cp, t)
        cv2.circle(img, (int(p[0, 0]), int(p[0, 1])), 5, (0, 255, 0), thickness=3)
    cv2.circle(img, (cp[0, 0], cp[0, 1]), 5, (0, 0, 255), thickness=3)
    cv2.circle(img, (cp[1, 0], cp[1, 1]), 5, (0, 0, 255), thickness=3)
    cv2.circle(img, (cp[2, 0], cp[2, 1]), 5, (0, 0, 255), thickness=3)
    cv2.circle(img, (cp[3, 0], cp[3, 1]), 5, (0, 0, 255), thickness=3)
    cv2.imwrite('./temp.jpg', img)

    img2 = np.zeros(img_dim, dtype=np.float32)
    lmk1 = np.zeros((n, 2), dtype=np.float32)
    # t1 = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # t1 = [0, 0.1, 0.2, 0.3, 0.5, 1.0]
    t_rnd = np.linspace(0.01, 0.99, 99)
    np.random.seed(100)
    indices = np.random.randint(0, 98, n - 2)
    indices = sorted(indices)
    t_rnd = t_rnd[indices]
    t1 = [0.0]
    t1.extend(t_rnd)
    t1.append(1.0)
    print(t1)
    for i in range(n):
        lmk1[i] = bezier_curve(cp, t1[i])
    for lm in lmk1:
        cv2.circle(img2, (int(lm[0]), int(lm[1])), 5, (0, 255, 0), thickness=3)
    cv2.imwrite('./temp2.jpg', img2)

    norms = get_sequential_distance(lmk1)

    norms = norms / sum(norms)

    t1_1 = cumulate(norms)
    t1_2 = [0.0]
    t1_2.extend(t1_1)
    print(t1_2)

    # cp2_predict = bezier_curve_get_cp(lmk1, t1, 4)                                      # Calculate control points
    cp2_predict = bezier_curve_get_cp(lmk1, t1_2, 4)  # Calculate control points
    cp2_predict2 = cp2_predict.copy()
    cp2_predict2[0] = cp[0]
    cp2_predict2[-1] = cp[-1]

    cp2 = cp2_predict2

    print(cp)
    print(cp2_predict)

    img3 = np.zeros(img_dim, dtype=np.float32)
    for t in r:
        p = bezier_curve(cp2, t)
        # p = bezier_curve(cp2_predict2, t)
        cv2.circle(img3, (int(p[0, 0]), int(p[0, 1])), 5, (0, 255, 0), thickness=3)
    cv2.circle(img3, (cp2[0, 0], cp2[0, 1]), 5, (0, 0, 255), thickness=3)
    cv2.circle(img3, (cp2[1, 0], cp2[1, 1]), 5, (0, 0, 255), thickness=3)
    cv2.circle(img3, (cp2[2, 0], cp2[2, 1]), 5, (0, 0, 255), thickness=3)
    cv2.circle(img3, (cp2[3, 0], cp2[3, 1]), 5, (0, 0, 255), thickness=3)
    cv2.imwrite('./temp3.jpg', img3)

    img4 = np.zeros(img_dim, dtype=np.float32)
    t2 = [0, 0.25, 0.5, 0.75, 1.0]
    for t in t2:
        p = bezier_curve(cp2_predict, t)
        cv2.circle(img4, (int(p[0, 0]), int(p[0, 1])), 5, (0, 255, 0), thickness=3)
    cv2.imwrite('./temp4.jpg', img4)


def eyebrow():
    img_dim = (600, 600, 3)  # image size
    img = np.zeros(img_dim, dtype=np.float32)
    pts_eyebrow = np.array([[151.58, 98.881], [141.506, 92.556], [129.324, 89.745], [118.079, 90.682]], dtype=np.float32)
    # pts_eyebrow = [[118.079, 90.682], [129.324, 89.745], [141.506, 92.556], [151.58, 98.881]]
    # pts_eyebrow *= 3
    t = [0, 0.3, 0.6, 1.0]
    bn = 4
    cp = bezier_curve_get_cp(pts_eyebrow, t, bn)
    print(cp)
    r = np.arange(0, 1, 0.01)
    n = 100
    for t in r:
        p = bezier_curve(cp, t)
        # p = bezier_curve(cp2_predict2, t)
        cv2.circle(img, (int(p[0, 0]), int(p[0, 1])), 5, (0, 255, 0), thickness=3)
    for i in range(bn):
        cv2.circle(img, (int(cp[i, 0]), int(cp[i, 1])), 5, (0, 0, 255), thickness=3)
    for lm in pts_eyebrow:
        cv2.circle(img, (int(lm[0]), int(lm[1])), 5, (255, 0, 0), thickness=3)

    cv2.imshow("result", img)
    cv2.waitKey(0)


def eyebrow2():
    img_dim = (600, 600, 3)  # image size
    img = np.zeros(img_dim, dtype=np.float32)
    # pts = np.array([[151.58, 98.881], [141.506, 92.556], [129.324, 89.745], [118.079, 90.682]], dtype=np.float32)
    pts = np.array([[10, 300], [100, 200], [200, 200], [300, 300]])

    t = [0, 0.3, 0.7, 1.0]
    bn = 3
    cp = bezier_curve_get_cp(pts, t, bn)
    print(cp)
    r = np.arange(0, 1, 0.01)
    n = 100
    for t in r:
        p = bezier_curve(cp, t)
        # p = bezier_curve(cp2_predict2, t)
        cv2.circle(img, (int(p[0, 0]), int(p[0, 1])), 5, (0, 255, 0), thickness=3)
    for i in range(bn):
        cv2.circle(img, (int(cp[i, 0]), int(cp[i, 1])), 5, (0, 0, 255), thickness=3)
    for lm in pts:
        cv2.circle(img, (int(lm[0]), int(lm[1])), 5, (255, 0, 0), thickness=3)

    cv2.imshow("result", img)
    cv2.waitKey(0)


if __name__ == '__main__':
    # simulation()
    eyebrow2()
