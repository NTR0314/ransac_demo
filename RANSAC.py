import math
from copy import copy
from math import cos, sin

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng

rng = default_rng()


class RANSAC:
    def __init__(self, data_pairs, n=4, k=100, t=0.05, d=10, model=None, loss=None, metric=None):
        self.data_pairs = data_pairs
        self.n = n  # `n`: Minimum number of data points to estimate parameters
        self.k = k  # `k`: Maximum iterations allowed
        self.t = t  # `t`: Threshold value to determine if points are fit well
        self.d = d  # `d`: Number of close data points required to assert model fits well
        self.model = model  # `model`: class implementing `fit` and `predict`
        self.loss = loss  # `loss`: function of `y_true` and `y_pred` that returns a vector
        self.metric = metric  # `metric`: function of `y_true` and `y_pred` and returns a float
        self.best_fit = None
        self.best_error = np.inf

    def fit(self):
        """

        :param X: List of tuples of point pairs of the original image and the transformed image
        :return: best homography found
        """
        for cur_iter in range(self.k):
            #
            ids = rng.permutation(self.data_pairs.shape[0])

            maybe_inliers = ids[: self.n]
            not_chosen_d = ids[self.n:]
            # TODO homography fit mit 4 vielen point pairs
            maybe_model = copy(self.model).fit(self.data_pairs[maybe_inliers])

            # TODO loss implementieren mit [n:] vielen pairs indem von dem neuen zuruecktransformiert wird auf das alte
            # thresholded = (self.loss(self.data_pairs[ids][self.n:], maybe_model.predict(self.data_pairs[ids][self.n:])) < self.t)
            thresholded = (self.loss(self.data_pairs[ids][self.n:], maybe_model.predict(self.data_pairs[ids][self.n:])) < self.t)

            inlier_ids = ids[self.n:][np.flatnonzero(thresholded).flatten()]

            if inlier_ids.size > self.d:
                inlier_points = np.hstack([maybe_inliers, inlier_ids])
                better_model = copy(self.model).fit(X[inlier_points], y[inlier_points])
                this_error = self.metric(
                    y[inlier_points], better_model.predict(X[inlier_points])
                )

                if this_error < self.best_error:
                    self.best_error = this_error
                    self.best_fit = maybe_model

                    # Plot
                    fig, ax = plt.subplots(1, 1)
                    ax.set_box_aspect(1)
                    plt.scatter(X[not_chosen_d], y[not_chosen_d])
                    plt.scatter(X[maybe_inliers], y[maybe_inliers])
                    plt.scatter(X[inlier_ids], y[inlier_ids])
                    line = np.linspace(-1, 1, num=100).reshape(-1, 1)
                    plt.plot(line, self.best_fit.predict(line), c="peru")
                    plt.savefig(f'{cur_iter}.png')

        return self

    def predict(self, X):
        return self.best_fit.predict(X)


def square_error_loss(y_true, y_pred):
    return (y_true - y_pred) ** 2


def mean_square_error(y_true, y_pred):
    return np.sum(square_error_loss(y_true, y_pred)) / y_true.shape[0]


class Homography:
    def __init__(self, min_eucl_dist=0.6):
        self.homography = None
        self.min_eucl_dist = min_eucl_dist

    def fit(self, pairs):
        # taken from: https://github.com/dastratakos/Homography-Estimation/blob/main/imageAnalysis.py
        """Solves for the homography given any number of pairs of points. Visit
        http://6.869.csail.mit.edu/fa12/lectures/lecture13ransac/lecture13ransac.pdf
        slide 9 for more details.

        Args:
            pairs (List[List[List]]): List of pairs of (x, y) points.

        Returns:
            np.ndarray: The computed homography.
        """
        A = []
        for x1, y1, x2, y2 in pairs:
            A.append([x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2])
            A.append([0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2])
        A = np.array(A)

        # Singular Value Decomposition (SVD)
        U, S, V = np.linalg.svd(A)

        # V has shape (9, 9) for any number of input pairs. V[-1] is the eigenvector
        # of (A^T)A with the smalles eigenvalue. Reshape into 3x3 matrix.
        H = np.reshape(V[-1], (3, 3))

        # Normalization
        H = (1 / H.item(8)) * H
        self.homography = H

    def predict(self, all_pairs):
        """

        :param all_pairs:
        :return: inliers, outliers
        """
        inliers = []
        outliers = []
        for x1, y1, x2, y2 in all_pairs:
            if math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) < self.min_eucl_dist:
                inliers.append([[x1, y1], [x2, y2]])
            else:
                outliers.append([[x1, y1], [x2, y2]])

        return inliers, outliers


class LinearRegressor:
    def __init__(self):
        self.params = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        r, _ = X.shape
        X = np.hstack([np.ones((r, 1)), X])
        self.params = np.linalg.inv(X.T @ X) @ X.T @ y
        return self

    def predict(self, X: np.ndarray):
        r, _ = X.shape
        X = np.hstack([np.ones((r, 1)), X])
        return X @ self.params


if __name__ == "__main__":
    bio_metr_point_image = np.array([[0.10107015, 0.7073236], [0.09953588, 0.69776027], [0.10107015, 0.66428858],
                                     [0.10260443, 0.65089991], [0.10260443, 0.6308169], [0.10107015, 0.60977756],
                                     [0.09953588, 0.57534954], [0.09493307, 0.53996519], [0.0933988, 0.50362451],
                                     [0.10567297, 0.45867682], [0.11027579, 0.7015856], [0.14096122, 0.70923627],
                                     [0.18852365, 0.7015856], [0.23301753, 0.68724059], [0.2729086, 0.66907025],
                                     [0.28211423, 0.63751124], [0.2729086, 0.61934089], [0.25756588, 0.60308322],
                                     [0.21614054, 0.58204388], [0.14402976, 0.5629172], [0.16090675, 0.5629172],
                                     [0.21767481, 0.55622286], [0.28518277, 0.51892585], [0.2943884, 0.4950175],
                                     [0.28211423, 0.47875983], [0.19926355, 0.44337548], [0.12868705, 0.44624448],
                                     [0.42940432, 0.52466385], [0.42787005, 0.50840617], [0.42633578, 0.48449783],
                                     [0.42940432, 0.46250215], [0.43093859, 0.44337548], [0.44167849, 0.58108754],
                                     [0.57516014, 0.50266817], [0.59203713, 0.51796951], [0.54294043, 0.50840617],
                                     [0.53833762, 0.48067249], [0.56748878, 0.45676415], [0.62885965, 0.44337548],
                                     [0.68869625, 0.45006981], [0.70403897, 0.47589082], [0.67642208, 0.50744984],
                                     [0.62118829, 0.52753285], [0.09953588, 0.3563491], [0.10107015, 0.33339709],
                                     [0.11334433, 0.34391676], [0.14096122, 0.36686877], [0.09800161, 0.34009143],
                                     [0.1041387, 0.30757608], [0.10720724, 0.27697339], [0.1041387, 0.26549739],
                                     [0.11641287, 0.32765909], [0.13328986, 0.35443643], [0.19159219, 0.36017444],
                                     [0.22688044, 0.34391676], [0.2345518, 0.31618308], [0.22994899, 0.28940573],
                                     [0.22994899, 0.27792973], [0.23762034, 0.29992541], [0.28057995, 0.34487309],
                                     [0.31126539, 0.34869843], [0.3434851, 0.33339709], [0.3434851, 0.30375074],
                                     [0.34655364, 0.28079873], [0.43860995, 0.32479008], [0.48310383, 0.32670275],
                                     [0.51072072, 0.33435342], [0.43554141, 0.34200409], [0.42173296, 0.31905208],
                                     [0.44474704, 0.29036207], [0.50765218, 0.27888606], [0.60891412, 0.37260677],
                                     [0.60277703, 0.35826177], [0.60277703, 0.33626609], [0.61044839, 0.30948874],
                                     [0.61044839, 0.28079873], [0.61198266, 0.26836639], [0.55828315, 0.33435342],
                                     [0.58436577, 0.33530976], [0.60584558, 0.33435342], [0.63653101, 0.33148442],
                                     [0.64420237, 0.33148442], [0.512255, 0.3477421], [0.50918645, 0.34869843],
                                     [0.49384373, 0.35443643], [0.46469257, 0.35539277], [0.44474704, 0.34678576],
                                     [0.43400713, 0.32670275], [0.42787005, 0.30375074], [0.44167849, 0.28940573],
                                     [0.47696674, 0.28079873], [0.50918645, 0.27697339], [0.54600898, 0.27314806],
                                     [0.45088412, 0.32383375], [0.46929539, 0.32670275], [0.69790188, 0.26741006],
                                     [0.69790188, 0.26836639], [0.69329907, 0.28749307], [0.69636761, 0.30661974],
                                     [0.71938169, 0.32000842], [0.74086149, 0.32383375], [0.78075256, 0.32000842],
                                     [0.69636761, 0.32574642], [0.69329907, 0.31618308], [0.69790188, 0.30375074],
                                     [0.73625868, 0.26167205], [0.73625868, 0.26167205], [0.3051283, 0.18707802],
                                     [0.30052549, 0.18994702], [0.25910015, 0.19855402], [0.21614054, 0.20142302],
                                     [0.17318093, 0.19377235], [0.15016685, 0.17368934], [0.19005792, 0.146912],
                                     [0.22534617, 0.13352332], [0.29131986, 0.12013465], [0.3051283, 0.10196431],
                                     [0.29745694, 0.08379397], [0.26677151, 0.07136163], [0.21614054, 0.06562363],
                                     [0.12408423, 0.06944896], [0.35269073, 0.12109099], [0.3649649, 0.11535298],
                                     [0.37570481, 0.10578964], [0.38951325, 0.09335731], [0.41099306, 0.0818813],
                                     [0.43400713, 0.06944896], [0.49230946, 0.11439665], [0.48003529, 0.10578964],
                                     [0.46929539, 0.09526997], [0.45548694, 0.07709963], [0.44321277, 0.06657996],
                                     [0.42633578, 0.05510395], [0.40178743, 0.03693361], [0.38644471, 0.03406461],
                                     [0.59970849, 0.10483331], [0.58129723, 0.11344032], [0.57055732, 0.11344032],
                                     [0.5444747, 0.10961498], [0.53220053, 0.09813897], [0.54600898, 0.09144464],
                                     [0.56288597, 0.08570664], [0.57822868, 0.0790123], [0.59817422, 0.07231796],
                                     [0.5935714, 0.05892929], [0.56288597, 0.05319129], [0.5337348, 0.05223495],
                                     [0.75927276, 0.48449783], [0.77154693, 0.48449783], [0.79149246, 0.48162883],
                                     [0.80530091, 0.47971616], [0.81450654, 0.48067249], [0.83445207, 0.47875983],
                                     [0.8712746, 0.47875983], [0.87894596, 0.47875983], [0.63806528, 0.51892585],
                                     [0.59663994, 0.51796951], [0.56748878, 0.50649351], [0.54754325, 0.47684716],
                                     [0.62885965, 0.44337548], [0.68409344, 0.43859381], [0.70710751, 0.45580782],
                                     [0.70710751, 0.47302182], [0.67642208, 0.50744984], [0.43707568, 0.51701318],
                                     [0.43400713, 0.50936251], [0.43247286, 0.4892795], [0.43707568, 0.45963315],
                                     [0.43707568, 0.45389515], [0.19312646, 0.55909187], [0.20846918, 0.55622286],
                                     [0.26370296, 0.54092152], [0.2729086, 0.53327085], [0.29131986, 0.50266817],
                                     [0.27904568, 0.46728382], [0.26830578, 0.46154582], [0.20386636, 0.44624448]])
    # https://en.wikipedia.org/wiki/Rotation_matrix
    rot = np.array([
        [cos(math.pi / 6), - sin(math.pi / 6)],
        [sin(math.pi / 6), cos(math.pi / 6)],
    ])
    trans = np.array([-1, 0])
    scale = np.array([
        [2, 0],
        [0, 2]
    ])

    x = bio_metr_point_image[:, 0]
    y = bio_metr_point_image[:, 1]
    plt.scatter(x, y)

    bio_metr_point_image_new = bio_metr_point_image + trans
    bio_metr_point_image_new = np.array([rot @ x for x in bio_metr_point_image_new])
    bio_metr_point_image_new = np.array([scale @ x for x in bio_metr_point_image_new])

    x = bio_metr_point_image_new[:, 0]
    y = bio_metr_point_image_new[:, 1]
    plt.scatter(x, y)
    # plt.show()

    all_pairs = np.dstack((bio_metr_point_image, bio_metr_point_image_new))
