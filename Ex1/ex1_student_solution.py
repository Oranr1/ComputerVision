"""Projective Homography and Panorama Solution."""
import numpy as np

from typing import Tuple
from random import sample
from collections import namedtuple


from numpy.linalg import svd
from scipy.interpolate import griddata


PadStruct = namedtuple('PadStruct',
                       ['pad_up', 'pad_down', 'pad_right', 'pad_left'])


class Solution:
    """Implement Projective Homography and Panorama Solution."""
    def __init__(self):
        pass

    @staticmethod
    def compute_homography_naive(match_p_src: np.ndarray,
                                 match_p_dst: np.ndarray) -> np.ndarray:
        """Compute a Homography in the Naive approach, using SVD decomposition.

        Args:
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.

        Returns:
            Homography from source to destination, 3x3 numpy array.
        """

        # Homoganic coordiantes
        src_vec = np.stack((match_p_src[0], match_p_src[1], np.ones(match_p_src.shape[1])), axis=1)
        dst_vec = np.stack((match_p_dst[0], match_p_dst[1], np.ones(match_p_dst.shape[1])), axis=1)
        h_mat = np.array([]).reshape(0, 9)

        # Building equation matrix rows
        for i in range(dst_vec.shape[0]):
            h_row_u = np.concatenate([src_vec[i], np.zeros(3), -1 * dst_vec[i, 0] * src_vec[i]])
            h_row_v = np.concatenate([np.zeros(3), src_vec[i], -1 * dst_vec[i, 1] * src_vec[i]])
            h_mat = np.vstack((h_mat, h_row_u, h_row_v))

        # Compute transpose(H)*H
        res_mat = np.matmul(h_mat.T, h_mat)

        # Get eigenvalues and eigenvectors
        eigen_val, eigen_vecs = np.linalg.eig(res_mat)
        min_eigen_vec = eigen_vecs[:, np.argmin(eigen_val)]

        homography = np.reshape(min_eigen_vec, (3, 3))
        return homography

    @staticmethod
    def compute_forward_homography_slow(
            homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute a Forward-Homography in the Naive approach, using loops.

        Iterate over the rows and columns of the source image, and compute
        the corresponding point in the destination image using the
        projective homography. Place each pixel value from the source image
        to its corresponding location in the destination image.
        Don't forget to round the pixel locations computed using the
        homography.

        Args:
            homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination
            image height, width and color dimensions.

        Returns:
            The forward homography of the source image to its destination.
        """
        # New image after transformation
        new_image = np.zeros(dst_image_shape, dtype=np.uint8)

        # iterate through location of each pixel and calculate transformed location
        for i in range(src_image.shape[0]):
            for j in range(src_image.shape[1]):
                point = [float(j), float(i), 1.0]
                t_point = np.matmul(homography, point)
                t_point = np.round(t_point[0:2] / t_point[2]).astype(int)
                if (0<=t_point[0]<dst_image_shape[1]) and (0<=t_point[1]<dst_image_shape[0]): # Check if transformed point is in new boundaries
                    new_image[t_point[1], t_point[0]] = src_image[i, j, :]

        return new_image

    @staticmethod
    def compute_forward_homography_fast(
            homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute a Forward-Homography in a fast approach, WITHOUT loops.

        (1) Create a meshgrid of columns and rows.
        (2) Generate a matrix of size 3x(H*W) which stores the pixel locations
        in homogeneous coordinates.
        (3) Transform the source homogeneous coordinates to the target
        homogeneous coordinates with a simple matrix multiplication and
        apply the normalization you've seen in class.
        (4) Convert the coordinates into integer values and clip them
        according to the destination image size.
        (5) Plant the pixels from the source image to the target image according
        to the coordinates you found.

        Args:
            homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination.
            image height, width and color dimensions.

        Returns:
            The forward homography of the source image to its destination.
        """

        """INSERT YOUR CODE HERE"""
        mesh_x, mesh_y = np.meshgrid(range(src_image.shape[1]), range(src_image.shape[0]))
        src_coor = np.stack((mesh_x, mesh_y, np.ones(src_image.shape[:2])))
        src_coor = src_coor.reshape((3, -1))

        dst_coor = np.matmul(homography, src_coor)
        dst_coor_norm = np.round(dst_coor / dst_coor[2])[:2, :]

        copy_coor = np.append(dst_coor_norm, src_coor[:2, :], axis=0).astype(int)

        mask1 = copy_coor.min(axis=0) >= 0
        copy_coor = copy_coor[:, mask1]
        mask2 = copy_coor[0, :] < dst_image_shape[1]
        copy_coor = copy_coor[:, mask2]
        mask3 = copy_coor[1, :] < dst_image_shape[0]
        copy_coor = copy_coor[:, mask3]

        new_image = np.zeros(dst_image_shape, dtype=np.uint8)
        new_image[copy_coor[1, :], copy_coor[0, :]] = src_image[copy_coor[3, :], copy_coor[2, :], :]

        return new_image

    @staticmethod
    def test_homography(homography: np.ndarray,
                        match_p_src: np.ndarray,
                        match_p_dst: np.ndarray,
                        max_err: float) -> Tuple[float, float]:
        """Calculate the quality of the projective transformation model.

        Args:
            homography: 3x3 Projective Homography matrix.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.

        Returns:
            A tuple containing the following metrics to quantify the
            homography performance:
            fit_percent: The probability (between 0 and 1) validly mapped src
            points (inliers).
            dist_mse: Mean square error of the distances between validly
            mapped src points, to their corresponding dst points (only for
            inliers). In edge case where the number of inliers is zero,
            return dist_mse = 10 ** 9.
        """
        # return fit_percent, dist_mse
        """INSERT YOUR CODE HERE"""

        match_p_src = np.vstack((match_p_src, np.ones(match_p_src.shape[1])))

        match_p_dst_est = np.dot(homography, match_p_src)
        match_p_dst_est = match_p_dst_est / match_p_dst_est[2]

        distance_arr = ((match_p_dst_est[0, :] - match_p_dst[0, :])**2 +
                        (match_p_dst_est[1, :] - match_p_dst[1, :])**2)**0.5

        indices_of_inliers = (distance_arr <= max_err)

        fit_percent = np.sum(indices_of_inliers) / match_p_src.shape[1]

        inliers_dist_arr = distance_arr[indices_of_inliers]

        dis_mes = np.sum(inliers_dist_arr**2) / np.sum(indices_of_inliers)

        return fit_percent, dis_mes

    @staticmethod
    def meet_the_model_points(homography: np.ndarray,
                              match_p_src: np.ndarray,
                              match_p_dst: np.ndarray,
                              max_err: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return which matching points meet the homography.

        Loop through the matching points, and return the matching points from
        both images that are inliers for the given homography.

        Args:
            homography: 3x3 Projective Homography matrix.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.
        Returns:
            A tuple containing two numpy nd-arrays, containing the matching
            points which meet the model (the homography). The first entry in
            the tuple is the matching points from the source image. That is a
            nd-array of size 2xD (D=the number of points which meet the model).
            The second entry is the matching points form the destination
            image (shape 2xD; D as above).
        """
        # return mp_src_meets_model, mp_dst_meets_model
        """INSERT YOUR CODE HERE"""
        match_p_dst_est = np.dot(homography, match_p_src)

        distance_arr = ((match_p_dst_est[0, :] - match_p_dst[0, :]) ** 2 +
                        (match_p_dst_est[1, :] - match_p_dst[1, :]) ** 2) ** 0.5

        indices_of_inliers = (distance_arr <= max_err)

        mp_src_meets_model = match_p_src[:, indices_of_inliers]
        mp_dst_meets_model = match_p_dst[:, indices_of_inliers]

        return mp_src_meets_model, mp_dst_meets_model

    def compute_homography(self,
                           match_p_src: np.ndarray,
                           match_p_dst: np.ndarray,
                           inliers_percent: float,
                           max_err: float) -> np.ndarray:
        """Compute homography coefficients using RANSAC to overcome outliers.

        Args:
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            inliers_percent: The expected probability (between 0 and 1) of
            correct match points from the entire list of match points.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.
        Returns:
            homography: Projective transformation matrix from src to dst.
        """
        # use class notations:
        w = inliers_percent
        # t = max_err
        # p = parameter determining the probability of the algorithm to
        # succeed
        p = 0.99
        # the minimal probability of points which meets with the model
        d = 0.5
        # number of points sufficient to compute the model
        n = 4
        # number of RANSAC iterations (+1 to avoid the case where w=1)
        k = int(np.ceil(np.log(1 - p) / np.log(1 - w ** n))) + 1

        max_fit_percent = 0
        homography = np.array([])

        for iter in range(k):

            curr_random_indices = np.random.randint(0, match_p_dst.shape[1], n)

            match_p_src_slice = match_p_src[:, curr_random_indices]
            match_p_dst_slice = match_p_dst[:, curr_random_indices]

            curr_homography = self.compute_homography_naive(match_p_src_slice, match_p_dst_slice)

            fit_percent, dis_mes = self.test_homography(curr_homography, match_p_src, match_p_dst, max_err)

            if fit_percent > max_fit_percent:
                max_fit_percent = fit_percent
                homography = curr_homography

        return homography

    @staticmethod
    def compute_backward_mapping(
            backward_projective_homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute backward mapping.

        (1) Create a mesh-grid of columns and rows of the destination image.
        (2) Create a set of homogenous coordinates for the destination image
        using the mesh-grid from (1).
        (3) Compute the corresponding coordinates in the source image using
        the backward projective homography.
        (4) Create the mesh-grid of source image coordinates.
        (5) For each color channel (RGB): Use scipy's interpolation.griddata
        with an appropriate configuration to compute the bi-cubic
        interpolation of the projected coordinates.

        Args:
            backward_projective_homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination shape.

        Returns:
            The source image backward warped to the destination coordinates.
        """

        # return backward_warp
        """INSERT YOUR CODE HERE"""
        grid_x_t, grid_y_t = np.mgrid[0:dst_image_shape[0], 0:dst_image_shape[1]]

        grid_x = grid_y_t
        grid_y = np.flip(grid_x_t)

        dst_image_3d_mesh = np.dstack(((grid_x), (grid_y), np.ones(grid_x.shape)))
        dst_image_homogenous_coor = dst_image_3d_mesh.reshape(grid_x.shape[0] * grid_x.shape[1], 3).T

        src_image_homogenous_coor = np.dot(backward_projective_homography, dst_image_homogenous_coor)
        src_image_homogenous_coor = src_image_homogenous_coor / src_image_homogenous_coor[2]

        src_image_3d_mesh = src_image_homogenous_coor.T.reshape(dst_image_shape)
        dst_to_src_grid_x = src_image_3d_mesh[:, :, 0]
        dst_to_src_grid_y = src_image_3d_mesh[:, :, 1]

        src_grid_x_t, src_grid_y_t = np.mgrid[0:src_image.shape[0], 0:src_image.shape[1]]

        src_grid_x = src_grid_y_t
        src_grid_y = np.flip(src_grid_x_t)

        mesh_mat = np.dstack((src_grid_x, src_grid_y))

        src_coor = mesh_mat.reshape(-1, 2)
        # src_coor = np.vstack((src_coor.T[1], np.flip(src_coor.T[0]))).T

        backward_warp_r = griddata(src_coor, src_image[:, :, 0].reshape(1, -1)[0], (dst_to_src_grid_x, dst_to_src_grid_y), method='cubic')
        backward_warp_g = griddata(src_coor, src_image[:, :, 1].reshape(1, -1)[0], (dst_to_src_grid_x, dst_to_src_grid_y), method='cubic')
        backward_warp_b = griddata(src_coor, src_image[:, :, 2].reshape(1, -1)[0], (dst_to_src_grid_x, dst_to_src_grid_y), method='cubic')

        backward_warp = np.dstack((backward_warp_r, backward_warp_g, backward_warp_b))

        return backward_warp

    @staticmethod
    def find_panorama_shape(src_image: np.ndarray,
                            dst_image: np.ndarray,
                            homography: np.ndarray
                            ) -> Tuple[int, int, PadStruct]:
        """Compute the panorama shape and the padding in each axes.

        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            homography: 3x3 Projective Homography matrix.

        For each image we define a struct containing it's corners.
        For the source image we compute the projective transformation of the
        coordinates. If some of the transformed image corners yield negative
        indices - the resulting panorama should be padded with at least
        this absolute amount of pixels.
        The panorama's shape should be:
        dst shape + |the largest negative index in the transformed src index|.

        Returns:
            The panorama shape and a struct holding the padding in each axes (
            row, col).
            panorama_rows_num: The number of rows in the panorama of src to dst.
            panorama_cols_num: The number of columns in the panorama of src to
            dst.
            padStruct = a struct with the padding measures along each axes
            (row,col).
        """
        src_rows_num, src_cols_num, _ = src_image.shape
        dst_rows_num, dst_cols_num, _ = dst_image.shape
        src_edges = {}
        src_edges['upper left corner'] = np.array([1, 1, 1])
        src_edges['upper right corner'] = np.array([src_cols_num, 1, 1])
        src_edges['lower left corner'] = np.array([1, src_rows_num, 1])
        src_edges['lower right corner'] = \
            np.array([src_cols_num, src_rows_num, 1])
        transformed_edges = {}
        for corner_name, corner_location in src_edges.items():
            transformed_edges[corner_name] = homography @ corner_location
            transformed_edges[corner_name] /= transformed_edges[corner_name][-1]
        pad_up = pad_down = pad_right = pad_left = 0
        for corner_name, corner_location in transformed_edges.items():
            if corner_location[1] < 1:
                # pad up
                pad_up = max([pad_up, abs(corner_location[1])])
            if corner_location[0] > dst_cols_num:
                # pad right
                pad_right = max([pad_right,
                                 corner_location[0] - dst_cols_num])
            if corner_location[0] < 1:
                # pad left
                pad_left = max([pad_left, abs(corner_location[0])])
            if corner_location[1] > dst_rows_num:
                # pad down
                pad_down = max([pad_down,
                                corner_location[1] - dst_rows_num])
        panorama_cols_num = int(dst_cols_num + pad_right + pad_left)
        panorama_rows_num = int(dst_rows_num + pad_up + pad_down)
        pad_struct = PadStruct(pad_up=int(pad_up),
                               pad_down=int(pad_down),
                               pad_left=int(pad_left),
                               pad_right=int(pad_right))
        return panorama_rows_num, panorama_cols_num, pad_struct

    @staticmethod
    def add_translation_to_backward_homography(backward_homography: np.ndarray,
                                               pad_left: int,
                                               pad_up: int) -> np.ndarray:
        """Create a new homography which takes translation into account.

        Args:
            backward_homography: 3x3 Projective Homography matrix.
            pad_left: number of pixels that pad the destination image with
            zeros from left.
            pad_up: number of pixels that pad the destination image with
            zeros from the top.

        (1) Build the translation matrix from the pads.
        (2) Compose the backward homography and the translation matrix together.
        (3) Scale the homography as learnt in class.

        Returns:
            A new homography which includes the backward homography and the
            translation.
        """
        # return final_homography
        """INSERT YOUR CODE HERE"""
        pass

    def panorama(self,
                 src_image: np.ndarray,
                 dst_image: np.ndarray,
                 match_p_src: np.ndarray,
                 match_p_dst: np.ndarray,
                 inliers_percent: float,
                 max_err: float) -> np.ndarray:
        """Produces a panorama image from two images, and two lists of
        matching points, that deal with outliers using RANSAC.

        (1) Compute the forward homography and the panorama shape.
        (2) Compute the backward homography.
        (3) Add the appropriate translation to the homography so that the
        source image will plant in place.
        (4) Compute the backward warping with the appropriate translation.
        (5) Create the an empty panorama image and plant there the
        destination image.
        (6) place the backward warped image in the indices where the panorama
        image is zero.
        (7) Don't forget to clip the values of the image to [0, 255].


        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            inliers_percent: The expected probability (between 0 and 1) of
            correct match points from the entire list of match points.
            max_err: A scalar that represents the maximum distance (in pixels)
            between the mapped src point to its corresponding dst point,
            in order to be considered as valid inlier.

        Returns:
            A panorama image.

        """
        # return np.clip(img_panorama, 0, 255).astype(np.uint8)
        """INSERT YOUR CODE HERE"""
        pass
