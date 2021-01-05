import numpy as np
from skimage.feature import ORB, match_descriptors
from skimage.color import rgb2gray
from skimage.transform import ProjectiveTransform
from skimage.transform import warp
from skimage.filters import gaussian
from numpy.linalg import inv

DEFAULT_TRANSFORM = ProjectiveTransform


def find_orb(img, n_keypoints=500):
    """Find keypoints and their descriptors in image.

    img ((W, H, 3)  np.ndarray) : 3-channel image
    n_keypoints (int) : number of keypoints to find

    Returns:
        (N, 2)  np.ndarray : keypoints
        (N, 256)  np.ndarray, type=np.bool  : descriptors
    """

    # your code here
    image_gray = rgb2gray(img)
    detector = ORB(n_keypoints=n_keypoints)
    detector.detect_and_extract(image_gray)
    return detector.keypoints, detector.descriptors


def center_and_normalize_points(points):
    """Center the image points, such that the new coordinate system has its
    origin at the centroid of the image points.

    Normalize the image points, such that the mean distance from the points
    to the origin of the coordinate system is sqrt(2).

    points ((N, 2) np.ndarray) : the coordinates of the image points

    Returns:
        (3, 3) np.ndarray : the transformation matrix to obtain the new points
        (N, 2) np.ndarray : the transformed image points
    """

    pointsh = np.row_stack([points.T, np.ones((points.shape[0]), )])
    matrix = np.zeros((3, 3))

    centre = np.mean(points, axis=0)
    points_new = points - centre
    dist = np.sqrt(np.sum(points_new ** 2) / points_new.shape[0])
    norm_cof = np.sqrt(2) / dist
    points_new *= norm_cof
    matrix = np.array([[norm_cof, 0, -norm_cof * centre[0]],
                       [0, norm_cof, -norm_cof * centre[1]],
                       [0, 0, 1]])
    return matrix, points_new


def make_hom_coord(src):
    return np.row_stack([src.T, np.ones(src.shape[0])]).T


def find_homography(src_keypoints, dest_keypoints):
    """Estimate homography matrix from two sets of N (4+) corresponding points.

    src_keypoints ((N, 2) np.ndarray) : source coordinates
    dest_keypoints ((N, 2) np.ndarray) : destination coordinates

    Returns:
        ((3, 3) np.ndarray) : homography matrix
    """

    src_matrix, src = center_and_normalize_points(src_keypoints)
    dest_matrix, dest = center_and_normalize_points(dest_keypoints)

    src = make_hom_coord(src)

    A = np.zeros((2 * src_keypoints.shape[0], 9))
    destx = dest[:, 0].reshape(dest.shape[0], 1)
    desty = dest[:, 1].reshape(dest.shape[0], 1)
    A[0::2] = np.row_stack([src.T * (-1),
                            np.zeros(src.shape).T,
                            src.T * destx.T]).T
    A[1::2] = np.row_stack([np.zeros((3, src.shape[0])),
                            src.T * (-1),
                            src.T * desty.T]).T

    H = np.linalg.svd(A)[2][-1].reshape((3, 3))
    return np.linalg.inv(dest_matrix) @ H @ src_matrix


def ransac_transform(src_keypoints, src_descriptors, dest_keypoints, dest_descriptors, max_trials=200,
                     residual_threshold=1, return_matches=False):
    """Match keypoints of 2 images and find ProjectiveTransform using RANSAC algorithm.

    src_keypoints ((N, 2) np.ndarray) : source coordinates
    src_descriptors ((N, 256) np.ndarray) : source descriptors
    dest_keypoints ((N, 2) np.ndarray) : destination coordinates
    dest_descriptors ((N, 256) np.ndarray) : destination descriptors
    max_trials (int) : maximum number of iterations for random sample selection.
    residual_threshold (float) : maximum distance for a data point to be classified as an inlier.
    return_matches (bool) : if True function returns matches

    Returns:
        skimage.transform.ProjectiveTransform : transform of source image to destination image
        (Optional)(N, 2) np.ndarray : inliers' indexes of source and destination images
    """
    # do matching of keypoints
    '''
    match_thr = 100
    new_src_keypoints = []
    new_dest_keypoints = []
    mask_src, mask_dest = [], []
    dest_take = np.ones(src_keypoints.shape[0])
    for i, d in enumerate(src_descriptors):
        for j, d_dest in enumerate(dest_descriptors):
            if dest_take[j] and np.sum(d != d_dest) < match_thr:
                mask_src.append(i)
                new_src_keypoints.append(src_keypoints[i])
                mask_dest.append(j)  # save in mask index of destination in initial array
                new_dest_keypoints.append(dest_keypoints[j])
                dest_take[j] = 0
                break
    '''
    mask = match_descriptors(src_descriptors, dest_descriptors, max_distance=1)
    new_src_keypoints = src_keypoints[mask[:, 0]]
    new_dest_keypoints = dest_keypoints[mask[:, 1]]

    if new_src_keypoints.shape != new_dest_keypoints.shape:
        print("=====================Different shape of src and dest!!!========================")
        return None, None

    # do ransac
    np.random.seed(0)
    inliers_ans = []
    for tr in range(max_trials):
        num_samples = np.random.choice(new_src_keypoints.shape[0], size=4, replace=False)
        src_samples, dest_samples = new_src_keypoints[num_samples], new_dest_keypoints[num_samples]
        H = find_homography(src_samples, dest_samples)
        inliers = np.argwhere(
            np.sqrt(np.sum((new_dest_keypoints - ProjectiveTransform(H)(new_src_keypoints)) ** 2, axis=1)) \
            < residual_threshold)
        if len(inliers) > len(inliers_ans):
            inliers_ans = inliers

    # do final inliers
    final_inliers = np.zeros((inliers_ans.shape[0], 2), dtype="int")
    for i, ind in enumerate(inliers_ans):
        final_inliers[i][0] = mask[int(ind)][0]
        final_inliers[i][1] = mask[int(ind)][1]

    H = find_homography(src_keypoints[final_inliers[:, 0]], dest_keypoints[final_inliers[:, 1]])
    return ProjectiveTransform(H), final_inliers


def find_simple_center_warps(forward_transforms):
    """Find transformations that transform each image to plane of the central image.

    forward_transforms (Tuple[N]) : - pairwise transformations

    Returns:
        Tuple[N + 1] : transformations to the plane of central image
    """
    image_count = len(forward_transforms) + 1
    center_index: int = (image_count - 1) // 2

    result = [DEFAULT_TRANSFORM()] * image_count
    result[center_index] = DEFAULT_TRANSFORM()

    for i in range(center_index - 1, -1, -1):
        result[i] = forward_transforms[i][0] + result[i + 1]
    for i in range(center_index + 1, image_count):
        result[i] = result[i - 1] + ProjectiveTransform(np.linalg.inv(forward_transforms[i - 1][0].params))
    return tuple(result)


def get_corners(image_collection, center_warps):
    """Get corners' coordinates after transformation."""
    for img, transform in zip(image_collection, center_warps):
        height, width, _ = img.shape
        corners = np.array([[0, 0],
                            [height, 0],
                            [height, width],
                            [0, width]])

        yield transform(corners)[:, ::-1]


def get_min_max_coords(corners):
    """Get minimum and maximum coordinates of corners."""
    corners = np.concatenate(corners)
    return corners.min(axis=0), corners.max(axis=0)


def get_final_center_warps(image_collection, simple_center_warps):
    """Find final transformations.

        image_collection (Tuple[N]) : list of all images
        simple_center_warps (Tuple[N])  : transformations unadjusted for shift

        Returns:
            Tuple[N] : final transformations,

        """
    # your code here
    corners = tuple(get_corners(image_collection, simple_center_warps))
    [y_min, x_min], [y_max, x_max] = get_min_max_coords(corners)
    tr_shift = ProjectiveTransform(np.array([[1, 0, -min(x_min, 0)],
                                             [0, 1, -min(y_min, 0)],
                                             [0, 0, 1]]))
    new_warps = []
    for tr in simple_center_warps:
        new_warps.append(tr + tr_shift)
    return tuple(new_warps), tuple((int(x_max - x_min), int(y_max - y_min)))


def rotate_transform_matrix(transform):
    """Rotate matrix so it can be applied to row:col coordinates."""
    matrix = transform.params[(1, 0, 2), :][:, (1, 0, 2)]
    return type(transform)(matrix)



def make_inverse_map(matrix):
    return rotate_transform_matrix(ProjectiveTransform(inv(matrix)))

def warp_image(image, transform, output_shape):
    """Apply transformation to an image and its mask

    image ((W, H, 3)  np.ndarray) : image for transformation
    transform (skimage.transform.ProjectiveTransform): transformation to apply
    output_shape (int, int) : shape of the final pano

    Returns:
        (W, H, 3)  np.ndarray : warped image
        (W, H)  np.ndarray : warped mask
    """
    return warp(image, make_inverse_map(transform.params), output_shape=output_shape), \
           warp(np.ones(image.shape[:2]), make_inverse_map(transform.params),
                output_shape=output_shape).astype(np.bool8)


def merge_pano(image_collection, final_center_warps, output_shape):
    """ Merge the whole panorama

    image_collection (Tuple[N]) : list of all images
    final_center_warps (Tuple[N])  : transformations
    output_shape (int, int) : shape of the final pano

    Returns:
        (output_shape) np.ndarray: final pano
    """
    result = np.zeros(output_shape + (3,))
    result_mask = np.zeros(output_shape, dtype=np.bool8)

    for i, image in enumerate(image_collection[::-1]):
        warp_imag, warp_mask = warp_image(image, final_center_warps[len(final_center_warps) - i - 1], output_shape)
        result[warp_mask] = warp_imag[warp_mask]
        result_mask += warp_mask

    return result


def normalize(image):
    mn = np.min(image)
    mx = np.max(image)
    return (image - mn) / (mx - mn)


def get_gaussian_pyramid(image, n_layers=10, sigma=3):
    """Get Gaussian pyramid.

    image ((W, H, 3)  np.ndarray) : original image
    n_layers (int) : number of layers in Gaussian pyramid
    sigma (int) : Gaussian sigma

    Returns:
        tuple(n_layers) Gaussian pyramid

    """
    pyramid = [image.copy()]
    image = image.copy()
    image_layer = np.copy(image).astype(np.float64)
    for i in range(n_layers - 1):
        image_layer = np.clip(gaussian(image_layer, sigma=sigma), 0, 1).astype(np.float64)
        pyramid.append(image_layer)
    return tuple(pyramid)


def get_laplacian_pyramid(image, n_layers=10, sigma=3):
    """Get Laplacian pyramid

    image ((W, H, 3)  np.ndarray) : original image
    n_layers (int) : number of layers in Laplacian pyramid
    sigma (int) : Gaussian sigma

    Returns:
        tuple(n_layers) Laplacian pyramid
    """
    pyramid_lap = []
    pyramid_gaus = get_gaussian_pyramid(image, n_layers, sigma)
    layer1 = pyramid_gaus[0]
    for layer2 in pyramid_gaus[1:]:
        pyramid_lap.append(layer1 - layer2)
        layer1 = layer2
    pyramid_lap.append(pyramid_gaus[-1])
    return tuple(pyramid_lap)

def merge_laplacian_pyramid(laplacian_pyramid):
    """Recreate original image from Laplacian pyramid

    laplacian pyramid: tuple of np.array (h, w, 3)

    Returns:
        np.array (h, w, 3)
    """
    return sum(laplacian_pyramid)


def increase_contrast(image_collection):
    """Increase contrast of the images in collection"""
    result = []

    for img in image_collection:
        img = img.copy()
        for i in range(img.shape[-1]):
            img[:, :, i] -= img[:, :, i].min()
            img[:, :, i] /= img[:, :, i].max()
        result.append(img)

    return result


def gaussian_merge_pano(image_collection, final_center_warps, output_shape, n_layers=10, image_sigma=3, merge_sigma=3):
    """ Merge the whole panorama using Laplacian pyramid

    image_collection (Tuple[N]) : list of all images
    final_center_warps (Tuple[N])  : transformations
    output_shape (int, int) : shape of the final pano
    n_layers (int) : number of layers in Laplacian pyramid
    image_sigma (int) :  sigma for Gaussian filter for images
    merge_sigma (int) : sigma for Gaussian filter for masks

    Returns:
        (output_shape) np.ndarray: final pano
    """
    result = np.zeros(output_shape + (3,))
    result_mask = np.zeros(output_shape, dtype=np.bool8)
    corners = tuple(get_corners(image_collection, final_center_warps))

    borderx = 0
    for i, image in enumerate(image_collection[::1]):
        warp_imag, warp_mask = warp_image(image, final_center_warps[i], output_shape)
        if i == 0:
            result[warp_mask] = warp_imag[warp_mask]
            borderx = int(corners[0][2][0])
        else:
            new_mask = np.zeros(result_mask.shape, dtype=np.bool8)
            new_mask[:, :borderx-250] = True
            borderx = int(corners[i][2][0])

            LA = get_laplacian_pyramid(result, n_layers, image_sigma)
            LB = get_laplacian_pyramid(warp_imag, n_layers, image_sigma)
            GM = get_gaussian_pyramid(new_mask, n_layers, merge_sigma)
            LS = []
            for la, lb, gm in zip(LA, LB, GM):
                ls = la * gm.reshape((gm.shape[0], gm.shape[1], 1)) + \
                     lb * (1 - gm).reshape((gm.shape[0], gm.shape[1], 1))
                LS.append(ls)
            result = merge_laplacian_pyramid(LS)
        result_mask += warp_mask
    return result
