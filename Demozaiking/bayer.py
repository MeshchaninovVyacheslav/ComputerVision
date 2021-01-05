import numpy as np
from scipy.signal import convolve2d


def build_mask(mask_row, n_rows):
    mask = np.tile(mask_row, (n_rows // 2, 1))
    if n_rows % 2 == 1:
        mask = np.row_stack((mask, mask_row[0]))
    return mask


def get_bayer_masks(n_rows, n_cols):
    mask_row_red = np.zeros((2, n_cols), dtype='bool')
    mask_row_red[0, 1::2] = 1
    mask_red = build_mask(mask_row_red, n_rows)

    mask_row_green = np.zeros((2, n_cols), dtype='bool')
    mask_row_green[0, ::2] = 1
    mask_row_green[1, 1::2] = 1
    mask_green = build_mask(mask_row_green, n_rows)

    mask_row_blue = np.zeros((2, n_cols), dtype='bool')
    mask_row_blue[1, 0::2] = 1
    mask_blue = build_mask(mask_row_blue, n_rows)

    return np.dstack((mask_red, mask_green, mask_blue))


def get_colored_img(raw_img):
    mask = get_bayer_masks(raw_img.shape[0], raw_img.shape[1])
    return np.dstack((raw_img, raw_img, raw_img)) * mask


def bilinear_interpolation(colored_img):
    kernel = np.ones((3, 3))
    cnv = np.dstack((convolve2d(colored_img[:, :, 0], kernel, mode='same'),
                     convolve2d(colored_img[:, :, 1], kernel, mode='same'),
                     convolve2d(colored_img[:, :, 2], kernel, mode='same')))

    mask = get_bayer_masks(colored_img.shape[0], colored_img.shape[1])
    mask_cnv = np.dstack((convolve2d(mask[:, :, 0], kernel, mode='same'),
                          convolve2d(mask[:, :, 1], kernel, mode='same'),
                          convolve2d(mask[:, :, 2], kernel, mode='same')))
    cnv = cnv / mask_cnv
    return np.array(cnv, dtype='uint8') * (~mask) + colored_img


def improved_interpolation(raw_img):
    g_at_r = np.array([[0, 0, -1, 0, 0],
                       [0, 0, 2, 0, 0],
                       [-1, 2, 4, 2, -1],
                       [0, 0, 2, 0, 0],
                       [0, 0, -1, 0, 0]], dtype='float32') / 8
    g_at_b = g_at_r
    r_at_g1 = np.array([[0, 0, 1 / 2, 0, 0],
                        [0, -1, 0, -1, 0],
                        [-1, 4, 5, 4, -1],
                        [0, -1, 0, -1, 0],
                        [0, 0, 1 / 2, 0, 0]]) / 8
    r_at_g2 = np.array([[0, 0, -1, 0, 0],
                        [0, -1, 4, -1, 0],
                        [1 / 2, 0, 5, 0, 1 / 2],
                        [0, -1, 4, -1, 0],
                        [0, 0, -1, 0, 0]]) / 8
    r_at_b = np.array([[0, 0, -3 / 2, 0, 0],
                       [0, 2, 0, 2, 0],
                       [-3 / 2, 0, 6, 0, -3 / 2],
                       [0, 2, 0, 2, 0],
                       [0, 0, -3 / 2, 0, 0]]) / 8
    b_at_g1 = r_at_g1
    b_at_g2 = r_at_g2
    b_at_r = r_at_b

    img = np.array(get_colored_img(raw_img), dtype='float32')
    mask = get_bayer_masks(img.shape[0], img.shape[1])

    green = convolve2d(img[:, :, 0] + img[:, :, 1], g_at_r, mode='same') * mask[:, :, 0] + \
            convolve2d(img[:, :, 2] + img[:, :, 1], g_at_b, mode='same') * mask[:, :, 2]
    green = np.clip(green, 0, 255)

    mask_help1 = np.ones(raw_img.shape)
    mask_help1[0::2] = mask_help1[0::2] * 0
    mask_help2 = np.ones(raw_img.shape)
    mask_help2[:, 1::2] = mask_help2[:, 1::2] * 0
    red = convolve2d(img[:, :, 1] + img[:, :, 0], r_at_g1, mode='same') * mask[:, :, 1] * mask_help2 + \
          convolve2d(img[:, :, 1] + img[:, :, 0], r_at_g2, mode='same') * mask[:, :, 1] * mask_help1 + \
          convolve2d(img[:, :, 2] + img[:, :, 0], r_at_b, mode='same') * mask[:, :, 2]
    red = np.clip(red, 0, 255)

    mask_help1 = np.ones(raw_img.shape)
    mask_help1[0::2] = mask_help1[0::2] * 0
    mask_help2 = np.ones(raw_img.shape)
    mask_help2[:, 1::2] = mask_help2[:, 1::2] * 0
    blue = convolve2d(img[:, :, 1] + img[:, :, 2], b_at_g1, mode='same') * mask[:,:,1] * mask_help1 + \
           convolve2d(img[:, :, 1] + img[:, :, 2], b_at_g2, mode='same') * mask[:,:,1] * mask_help2 + \
           convolve2d(img[:, :, 0] + img[:, :, 2], b_at_r, mode='same') * mask[:,:,0]
    blue = np.clip(blue, 0, 255)

    return np.array(np.dstack((red, green, blue)) + img, dtype='uint8')


def compute_psnr(img_pred, img_gt):
    c = 3
    h = img_pred.shape[0]
    w = img_pred.shape[1]
    if (len(img_pred.shape) == 2):
        c = 2
    img_pred = np.array(img_pred, dtype='float64')
    img_gt = np.array(img_gt, dtype='float64')
    mse = np.sum((img_pred - img_gt) ** 2) / (c * h * w)
    if mse == 0:
        raise ValueError
    psnr = 10 * np.log10(np.max(img_gt ** 2) / mse)
    return psnr