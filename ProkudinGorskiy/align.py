import numpy as np
from skimage.transform import rescale


def mse(im1, im2):
    return np.sum((im1 - im2) ** 2) / (im1.shape[0] * im1.shape[1])


def searchMin(im1, im2, window):
    mse_ = np.inf
    h, w = im1.shape
    for i in np.arange(-window, window + 1):
        for j in np.arange(-window, window + 1):
            t = mse(im1[max(0, i):min(h, h + i),
                        max(0, j):min(w, w + j)],
                    im2[max(0, -i):min(h, h - i),
                        max(0, -j):min(w, w - j)])
            if t < mse_:
                ans = i, j
                mse_ = t
    return ans


def pyramid(im1, im2):
    thr = 500
    h, w = im1.shape
    if h > thr or w > thr:
        h_off, w_off = np.array(pyramid(rescale(im1, 0.5), rescale(im2, 0.5))) * 2
        add_off = searchMin(im1[max(0, h_off):min(h, h + h_off),
                                max(0, w_off):min(w, w + w_off)],
                            im2[max(0, -h_off):min(h, h - h_off),
                                max(0, -w_off):min(w, w - w_off)], 1)
        return np.array(add_off) + np.array((h_off, w_off))
    else:
        return searchMin(im1, im2, 15)


def align(image, coord):
    g_row, g_col = coord
    n = image.shape[0] // 3
    h_cut = int(n * 0.05)
    w_cut = int(image.shape[1] * 0.05)

    image_b = image[h_cut        :n - h_cut,     w_cut:-w_cut]
    image_g = image[h_cut + n    :2 * n - h_cut, w_cut:-w_cut]
    image_r = image[h_cut + 2 * n:3 * n - h_cut, w_cut:-w_cut]
    h, w = image_g.shape

    h1, w1 = pyramid(image_g, image_r)
    h2, w2 = pyramid(image_g, image_b)

    image_g = image_g[max(0, h1, h2)  : min(h, h + h1, h + h2), max(0, w1, w2)  :min(w, w + w1, w + w2)]
    image_r = image_r[max(0, -h1, -h2): min(h, h - h1, h - h2), max(0, -w1, -w2):min(w, w - w1, w - w2)]
    image_b = image_b[max(0, -h1, -h2): min(h, h - h1, h - h2), max(0, -w1, -w2):min(w, w - w1, w - w2)]

    return np.dstack((image_r, image_g, image_b)), \
           (-h2 + g_row - n, -w2 + g_col), \
           (-h1 + g_row + n, -w1 + g_col)
