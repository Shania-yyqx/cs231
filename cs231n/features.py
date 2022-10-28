from __future__ import print_function
from builtins import zip
from builtins import range
# from past.builtins import xrange

import matplotlib
import numpy as np
from scipy.ndimage import uniform_filter


def extract_features(imgs, feature_fns, verbose=False):
    """
    Given pixel data for images and several feature functions that can operate on
    single images, apply all feature functions to all images, concatenating the
    feature vectors for each image and storing the features for all images in
    a single matrix.

    Inputs:
    - imgs: N x H X W X C array of pixel data for N images.
    - feature_fns: List of k feature functions. The ith feature function should
      take as input an H x W x D array and return a (one-dimensional) array of
      length F_i.
    - verbose: Boolean; if true, print progress.

    Returns:
    An array of shape (N, F_1 + ... + F_k) where each column is the concatenation
    of all features for a single image.
    """
    num_images = imgs.shape[0]
    if num_images == 0:
        return np.array([])

    # Use the first image to determine feature dimensions
    # 
    feature_dims = []  
    first_image_features = []
    # feature_fns = hog_feature, lambda img: color_histogram_hsv(img, nbin=num_color_bins)
    for feature_fn in feature_fns:
        feats = feature_fn(imgs[0].squeeze())
        assert len(feats.shape) == 1, "Feature functions must be one-dimensional"
        feature_dims.append(feats.size)
        first_image_features.append(feats)

    # Now that we know the dimensions of the features, we can allocate a single
    # big array to store all features as columns.
    # 提取出来的总的特征向量维度 直方图+hsv
    total_feature_dim = sum(feature_dims)
    imgs_features = np.zeros((num_images, total_feature_dim))
    # 元素数组按水平方向进行组合 原来是【1，2】【3，4】，变成【1，2，3，4】
    # 把features储存成列向量
    imgs_features[0] = np.hstack(first_image_features).T 

    # Extract features for the rest of the images.
    for i in range(1, num_images):
        idx = 0
        for feature_fn, feature_dim in zip(feature_fns, feature_dims):
            next_idx = idx + feature_dim   #第一次执行的时候dim=直方图的维度
            imgs_features[i, idx:next_idx] = feature_fn(imgs[i].squeeze())
            idx = next_idx   #第2次执行的时候hsv特征的idx就从上一次的dim后面开始加上去
        if verbose and i % 1000 == 999:
            print("Done extracting features for %d / %d images" % (i + 1, num_images))

    return imgs_features


def rgb2gray(rgb):
    """Convert RGB image to grayscale

      Parameters:
        rgb : RGB image

      Returns:
        gray : grayscale image

    """
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])


# 提取一幅图像的hog特征
def hog_feature(im):
    """Compute Histogram of Gradient (HOG) feature for an image

         Modified from skimage.feature.hog
         http://pydoc.net/Python/scikits-image/0.4.2/skimage.feature.hog

       Reference:
         Histograms of Oriented Gradients for Human Detection
         Navneet Dalal and Bill Triggs, CVPR 2005

      Parameters:
        im : an input grayscale or rgb image

      Returns:
        feat: Histogram of Gradient (HOG) feature

    """

    # convert rgb to grayscale if needed
    if im.ndim == 3:
        image = rgb2gray(im)
    else:
        image = np.at_least_2d(im)

    sx, sy = image.shape  # image size
    orientations = 9  # number of gradient bins  每个细胞直方图通道个数
    cx, cy = (8, 8)  # pixels per cell  每个细胞像素个数

    # 
    gx = np.zeros(image.shape)
    gy = np.zeros(image.shape)
    #  :-1 除了最后一个取全部
    # 计算x方向的梯度（进行一次行方向上的差分）
    gx[:, :-1] = np.diff(image, n=1, axis=1)  # compute gradient on x-direction
    # 计算y方向的梯度（进行一次列方向上的差分）
    gy[:-1, :] = np.diff(image, n=1, axis=0)  # compute gradient on y-direction
    # 梯度幅值
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)  # gradient magnitude
    # 梯度方向
    grad_ori = np.arctan2(gy, (gx + 1e-15)) * (180 / np.pi) + 90  # gradient orientation

    #一个区间的细胞个数 
    n_cellsx = int(np.floor(sx / cx))  # number of cells in x
    n_cellsy = int(np.floor(sy / cy))  # number of cells in y 
    # compute orientations integral images 为每个细胞单元构建直方图
    orientation_histogram = np.zeros((n_cellsx, n_cellsy, orientations))
    for i in range(orientations):  #每个细胞直方图有9个通道
        # create new integral image for this orientation
        # isolate orientations in this range
        temp_ori = np.where(grad_ori < 180 / orientations * (i + 1), grad_ori, 0)
        temp_ori = np.where(grad_ori >= 180 / orientations * i, temp_ori, 0)
        # select magnitudes for those orientations
        cond2 = temp_ori > 0
        #吧细胞单元组合成大的快，块内归一化梯度直方图
        temp_mag = np.where(cond2, grad_mag, 0)
        orientation_histogram[:, :, i] = uniform_filter(temp_mag, size=(cx, cy))[
            round(cx / 2) :: cx, round(cy / 2) :: cy
        ].T

    return orientation_histogram.ravel()


def color_histogram_hsv(im, nbin=10, xmin=0, xmax=255, normalized=True):
    """
    Compute color histogram for an image using hue.

    Inputs:
    - im: H x W x C array of pixel data for an RGB image.
    - nbin: Number of histogram bins. (default: 10)
    - xmin: Minimum pixel value (default: 0)
    - xmax: Maximum pixel value (default: 255)
    - normalized: Whether to normalize the histogram (default: True)

    Returns:
      1D vector of length nbin giving the color histogram over the hue of the
      input image.
    """
    ndim = im.ndim
    bins = np.linspace(xmin, xmax, nbin + 1)
    hsv = matplotlib.colors.rgb_to_hsv(im / xmax) * xmax
    imhist, bin_edges = np.histogram(hsv[:, :, 0], bins=bins, density=normalized)
    imhist = imhist * np.diff(bin_edges)

    # return histogram
    return imhist


# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

pass

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
