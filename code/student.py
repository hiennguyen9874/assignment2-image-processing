import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, feature, img_as_int
from skimage.measure import regionprops


def get_interest_points(image, feature_width):
    '''
    Returns interest points for the input image

    (Please note that we recommend implementing this function last and using cheat_interest_points()
    to test your implementation of get_features() and match_features())

    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You do not need to worry about scale invariance or keypoint orientation estimation
    for your Harris corner detector.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious (false/fake) interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.feature.peak_local_max (experiment with different min_distance values to get good results)
        - skimage.measure.regionprops


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :feature_width:

    :returns:
    :xs: an np array of the x coordinates of the interest points in the image
    :ys: an np array of the y coordinates of the interest points in the image

    :optional returns (may be useful for extra credit portions):
    :confidences: an np array indicating the confidence (strength) of each interest point
    :scale: an np array indicating the scale of each interest point
    :orientation: an np array indicating the orientation of each interest point
    '''
    alpha = 0.06
    threshold = 0.01
    stride = 2
    sigma = 0.1
    min_distance = 3
    sigma0 = 0.1

    # print(f'alpha: {alpha}, threshold: {threshold}, stride: {stride}, sigma: {sigma}, min_distance: {min_distance}')

    #step1: blur image (optional)
    filtered_image = filters.gaussian(image, sigma=sigma)

    # step2: calculate gradient of image
    I_x = filters.sobel_v(filtered_image)
    I_y = filters.sobel_h(filtered_image)

    # step3: calculate Gxx, Gxy, Gyy
    I_xx = np.square(I_x)
    I_xy = np.multiply(I_x, I_y)
    I_yy = np.square(I_y)

    I_xx = filters.gaussian(I_xx, sigma=sigma0)
    I_xy = filters.gaussian(I_xy, sigma=sigma0)
    I_yy = filters.gaussian(I_yy, sigma=sigma0)

    listC = np.zeros_like(image)

    # step4: caculate C matrix
    for y in range(0, image.shape[0]-feature_width, stride):
        for x in range(0, image.shape[1]-feature_width, stride):
            # matrix 17x17
            Sxx = np.sum(I_xx[y:y+feature_width+1, x:x+feature_width+1])
            Syy = np.sum(I_yy[y:y+feature_width+1, x:x+feature_width+1])
            Sxy = np.sum(I_xy[y:y+feature_width+1, x:x+feature_width+1])

            detC = (Sxx * Syy) - (Sxy**2)
            traceC = Sxx + Syy
            C = detC - alpha*(traceC**2)
            
            if C > threshold:
                listC[y+feature_width//2, x+feature_width//2] = C

    # step5: using non-maximal suppression
    ret = feature.peak_local_max(listC, min_distance=min_distance, threshold_abs=threshold)
    return ret[:, 1], ret[:, 0]


def get_features(image, x, y, feature_width):
    '''
    Returns feature descriptors for a given set of interest points.

    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT-like descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width / 4 pixels square.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.filters (library)

    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates of interest points
    :y: np array of y coordinates of interest points
    :feature_width: in pixels, is the local feature width. You can assume
                    that feature_width will be a multiple of 4 (i.e. every cell of your
                    local SIFT-like feature will have an integer width and height).
    If you want to detect and describe features at multiple scales or
    particular orientations you can add input arguments.

    :returns:
    :features: np array of computed features. It should be of size
            [len(x) * feature dimensionality] (for standard SIFT feature
            dimensionality is 128)

    '''
    x = np.round(x).astype(int)
    y = np.round(y).astype(int)

    sigma_gradient_image = 0.1
    sigma_16x16 = 0.4
    threshold = 0.2

    # print(f'sigma_gradient_image: {sigma_gradient_image}, sigma_16x16: {sigma_16x16}, threshold: {threshold}')

    features = np.zeros((len(x), 4, 4, 8))
    
    # step0: blur image (optional)
    filtered_image = filters.gaussian(image, sigma=sigma_gradient_image)
    
    # step1: compute the gradient of image
    d_im_x = filters.sobel_v(filtered_image)
    d_im_y = filters.sobel_h(filtered_image)

    magnitude_gradient = np.sqrt(np.add(np.square(d_im_x), np.square(d_im_y)))
    direction_gradient = np.arctan2(d_im_y, d_im_x)
    direction_gradient[direction_gradient < 0] += 2 * np.pi

    # step2:
    # image.shape[0] = 1024
    # image.shape[1] = 768
    # x (0 -> 768)
    # y (0 -> 1024)
    for n, (x_, y_) in enumerate(zip(x, y)):
        # get windows of key point(x, y)
        rows = (y_ - feature_width//2, y_ + feature_width//2 + 1)
        cols = (x_ - feature_width//2, x_ + feature_width//2 + 1)

        if rows[0] < 0:
            rows = (0, feature_width+1)
        if rows[1] > image.shape[0]:
            rows = (image.shape[0]-feature_width-1, image.shape[0]-1)

        if cols[0] < 0:
            cols = (0, feature_width+1)
        if cols[1] > image.shape[1]:
            cols = (image.shape[1]-feature_width-1, image.shape[1]-1)

        # get gradient and angle of key point
        magnitude_window = magnitude_gradient[rows[0]:rows[1], cols[0]:cols[1]]
        direction_window = direction_gradient[rows[0]:rows[1], cols[0]:cols[1]]

        # Gaussian filter on window
        magnitude_window = filters.gaussian(
            magnitude_window, sigma=sigma_16x16)
        direction_window = filters.gaussian(
            direction_window, sigma=sigma_16x16)

        for i in range(feature_width//4):
            for j in range(feature_width//4):
                current_magnitude = magnitude_window[i*feature_width//4: (
                    i+1)*feature_width//4, j*feature_width//4:(j+1)*feature_width//4]

                current_direction = direction_window[i*feature_width//4: (
                    i+1)*feature_width//4, j*feature_width//4:(j+1)*feature_width//4]

                features[n, i, j] = np.histogram(current_direction.reshape(
                    -1), bins=8, range=(0, 2*np.pi), weights=current_magnitude.reshape(-1))[0]

    # Extract 8 x 16 values into 128-dim vector
    features = features.reshape((len(x), -1,))

    # Normalize vector to [0...1]
    norm = np.sqrt(np.square(features).sum(axis=1)).reshape(-1, 1)
    features = features / norm

    # Clamp all vector values > 0.2 to 0.2
    features[features >= threshold] = threshold

    # Re-normalize
    norm = np.sqrt(np.square(features).sum(axis=1)).reshape(-1, 1)
    features = features / norm

    return features


def match_features(im1_features, im2_features):
    '''
    Implements the Nearest Neighbor Distance Ratio Test to assign matches between interest points
    in two images.

    Please implement the "Nearest Neighbor Distance Ratio (NNDR) Test" ,
    Equation 4.18 in Section 4.1.3 of Szeliski.

    For extra credit you can implement spatial verification of matches.

    Please assign a confidence, else the evaluation function will not work. Remember that
    the NNDR test will return a number close to 1 for feature points with similar distances.
    Think about how confidence relates to NNDR.

    This function does not need to be symmetric (e.g., it can produce
    different numbers of matches depending on the order of the arguments).

    A match is between a feature in im1_features and a feature in im2_features. We can
    represent this match as a the index of the feature in im1_features and the index
    of the feature in im2_features

    :params:
    :im1_features: an np array of features returned from get_features() for interest points in image1
    :im2_features: an np array of features returned from get_features() for interest points in image2

    :returns:
    :matches: an np array of dimension k x 2 where k is the number of matches. The first
            column is an index into im1_features and the second column is an index into im2_features
    :confidences: an np array with a real valued confidence for each match
    '''
    threshold = 0.8

    # print(f'threshold: {threshold}')

    matches = []
    confidences = []

    for i in range(im1_features.shape[0]):
        distances = np.sqrt(np.square(np.subtract(
            im1_features[i, :], im2_features)).sum(axis=1))
        index_sorted = np.argsort(distances)
        if distances[index_sorted[0]] / distances[index_sorted[1]] < threshold:
            matches.append([i, index_sorted[0]])
            confidences.append(
                1.0 - distances[index_sorted[0]]/distances[index_sorted[1]])
    matches = np.asarray(matches)
    confidences = np.asarray(confidences)
    return matches, confidences
