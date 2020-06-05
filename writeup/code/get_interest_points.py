alpha = 0.06
threshold = 0.01
stride = 2
sigma = 0.1
min_distance = 3
sigma0 = 0.1

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