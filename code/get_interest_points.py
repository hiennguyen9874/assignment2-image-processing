# Method 1: Sobel
    # Sobel operator kernels
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    start_time = time.time()
    I_x1 = cv.filter2D(image, -1, sobel_kernel_x)
    I_y1 = cv.filter2D(image, -1, sobel_kernel_y)

    I_x1 = filters.gaussian(I_x1, sigma)
    I_y1 = filters.gaussian(I_y1, sigma)
    print("---method 1: Sobel %s seconds ---" % (time.time() - start_time))

    # Method 2: Sobel
    start_time = time.time()
    I_x2 = cv.Sobel(image, cv.CV_8U, 1, 0, ksize=3)
    I_y2 = cv.Sobel(image, cv.CV_8U, 0, 1, ksize=3)

    I_x2 = filters.gaussian(I_x2, sigma)
    I_y2 = filters.gaussian(I_y2, sigma)
    print("---method 2: Sobel %s seconds ---" % (time.time() - start_time))

    # Method 3: Scharr
    start_time = time.time()
    I_x3 = np.abs(cv.Scharr(image, cv.CV_32F, 1, 0))
    I_y3 = np.abs(cv.Scharr(image, cv.CV_32F, 0, 1))

    I_x3 = filters.gaussian(I_x3, sigma)
    I_y3 = filters.gaussian(I_y3, sigma)
    print("---method 3: Scharr %s seconds ---" % (time.time() - start_time))

    # Method 4: Prewitt
    prewitt_kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    prewitt_kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    start_time = time.time()
    I_x4 = cv.filter2D(image, -1, prewitt_kernel_x)
    I_y4 = cv.filter2D(image, -1, prewitt_kernel_y)

    I_x4 = filters.gaussian(I_x4, sigma)
    I_y4 = filters.gaussian(I_y4, sigma)
    print("---method 4: Prewitt %s seconds ---" % (time.time() - start_time))

    # method 5: Roberts
    roberts_kernel_x = np.array([[1, 0], [0, -1]])
    roberts_kernel_y = np.array([[0, -1], [1, 0]])
    start_time = time.time()
    I_x5 = cv.filter2D(image, -1, roberts_kernel_x)
    I_y5 = cv.filter2D(image, -1, roberts_kernel_y)

    I_x5 = filters.gaussian(I_x5, sigma)
    I_y5 = filters.gaussian(I_y5, sigma)
    print("---method 5: Roberts %s seconds ---" % (time.time() - start_time))