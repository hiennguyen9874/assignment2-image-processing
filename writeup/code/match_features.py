threshold = 0.8

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