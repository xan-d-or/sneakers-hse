import cv2

def extract_hog_features(image):
    winSize = (64, 64)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    img_resized = cv2.resize(image, winSize)
    features = hog.compute(img_resized).flatten()
    return features