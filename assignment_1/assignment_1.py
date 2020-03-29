import cv2
import numpy as np

baboon = cv2.imread("../input/baboon.png", cv2.IMREAD_GRAYSCALE)

# Assumptions: image has depth of 256 bits
def quantization (img, levels, bits) :
    bins_range = list(range(0, (2**bits)+1, (2**bits)//levels))  # list [0,bins,bin_gap]
    bins_map = np.digitize(img, bins_range) - 1                  # Map value to bins
    expand = lambda x : x/(levels-1)*(2**bits-1)                 # Expand to new bit range
    return (expand(bins_map)).astype(np.uint8)                   # Note astype would change

# Assuptions: img is square, larger than and multiple of NR
def resolution (img, nr) :
    t = img.shape[0]//nr
    for i in range(nr):
        for j in range(nr):
            img[i*t:(i+1)*t,j*t:(j+1)*t] = img[i*t:(i+1)*t,j*t:(j+1)*t].mean()

    return img

# Assumptions: the values are chosen to not overflow
def grey_scaler (choice, params) :
    f_log = lambda X, c : c*np.log2(X+1)
    f_exp = lambda X, c : c*np.exp(X.astype(np.float64)/100)
    f_sqr = lambda X, c : c*X.astype(np.float64)**2
    f_srt = lambda X, c : c*np.sqrt(X)

    def spreader (X, a, b, alpha, beta, gamma) :
        a_map = X<=a
        b_map = X>b
        mid = np.logical_and(a_map, b_map)
        X[a_map] = alpha*X[a_map]
        X[mid] = beta*(X[mid]-alpha) + alpha*a
        X[b_map] = gamma*(X[b_map]-b) + beta*(b-a) + alpha*a
        return X
    
    # choices [  0  ,   1  ,   2  ,   3  ,    4    ]
    options = [f_log, f_exp, f_sqr, f_srt, spreader]
    return options[choice](*params).astype(np.uint8)

#### RESOLUTION FUNCTION OUTPUTS ####
# cv2.imshow("256x256 samples", resolution(baboon, 256))
# cv2.imshow("128x128 samples", resolution(baboon, 128))
# cv2.imshow("64x64 samples", resolution(baboon, 64))
# cv2.imshow("32x32 samples", resolution(baboon, 32))
# cv2.imshow("16x16 samples", resolution(baboon, 16))
# cv2.imshow("8x8 samples", resolution(baboon, 8))

#### QUANTIZATION FUNCTION OUTPUTS ####
# cv2.imshow("256 levels", quantization(baboon, 256, 8))
# cv2.imshow("128 levels", quantization(baboon, 128, 8))
# cv2.imshow("64 levels", quantization(baboon, 64, 8))
# cv2.imshow("32 levels", quantization(baboon, 32, 8))
# cv2.imshow("16 levels", quantization(baboon, 16, 8))
# cv2.imshow("8 levels", quantization(baboon, 8, 8))
# cv2.imshow("4 levels", quantization(baboon, 4, 8))
# cv2.imshow("2 levels", quantization(baboon, 2, 8))

#### GRAY_SCALER FUNCTION OUTPUTS ####
# cv2.imshow("log", grey_scaler(0, [baboon, 32]))
# cv2.imshow("exp", grey_scaler(1, [baboon, 20]))
# cv2.imshow("sqr", grey_scaler(2, [baboon, 0.0039]))
# cv2.imshow("srt_root", grey_scaler(3, [baboon, 16]))
# cv2.imshow("spreader", grey_scaler(4, [baboon, 64, 192, 0.5, 1.5, 0.5]))

cv2.waitKey()
