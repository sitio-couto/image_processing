import cv2
import numpy as np 

baboon = cv2.imread("input/baboon.png")

def quantization (img, bins) :
    gap = 256//bins
    bins_range = [gap*i for i in range(bins+1)]
    bins_map = (np.digitize(img, bins_range)-1)/(bins-1)
    return (bins_map*255).astype(np.uint8)

# cv2.imshow("256 levels", quantization(baboon, 256))
# cv2.imshow("64 levels", quantization(baboon, 64))
# cv2.imshow("32 levels", quantization(baboon, 32))
# cv2.imshow("16 levels", quantization(baboon, 16))
# cv2.imshow("8 levels", quantization(baboon, 8))
# cv2.imshow("4 levels", quantization(baboon, 4))
# cv2.imshow("2 levels", quantization(baboon, 2))
# cv2.waitKey()

# def resolution (img, new_res) :
#     step = img.shape[0]//new_res
#     for i in range()
#     return np.resize(img[::step, ::step], (img.shape[0],img.shape[0]))

# cv2.imshow("256 levels", resolution(baboon, 512))
# cv2.imshow("128 levels", resolution(baboon, 128))
# cv2.imshow("64 levels", resolution(baboon, 64))
# cv2.imshow("32 levels", resolution(baboon, 32))
# cv2.imshow("16 levels", resolution(baboon, 16))
# cv2.waitKey()


def grey_scaler (choice, params) :
    f_log = lambda X, c : c*np.log(X+1)
    f_exp = lambda X, c : c*np.exp(X)
    f_sqr = lambda X, c : c*X**2
    f_srt = lambda X, c : c*np.sqrt(X)

    def spreader (X, a, b, alpha, beta, gamma) :
        X[X<=a] = alpha*X[X<=a]
        X[a<X<=b] = beta*(X[a<X<=b]-alpha) + alpha*a
        X[b<X] = gamma*(X[b<X]-b) + beta*(b-a) + alpha*a

    options = [f_log, f_exp, f_sqr, f_srt, spreader]
    return options[choice](*params).astype(np.uint8)

# cv2.imshow("log", grey_scaler(0, [baboon, 1000]))
cv2.imshow("log", grey_scaler(1, [baboon, 0.01]))
# cv2.imshow("64 levels", resolution(baboon, 64))
# cv2.imshow("32 levels", resolution(baboon, 32))
# cv2.imshow("16 levels", resolution(baboon, 16))
cv2.waitKey()
