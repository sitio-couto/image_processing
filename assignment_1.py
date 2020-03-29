import cv2
import numpy as np 

baboon = cv2.imread("input/baboon.png")
baboon = cv2.cvtColor(baboon, cv2.COLOR_BGR2GRAY) 

def quantization (img, levels, bits) :
    bins_range = list(range(0, (2**bits)+1, (2**bits)//levels))  # list [0,bins,bin_gap]
    bins_map = np.digitize(img, bins_range) - 1                  # Map value to bins
    expand = lambda x : x/(levels-1)*(2**bits-1)                 # Expand to new bit range
    return (expand(bins_map)).astype(np.uint8)                   # Note astype would change

def quantization_2 (img, levels) :
    shift = np.ceil(np.log2(img.max())) - np.ceil(np.log2(levels))
    print(shift)
    if shift <= 0 : return img
    img = img//(2**shift)
    print((img*2**shift).astype(np.uint8).max())
    return (img*2**shift).astype(np.uint8)

cv2.imshow("256 levels", quantization(baboon, 256, 8))
cv2.imshow("128 levels", quantization(baboon, 128, 8))
cv2.imshow("64 levels", quantization(baboon, 64, 8))
cv2.imshow("32 levels", quantization(baboon, 32, 8))
cv2.imshow("16 levels", quantization(baboon, 16, 8))
cv2.imshow("8 levels", quantization(baboon, 8, 8))
cv2.imshow("4 levels", quantization(baboon, 4, 8))
cv2.imshow("2 levels", quantization(baboon, 2, 8))
cv2.waitKey()

def resolution (img, nr) :
    if nr > img.shape[0]//2 : return img

    t = img.shape[0]//nr
    for i in range(nr):
        for j in range(nr):
            img[i*t:(i+1)*t,j*t:(j+1)*t] = img[i*t:(i+1)*t,j*t:(j+1)*t].mean()

    return img

# cv2.imshow("256 levels", resolution(baboon, 512))
# cv2.imshow("128 levels", resolution(baboon, 128))
# cv2.imshow("64 levels", resolution(baboon, 64))
# cv2.imshow("32 levels", resolution(baboon, 32))
# cv2.imshow("16 levels", resolution(baboon, 16))
# cv2.imshow("8 levels", resolution(baboon, 8))


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
# cv2.imshow("log", grey_scaler(1, [baboon, 0.01]))
# cv2.imshow("64 levels", resolution(baboon, 64))
# cv2.imshow("32 levels", resolution(baboon, 32))
# cv2.imshow("16 levels", resolution(baboon, 16))
cv2.waitKey()
