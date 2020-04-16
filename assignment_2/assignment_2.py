import cv2
import numpy as np
from scipy import ndimage
from scipy import misc
import sys, argparse

# h1 - ?????????????
h1_ukn = np.array([[ 0,  0, -1,  0,  0],
                     [ 0, -1, -2, -1,  0],
                     [-1, -2, 16, -2, -1],
                     [ 0, -1, -2, -1,  0],
                     [ 0,  0, -1,  0,  0]])

# h2 - gaussian filter com sigma=1.0
h2_gauss = np.array([[1,  4,  6,  4, 1],
                     [4, 16, 24, 16, 4],
                     [6, 24, 36, 24, 6],
                     [4, 16, 24, 16, 4],
                     [1,  4,  6,  4, 1]])/256

# h3 - gradient detection (embossing) WEST
h3_east_emboss = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]])

# h4 - gradient detection (embossing) NORTH
h4_north_emboss = np.array([[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]])

# h5 - filtro passa altas
h5_high_pass = np.array([[-1, -1, -1],
                         [-1,  8, -1],
                         [-1, -1, -1]])

# h6 - arithimetic mean
h6_arith_mean = np.array([[1, 1, 1],
                          [1, 1, 1],
                          [1, 1, 1]])/9

# h7 - right diagonal edges detection
h7_right_diag = np.array([[-1, -1,  2],
                          [-1,  2, -1],
                          [ 2, -1, -1]])

# h8 - left diagonal edges detection
h8_left_diag = np.array([[ 2, -1, -1],
                         [-1,  2, -1],
                         [-1, -1,  2]])

# img = cv2.imread('../input/butterfly.png', cv2.IMREAD_GRAYSCALE)
# test = ndimage.convolve(img, h3_west_emboss, mode='constant', cval=0.0)
# cv2.imshow("test",test.astype(np.uint8))
# cv2.waitKey()

def main(argv):
    parser = argparse.ArgumentParser(
        description='''Apply selected filter to input image. 
        If more than one defined, sums their individual outputs.'''
        )
    parser.add_argument('input', help='Path of the input image.')
    parser.add_argument('ker', type=int, choices=list(range(1,9)),
                        help='ID (x) of the kernel (Hx) to be used.'
                        )
    parser.add_argument('--sum', type=int, choices=list(range(1,9)), 
                        help='Takes a second filter and sum the results with: sqrt(ker**2 + pit**2)'
                        )
    parser.add_argument('-o','--output', 
                        help='Path where the output image will be saved.'
                        )

    args = parser.parse_args()

    kernels = {
        1:h1_ukn,
        2:h2_gauss,
        3:h3_east_emboss,
        4:h4_north_emboss,
        5:h5_high_pass,
        6:h6_arith_mean,
        7:h7_right_diag,
        8:h8_left_diag
    }

    print(f"Loading image {args.input} ...")
    img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    ker = cv2.flip(kernels[args.ker], -1)
    out = cv2.filter2D(img, -1, ker)
    
    print(out)

    if args.sum : 
        ker = cv2.flip(kernels[args.sum], -1)
        out1 = out.astype(np.float32)
        out2 = cv2.filter2D(img, -1, ker).astype(np.float32)
        out = np.sqrt(out1**2 + out2**2)
        out[out>255] = 255
        out = out.astype(np.uint8)
 
    if not args.output:
        cv2.imshow("test", out)
        cv2.waitKey()
    else:
        cv2.imwrite(args.output, out)
    

if __name__ == "__main__":
   main(sys.argv)
