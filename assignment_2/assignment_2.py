import cv2
import numpy as np
from scipy import ndimage
from scipy import misc
import sys, getopt

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
h3_west_emboss = np.array([[-1, 0, 1],
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
    try:
        print(argv)
        ker = {
            '1':h1_ukn, 
            '2':h2_gauss,
            '3':h3_west_emboss,
            '4':h4_north_emboss,
            '5':h5_high_pass,
            '6':h6_arith_mean,
            '7':h7_right_diag,
            '8':h8_left_diag
            }.get(argv[1])
            
        input_path  = argv[2]

    except:
        print(f'''
        Script Usage:
        \t{argv[0]} <function> <functionargs> <inputfilepath>'

        Make sure the command in syntatically correct.
         - Function argument must match function name
         - Arguments must be separated using commas => arg1,arg2,arg3
         - A existing input path must be given
         ''')
        sys.exit(2)

    print(f"Loading image {input_path} ...")
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    test = ndimage.convolve(img, ker, mode='constant', cval=0.0)
    cv2.imshow("test",test.astype(np.uint8))
    cv2.waitKey()
    

if __name__ == "__main__":
   main(sys.argv)
