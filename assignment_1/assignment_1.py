import cv2
import numpy as np
import sys, getopt

# Assuptions: img is square, larger than and multiple of NR
def resolution (img, nr) :
    t = img.shape[0]//nr
    for i in range(nr):
        for j in range(nr):
            img[i*t:(i+1)*t,j*t:(j+1)*t] = img[i*t:(i+1)*t,j*t:(j+1)*t].mean()

    return img

# Assumptions: image has depth of 256 bits
def quantization (img, levels, bits=8) :
    bins_range = list(range(0, (2**bits)+1, (2**bits)//levels))  # list [0,bins,bin_gap]
    bins_map = np.digitize(img, bins_range) - 1                  # Map value to bins
    expand = lambda x : x/(levels-1)*(2**bits-1)                 # Expand to new bit range
    return (expand(bins_map)).astype(np.uint8)                   # Note astype would change

# Assumptions: the values are chosen to not overflow
def grey_scaler (img, choice, params) :
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
    return options[choice](img, *params).astype(np.uint8)



def main(argv):
    try:
        print(argv)
        function = {
            'quantization':quantization, 
            'resolution':resolution,
            'greyscaler':grey_scaler
            }.get(argv[1])

        func_args = [float(x) if '.' in x else int (x) for x in argv[2].split(',')]

        input_path  = argv[3]


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
    func_args = [img]+func_args
    
    if function == grey_scaler :
        print(f"choices => 0-log | 1-exp | 2-square | 3-sqrt | 4-spreader")
        cv2.imshow("Image", function(func_args[0], func_args[1], func_args[2:]))
    else:
        cv2.imshow("Image", function(*func_args))
    cv2.waitKey()

if __name__ == "__main__":
   main(sys.argv)

#### INPUT IMAGE USED ####
# baboon = cv2.imread('../input/baboon.png', cv2.IMREAD_GRAYSCALE)

#### RESOLUTION FUNCTION OUTPUTS ####
# cv2.imshow("256x256 samples", resolution(baboon, 256))
# cv2.imshow("128x128 samples", resolution(baboon, 128))
# cv2.imshow("64x64 samples", resolution(baboon, 64))
# cv2.imshow("32x32 samples", resolution(baboon, 32))
# cv2.imshow("16x16 samples", resolution(baboon, 16))
# cv2.imshow("8x8 samples", resolution(baboon, 8))

#### QUANTIZATION FUNCTION OUTPUTS ####
# cv2.imshow("256 levels", quantization(baboon, 256))
# cv2.imshow("128 levels", quantization(baboon, 128))
# cv2.imshow("64 levels", quantization(baboon, 64))
# cv2.imshow("32 levels", quantization(baboon, 32))
# cv2.imshow("16 levels", quantization(baboon, 16))
# cv2.imshow("8 levels", quantization(baboon, 8))
# cv2.imshow("4 levels", quantization(baboon, 4))
# cv2.imshow("2 levels", quantization(baboon, 2))

#### GRAY_SCALER FUNCTION OUTPUTS ####
# cv2.imshow("log", grey_scaler(baboon, 0, [32]))
# cv2.imshow("exp", grey_scaler(baboon, 1, [20]))
# cv2.imshow("sqr", grey_scaler(baboon, 2, [0.0039]))
# cv2.imshow("srt_root", grey_scaler(baboon, 3, [16]))
# cv2.imshow("spreader", grey_scaler(baboon, 4, [64, 192, 0.5, 1.5, 0.5]))

# cv2.waitKey()