import cv2
import numpy as np
from scipy import ndimage
from scipy import misc
import sys, argparse
from matplotlib import pyplot as plt

def filter(fft, ty, r1, r2):

    w,h = fft.shape
    center = (w//2, h//2)

    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X-center[0])**2 + (Y-center[1])**2)
    
    if ty=='low':
        filt = (dist <= r1)
    elif ty=='band':
        A = (r1 <= dist)
        B = (dist <= r2)
        filt = np.logical_and(A,B)
    elif ty=='high':
        filt = (dist > r1)
    else:
        return None

    cp = fft.copy()
    cp[np.logical_not(filt)] = 0+0j
    return cp

def mag(fft):
    img = np.log(1+np.abs(fft))
    norm_image = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    norm_image = norm_image.astype(np.uint8)
    return norm_image

def process(img, ker, r1, r2):
    transformed = np.fft.fft2(img)                 # Apply FFT on image
    centralized = np.fft.fftshift(transformed)     # Centralize FFT origin
    filtered = filter(centralized, ker, r1, r2)    # Filter image with mask
    decentralized = np.fft.ifftshift(filtered)     # Decentralize filtered FFT
    reverted = np.abs(np.fft.ifft2(decentralized)) # Revert FFT and get intensities

    plt.subplot(231),plt.imshow(img, 'gray')
    plt.title('Read'), plt.xticks([]), plt.yticks([])
    plt.subplot(232),plt.imshow(mag(transformed), 'gray')
    plt.title('Transform'), plt.xticks([]), plt.yticks([])
    plt.subplot(233),plt.imshow(mag(centralized), 'gray')
    plt.title('Centralize'), plt.xticks([]), plt.yticks([])
    plt.subplot(234),plt.imshow(mag(filtered), 'gray')
    plt.title('Filter'), plt.xticks([]), plt.yticks([])
    plt.subplot(235),plt.imshow(mag(decentralized), 'gray')
    plt.title('Decentralize'), plt.xticks([]), plt.yticks([])
    plt.subplot(236),plt.imshow(reverted, 'gray')
    plt.title('Revert FFT'), plt.xticks([]), plt.yticks([])
    plt.show()
 

# https://medium.com/@hicraigchen/digital-image-processing-using-fourier-transform-in-python-bcb49424fd82

def check_radius(radius):
    radius = float(radius)
    if radius <= 0:
        raise argparse.ArgumentTypeError("Radius must always be positive.")
    return radius

#### MAIN FUNCTION - argparsing and filter call ####
def main(argv):
    parser = argparse.ArgumentParser(
        description='''Apply selected filter to input image. 
        If more than one defined, sums their individual outputs.'''
        )
    parser.add_argument('input', help='Path of the input image.')
    parser.add_argument('-f','--filter', type=str, choices={'low','band','high'},
                        help='Select a filter (low, band and high-pass) to apply'
                        )
    parser.add_argument('-r1', type=check_radius,
                        help='Radius of the filter.'
                        )
    parser.add_argument('-r2', type=check_radius,
                        help='External radius for band pass filter.'
                        )
    parser.add_argument('-o','--output', 
                        help='Path where the output image will be saved.'
                        )

    args = parser.parse_args()

    if args.filter=='band':
        if not args.r1: raise Exception("Define '-r1' (Inner Radius)")
        if not args.r2: raise Exception("Define '-r2' (Outer Radius)")
    elif args.filter:
        if not args.r1: raise Exception("Define '-r1' (Kernel Radius)")

    print(f"Loading image {args.input} ...")
    img = cv2.imread(args.input, 0)

    print(f"Processing...")
    out = process(img, args.filter, args.r1, args.r2)

    # if not args.output:
    #     cv2.imshow("test", out)
    #     cv2.waitKey()
    # else:
    #     cv2.imwrite(args.output, out)

    

if __name__ == "__main__":
   main(sys.argv)
