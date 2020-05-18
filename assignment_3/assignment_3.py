import sys, argparse
import cv2
import numpy as np
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
        return fft

    cp = fft.copy()
    cp[np.logical_not(filt)] = 0+0j
    return cp

def mag(fft):
    img = np.log(1+np.abs(fft))
    return img.astype(np.uint8)

def pipeline(img, ker, r1, r2):
    '''Show entire processing pipeline printing the image at every state.'''
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
    plt.title('Inverse Transform'), plt.xticks([]), plt.yticks([])
    return plt
 
def process(img, ker, r1, r2):
    '''Process and return image.'''
    transformed = np.fft.fft2(img)                  
    centralized = np.fft.fftshift(transformed)     
    filtered = filter(centralized, ker, r1, r2)     
    decentralized = np.fft.ifftshift(filtered)     
    reverted = np.abs(np.fft.ifft2(decentralized)) 
    return reverted.astype(np.uint8)

def compress(img, percent):
    '''Compress and return image.'''
    transformed = np.fft.fft2(img)                  
    centralized = np.fft.fftshift(transformed)
    
    # Compress image
    spec = mag(centralized)
    base = spec.min()
    maxval = spec.max()
    tresh = base+(maxval-base)*(percent/100)
    compressed = centralized.copy()
    print(f"Compressing {len(compressed[spec < tresh])} values...")
    compressed[spec < tresh] = 0+0j

    decentralized = np.fft.ifftshift(compressed)     
    reverted = np.abs(np.fft.ifft2(decentralized)) 
    return reverted.astype(np.uint8)

# https://medium.com/@hicraigchen/digital-image-processing-using-fourier-transform-in-python-bcb49424fd82

def nonneg_float(radius):
    radius = float(radius)
    if radius < 0:
        raise argparse.ArgumentTypeError("Radius must always be positive.")
    return radius

#### MAIN FUNCTION - argparsing and filter call ####
def main(argv):
    parser = argparse.ArgumentParser(
        description='''Apply selected filter usinf FFT to input image. 
        If no output file is defined, the image is exhibited on the screen.'''
        )
    parser.add_argument('input', help='Path of the input image.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-f','--filter', type=str, choices={'low','band','high'},
                        help='Select a filter (low, band or high-pass) to apply'
                        )
    group.add_argument('-c','--compress', type=nonneg_float,
                        help='Define the threshold for the compression (min-max range percentage)'
                        )
    parser.add_argument('-r1', type=nonneg_float,
                        help='Radius of the filter.'
                        )
    parser.add_argument('-r2', type=nonneg_float,
                        help='External radius for band pass filter.'
                        )
    parser.add_argument('--pipeline', action='store_true',
                        help='Show the image state trough the entire pipeline.'
                        )                    
    parser.add_argument('-o','--output', 
                        help='Path where the output image will be saved.'
                        )

    args = parser.parse_args()

    if args.filter=='band':
        if args.r1==None: raise Exception("Define '-r1' (Inner Radius)")
        if args.r2==None: raise Exception("Define '-r2' (Outer Radius)")
    elif args.filter:
        if args.r1==None: raise Exception("Define '-r1' (Kernel Radius)")

    print(f"Loading image {args.input} ...")
    img = cv2.imread(args.input, 0)

    print(f"Processing...")
    
    # Select which process to run
    if args.compress:
        out = compress(img, args.compress)  
    elif args.pipeline:
        out = pipeline(img, args.filter, args.r1, args.r2)
    else:
        out = process(img, args.filter, args.r1, args.r2)
    
    # Select what to do with the results
    if not args.output:
        if args.pipeline:
            out.show()
        else:
            cv2.imshow("Final Image", out)
            cv2.waitKey()
    else:
        print(f"Saving to {args.output}...")
        if args.pipeline:
            out.plt.savefig(args.output)
        else:
            cv2.imwrite(args.output, out)

    print("Done.")

if __name__ == "__main__":
   main(sys.argv)
