import sys, argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt

def convert_pbm(path, threshold):
    img = cv2.imread(path, 0)
    if args.verbose:
        cv2.imshow("test",cv2.resize(img, (0,0), fx=.75, fy=.75))
        cv2.waitKey()
    img[img >  threshold] = 255
    img[img <= threshold] = 0
    for i,l in enumerate(reversed(path)):
        if l=='.': 
            out = path[:-(i+1)]+'.pbm'
            break
    if args.verbose:
        cv2.imshow("test",cv2.resize(img, (0,0), fx=.75, fy=.75))
        cv2.waitKey()
    cv2.imwrite(out, img, [cv2.IMWRITE_PXM_BINARY])
    print(f"Image converted and saved to: {out}")
    exit()

#### MAIN FUNCTION - argparsing and filter call ####
def main(args):
    if args.to_pbm: 
        print("Converting image to PBM format...")
        convert_pbm(args.input, args.to_pbm)

    print(f"Loading image {args.input} ...")
    img = cv2.imread(args.input, 0)

    print(f"Processing...")
    cv2.imshow("PBM",cv2.resize(img, (0,0), fx=.75, fy=.75))
    cv2.waitKey()

    # # Select what to do with the results
    # if not args.output:
    #     if args.pipeline:
    #         out.show()
    #     else:
    #         cv2.imshow("Final Image", out)
    #         cv2.waitKey()
    # else:
    #     print(f"Saving to {args.output}...")
    #     if args.pipeline:
    #         out.plt.savefig(args.output)
    #     else:
    #         cv2.imwrite(args.output, out)

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='''Segment text in a given image.'''
        )
    parser.add_argument('input', help='Path of the input image.')
    parser.add_argument('-b','--to-pbm', type=int,metavar="[0-255]",
                        help='Convert image to PBM given a threshold and save to path.'
                        )
    parser.add_argument('-v','--verbose', action='store_true',
                        help='Exhibit images for each processing step.'
                        )
    args = parser.parse_args()

    if args.to_pbm and args.to_pbm not in range(0,256):
        assert False, "Select a threshold in range [0-255] for PBM conversion"
    
    main(args)
