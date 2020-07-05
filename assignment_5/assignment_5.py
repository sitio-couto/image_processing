import sys, argparse
import cv2
import numpy as np

#####################
# Slap code in here #
#####################


#### MAIN FUNCTION - argparsing and filter call ####
def main(args):
    if args.to_pbm: 
        print("Converting image to PBM format...")
        convert_pbm(args.input, args.to_pbm)

    print(f"Loading image {args.input} ...")
    pbm = cv2.imread(args.input, -1)

    print(f"Processing...")

    rgb = cv2.cvtColor(pbm.copy(),cv2.COLOR_GRAY2RGB)
    neg = cv2.bitwise_not(pbm.copy())

    # Check for lines
    lines = classification(
                rgb, neg, 
                kernels  = [(1,100), (200,1), (1,30)], 
                bw_range = (0.5,0.9), 
                tr_range = (0,0.08), 
                color    = (255,0,0),
                tag      = args.tag_lines or args.tag_all
                )

    # Check for words
    words = classification(
                rgb, neg, 
                kernels  = [(1,12), (4,1), (8,12)], 
                bw_range = (0.35,0.95), 
                tr_range = (0,0.2), 
                color    = (0,200,0),
                tag      = args.tag_all or not args.tag_lines
                )

    verbose = args.verbose or args.double_verbose
    if verbose or args.show or not args.output: 
        show(f"Segmentation", rgb)
        # cv2.imwrite('./outputs/font.png', rgb)
    print(f"--------------------")
    print(f"Lines Count: {lines:>5}")
    print(f"Word Count:  {words:>5}")
    print(f"--------------------")

    # Save segmented image as a PBM
    if args.output:
        print(f"Saving to {args.output}...")
        grey = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        grey[grey<250] = 0
        cv2.imwrite(args.output, grey, [cv2.IMWRITE_PXM_BINARY])

    print("Done.")

#### PROCESS COMMAND LINE INPUT #####
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='''Segment text in a given image.'''
        )
    parser.add_argument('input', help='Path of the input image.')
    parser.add_argument('-b','--to-pbm', type=int, metavar="[0-255]",
                        help='Convert image to PBM given a threshold and save to path.'
                        )
    parser.add_argument('-s','--show', action='store_true',
                        help='Exhibit final segmented text image.'
                        )
    parser.add_argument('-d','--debug', action='store_true',
                        help='Show which blocks do not match lines or words.'
                        )
    parser.add_argument('-v','--verbose', action='store_true',
                        help='Exhibit images with the connected components.'
                        )
    parser.add_argument('-vv','--double-verbose', action='store_true',
                        help='Exhibit images for each processing step.'
                        )
    parser.add_argument('-o','--output', type=str,
                        help='Saves PBM file to output path.'
                        )
    parser.add_argument('-l','--tag-lines', action='store_true',
                        help='Draw bounding boxes for each line on the image.'
                        )
    parser.add_argument('-a','--tag-all', action='store_true',
                        help='Draw bounding boxes for each word on the image.'
                        )
    args = parser.parse_args()

    if args.to_pbm and args.to_pbm not in range(0,256):
        assert False, "Select a threshold in range [0-255] for PBM conversion"
    
    main(args)
