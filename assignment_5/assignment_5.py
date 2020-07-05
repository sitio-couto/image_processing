import sys, argparse, cv2
import numpy as np

#####################
# Slap code in here #
#####################


#### MAIN FUNCTION - argparsing and filter call ####
def main(args):

    print(f"Loading Image A: {args.image_A}")
    imgA = cv2.imread(args.image_A, -1)
    print(f"Loading Image B: {args.image_B}")
    imgB = cv2.imread(args.image_B, -1)

    print(f"Processing...")
    # Finding interest points
    desc = {
        'sift':cv2.xfeatures2d.SIFT_create(),
        'surf':cv2.xfeatures2d.SURF_create(),
        'orb':cv2.ORB_create()}.get(args.descriptor)
    kp1, des1 = desc.detectAndCompute(imgA,None)
    kp2, des2 = desc.detectAndCompute(imgB,None)
    cv2.imshow("A", imgA)
    cv2.imshow('original_image_left_keypoints',cv2.drawKeypoints(imgA,kp1,None))
    cv2.waitKey()
    exit()

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
        description='''Join two images.'''
        )
    parser.add_argument('image_A',
                        help='Path to the first part of the image to be stiched.')
    parser.add_argument('image_B',
                        help='Path to the second part of the image to be stiched.')
    parser.add_argument('descriptor', type=str, choices=['sift','surf','orb'],
                        help='Select descriptor to match the images features.'
                        )
    # parser.add_argument('-b','--to-pbm', type=int, metavar="[0-255]",
    #                     help='Convert image to PBM given a threshold and save to path.'
    #                     )
    # parser.add_argument('-s','--show', action='store_true',
    #                     help='Exhibit final segmented text image.'
    #                     )
    parser.add_argument('-v','--verbose', action='store_true',
                        help='Exhibit images with the connected components.'
                        )
    # parser.add_argument('-vv','--double-verbose', action='store_true',
    #                     help='Exhibit images for each processing step.'
    #                     )
    # parser.add_argument('-o','--output', type=str,
    #                     help='Saves PBM file to output path.'
    #                     )
    # parser.add_argument('-l','--tag-lines', action='store_true',
    #                     help='Draw bounding boxes for each line on the image.'
    #                     )
    # parser.add_argument('-a','--tag-all', action='store_true',
    #                     help='Draw bounding boxes for each word on the image.'
    #                     )
    args = parser.parse_args()
   
    main(args)
