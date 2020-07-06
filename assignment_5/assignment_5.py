import sys, argparse, cv2
import numpy as np

def trim(frame):
    #crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop top
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    #crop top
    if not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    #crop top
    if not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame

#### MAIN FUNCTION - argparsing and filter call ####
def main(args):

    print(f"Loading Image A: {args.image_A}")
    imgA = cv2.imread(args.image_A)
    # imgA = cv2.resize(imgA, (0,0), fx=1, fy=1)
    imgAG = cv2.cvtColor(imgA,cv2.COLOR_BGR2GRAY)

    print(f"Loading Image B: {args.image_B}")
    imgB = cv2.imread(args.image_B)
    # imgB = cv2.resize(imgB, (0,0), fx=1, fy=1)
    imgBG = cv2.cvtColor(imgB,cv2.COLOR_BGR2GRAY)

    print(f"Processing...")

    # Picking Feature Extractor
    desc = {
        'sift':cv2.xfeatures2d.SIFT_create(),
        'surf':cv2.xfeatures2d.SURF_create(),
        'orb':cv2.ORB_create()
        }.get(args.descriptor)
    
    # find the key points and descriptors with SIFT
    kp1, des1 = desc.detectAndCompute(imgAG,None)
    kp2, des2 = desc.detectAndCompute(imgBG,None)
    
    # Catch features that match the most
    match = cv2.BFMatcher()
    matches = match.knnMatch(des1,des2,k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.1*n.distance:
            good.append(m)
    good = np.asarray(good)

    draw_params = dict( matchColor = (0,255,0), # draw matches in green color
                        singlePointColor = None,
                        flags = 2)

    links = cv2.drawMatches(imgA,kp1,imgB,kp2,good,None,**draw_params)
    
    if args.verbose:
        cv2.imshow("Linked Features", links)
        cv2.waitKey()
    
    MIN_MATCH = 10
    if len(good) > MIN_MATCH:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h,w = imgAG.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts, M)
        img = cv2.polylines(imgBG,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        if args.verbose:
            cv2.imshow("Sliced and Rotated", img)
            cv2.waitKey()
    else:
        raise Exception(f'Too Few Mathches (got/need: {len(good)}/{MIN_MATCH})')

    dst = cv2.warpPerspective(imgA,M,(imgB.shape[1] + imgA.shape[1], imgB.shape[0]))
    dst[0:imgB.shape[0],0:imgB.shape[1]] = imgB

    if args.verbose:
        cv2.imshow("Joined and Colored", dst)
        cv2.imshow("Joined and Colored", imgB)
        cv2.waitKey()
    
    cv2.imshow("original_image_stitched_crop.jpg", trim(dst))
    cv2.waitKey()

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
