import sys, argparse, cv2
import numpy as np


#### MAIN FUNCTION - argparsing and filter call ####
def main(args):

    print(f"Loading Image A: {args.image_A}")
    imgA = cv2.imread(args.image_A)
    imgAG = cv2.cvtColor(imgA,cv2.COLOR_BGR2GRAY)

    print(f"Loading Image B: {args.image_B}")
    imgB = cv2.imread(args.image_B)
    imgBG = cv2.cvtColor(imgB,cv2.COLOR_BGR2GRAY)

    print(f"Processing...")

    # Picking Feature Extractor
    desc = {
        'sift':cv2.xfeatures2d.SIFT_create(),
        'surf':cv2.xfeatures2d.SURF_create(),
        'brief':cv2.xfeatures2d.BRIEF_create(),
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
        if m.distance < args.threshold*n.distance:
            good.append(m)
    good = np.asarray(good)

    draw_params = dict( matchColor = (0,255,0), # draw matches in green color
                        singlePointColor = None,
                        flags = 2)

    links = cv2.drawMatches(imgA,kp1,imgB,kp2,good,None,**draw_params)
    
    if args.verbose:
        cv2.imshow("Linked Features", links)
        cv2.waitKey()

    # Check if there are at least 4 points
    assert len(good)>=4, Exception(f'Too Few Mathches (got/need: {len(good)}/{MIN_MATCH})')

    goal = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    curr = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(curr, goal, cv2.RANSAC, 5.0)
    hei,wid = imgAG.shape
    pts = np.float32([ [0,0],[0,hei-1],[wid-1,hei-1],[wid-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, M)
    img = cv2.polylines(imgBG, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    
    if args.verbose:
        cv2.imshow("Merged", img)
        cv2.waitKey()

    # Transform image
    dst = cv2.warpPerspective(imgB,M,(imgA.shape[1] + imgB.shape[1], imgA.shape[0]))
    dst[0:imgA.shape[0],0:imgA.shape[1]] = imgA

    # Save image
    if args.output:
        print(f"Saving to {args.output}...")
        grey = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        grey[grey<250] = 0
        cv2.imwrite(args.output, grey, [cv2.IMWRITE_PXM_BINARY])
    else: 
        cv2.imshow("Stitched Image", dst)
        cv2.waitKey()

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
    parser.add_argument('descriptor', type=str, choices=['sift','surf','brief','orb'],
                        help='Select descriptor to match the images features.'
                        )
    parser.add_argument('-t','--threshold', nargs='?', default=0.1, type=float,
                        help='Define error to accept matched points.'
                        )
    parser.add_argument('-o','--output', type=str,
                        help='Path of where to save the stiched image.'
                        )
    parser.add_argument('-v','--verbose', action='store_true',
                        help='Exhibit images with the connected components.'
                        )

    args = parser.parse_args()
   
    print(args.threshold)

    main(args)
