import sys, argparse, cv2
import numpy as np

def trim_edges(dst):
    img = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
    lines   = img.sum(axis=0)
    colunms = img.sum(axis=1)
    
    top,bottom = 0,(img.shape[0]-1)
    left,right = -1,(img.shape[1]-1)

    while not lines[left]: left += 1
    while not colunms[top]: top += 1
    while not lines[right]: right -= 1
    while not colunms[bottom]: bottom -= 1

    return dst[top:bottom, left:right]

def stich(desc, imgA, imgAG, imgB,imgBG):
    # Fime images features using the descriptors
    kpA,desA = desc.detectAndCompute(imgAG,None)
    kpB,desB = desc.detectAndCompute(imgBG,None)

    # Fetch mathches above the threshold (Brute Force)
    good = []
    for a,b in cv2.BFMatcher().knnMatch(desA,desB,k=2):
        if a.distance < args.threshold*b.distance: good.append(a)

    # Draw matched features for the image pair
    draw_params = dict(matchColor = (0,255,0), flags = 2)
    links = cv2.drawMatches(imgA,kpA,imgB,kpB,good,None,**draw_params)
    
    if args.verbose:
        cv2.imshow("Linked Features", links)
        cv2.waitKey()
        
    # Check if there are at least 4 points
    assert len(good)>=4, Exception(f'Too Few Mathches (got/need: {len(good)}/{4})')

    # Get homography matrix to apply warping
    goal = np.float32([ kpA[p.queryIdx].pt for p in good ]).reshape(-1,1,2)
    curr = np.float32([ kpB[p.trainIdx].pt for p in good ]).reshape(-1,1,2)
    M,_ = cv2.findHomography(curr, goal, cv2.RANSAC, 5.0)

    # Transform image
    dst = cv2.warpPerspective(imgB,M,(imgA.shape[1] + imgB.shape[1], imgA.shape[0] + imgB.shape[0]))
    dst[0:imgA.shape[0],0:imgA.shape[1]] = imgA
    
    # Trim black edges and return
    return trim_edges(dst)

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
        'brisk':cv2.BRISK_create(),
        'orb':cv2.ORB_create()
        }.get(args.descriptor)
    
    # Stich images together
    result = stich(desc, imgA, imgAG, imgB,imgBG)

    print("Done.")

    # Save image
    if args.output:
        print(f"Saving to {args.output}...")
        cv2.imwrite(args.output, result)
    if not args.output or args.verbose: 
        cv2.imshow("Stitched Image", result)
        cv2.waitKey()

#### PROCESS COMMAND LINE INPUT #####
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='''Join two images.'''
        )
    parser.add_argument('image_A',
                        help='Path to the first part of the image to be stiched.')
    parser.add_argument('image_B',
                        help='Path to the second part of the image to be stiched.')
    parser.add_argument('descriptor', type=str, choices=['sift','surf','brisk','orb'],
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
    main(args)
