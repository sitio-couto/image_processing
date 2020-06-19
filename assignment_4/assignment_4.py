import sys, argparse
import cv2
import numpy as np

def show(msg, img, res=.6):
    cv2.imshow(msg,cv2.resize(img, (0,0), fx=res, fy=res))
    cv2.waitKey()

def convert_pbm(path, threshold):
    img = cv2.imread(path, 0)
    if args.verbose: show("Raw Input", img)
        
    img[img >  threshold] = 255
    img[img <= threshold] = 0
    for i,l in enumerate(reversed(path)):
        if l=='.': 
            out = path[:-(i+1)]+'.pbm'
            break

    if args.verbose: show("Raw Input", img)
    cv2.imwrite(out, img, [cv2.IMWRITE_PXM_BINARY])
    print(f"Image converted and saved to: {out}")
    exit()

def morphing (img, kA, kB, kC):
    img = img.copy()
    if args.double_verbose: show("Negated Input", img)
    
    # Join and isolate sets with horizontal proximity
    kernel = np.ones(kA,np.uint8)
    horizontal = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    if args.double_verbose: show(f"Horizontal Closing {kA}", horizontal)

    # Join and isolate sets with vertical proximity
    kernel = np.ones(kB,np.uint8)
    vertical = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    if args.double_verbose: show(f"Vertical Closing {kB}", vertical)

    # Intersect closings to tag blocks in the image
    intersection = cv2.bitwise_and(horizontal, vertical)
    if args.double_verbose: show(f"Intesection", intersection)

    # Refine blocks to improve connectivity
    kernel = np.ones(kC,np.uint8)
    refined = cv2.morphologyEx(intersection, cv2.MORPH_CLOSE, kernel)
    
    # Fetch connected components from the segmented image
    stats = cv2.connectedComponentsWithStats(refined, 4, cv2.CV_32S)[2]
    return refined,stats

def ratios(box, wid, hei):
    qnt_black = (box==255).sum()
    bw_ratio = qnt_black/(hei*wid)

    transitions = 0
    for i in range (0, hei):
        transitions += (box[i, :-1] < box[i, 1:]).sum()
    for i in range (0, wid):
        transitions += (box[:-1, i] < box[1:, i]).sum()

    if transitions:
        trans_ratio = transitions/qnt_black
    else:
        trans_ratio = 0

    return bw_ratio,trans_ratio

def classification(
                rgb,neg,
                kernels  =None,
                bw_range =None,
                tr_range =None,
                color    =None,
                tag      =False
                ):

    check = lambda val,l: bool(l[0]<=val<=l[1])
    count = 0

    # Capturing lines
    morphed,stats = morphing(neg, *kernels)
    if args.verbose: show("Components", morphed)
    for x0,y0,wid,hei,_ in stats:
        # Extract the bouding box of a connected component
        box = morphed[y0:y0+hei, x0:x0+wid]
        bw_ratio,trans_ratio = ratios(box, wid, hei)
        # Checking if component is a line
        if check(bw_ratio, bw_range) and check(trans_ratio, tr_range):
            count += 1
            if tag:
                cv2.rectangle(rgb, (x0,y0), (x0+wid, y0+hei), color, 3)
        elif args.debug:
            print(f"\nBW Ratio: {bw_ratio}\nTR Ratio {trans_ratio}")
            cv2.rectangle(rgb, (x0,y0),(x0+wid,y0+hei), (0,0,255), 3)
            show("DEBUG", rgb)

    return count
                
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
