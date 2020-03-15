import cv2
import numpy as np 

baboon = cv2.imread("input/baboon.png")
butterfly = cv2.imread("input/butterfly.png")
city = cv2.imread("input/city.png")
house = cv2.imread("input/house.png")
seagull = cv2.imread("input/seagull.png")
# imgs = [baboon, butterfly, city, house, seagull]

# cv2.bitwise_not()

def negative (img) : 
    return 255 - img

def transform (img) : 
    return ((img/255)*100).astype(np.uint8) + 100

def invert (img) : 
    mask = [True, False]*((img.shape[0]+1)//2)
    img[mask] = np.flip(img[mask],axis=1)
    return img

def mirror (img) :
    mask = [False]*(img.shape[0]//2) 
    mask += [True]*(img.shape[0] - img.shape[0]//2)
    img[mask] = np.flip(img[mask[::-1]], axis=0)
    return img

def brightness (img, gamma) : 
    return (((img/255)**(1/gamma))*255).astype(np.uint8)

def bit_plane (img, plane) : 
    mask = np.full(img.shape, 2**plane, dtype=np.uint8)
    img[np.bitwise_and(img, mask) != 0] = 255
    return img

def mix (img1, img2, w1) : 
    return (img1*w1 + img2*(1-w1)).astype(np.uint8)

def mosaic (img, pos) :
    row_cut = list(range(img.shape[0]-img.shape[0]%len(pos), img.shape[0]))
    col_cut = list(range(img.shape[1]-img.shape[1]%len(pos[0]), img.shape[1]))
    np.delete(img, col_cut, axis=0)
    np.delete(img, row_cut, axis=1)
    H = img.shape[0]//len(pos) 
    W = img.shape[1]//len(pos[0])

    blocks = []
    for i in range(0,len(pos)):
        for j in range(0,len(pos[0])):
            blocks.append(img[H*i:H*i+H, W*j:W*j+W])
            
    result = np.empty(img.shape, dtype=np.uint8)
    for i in range(0,len(pos)):
        for j in range(0,len(pos[0])):
            result[H*i:H*i+H, W*j:W*j+W] = blocks[pos[i][j]-1]

    return result

tiles = [[6,11,13,3],[8,16,1,9],[12,14,2,7],[4,15,10,5]]
cv2.imshow("Mosaic", mosaic(baboon, tiles))
cv2.waitKey()