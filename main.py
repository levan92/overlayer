import cv2
import random
import numpy as np

from utils.cvat_xml_reader import XML_reader 

backdrop_fp = 'test_images/SMD_VIS_small/Imgs/MVI_1451_VIS_Haze_frame_30.png'
xml_fp =  'test_images/15_sampan_img.xml' 
root = 'test_images/small/'
target_classes = ['sampan']
xml_reader = XML_reader(xml_fp, root)

polygons_points = xml_reader.get_polygons(target_classes)

backdrop = cv2.imread(backdrop_fp)

for i, poly in enumerate(polygons_points):
    imgpath, points = poly
    img = cv2.imread(imgpath)

    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(points)], 255)
    sampan = cv2.bitwise_or(img, img, mask=mask)

    inv_mask = cv2.bitwise_not(mask)
    backdrop_resized = cv2.resize(backdrop, (img.shape[1], img.shape[0]))
    bg = cv2.bitwise_or(backdrop_resized, backdrop_resized, mask=inv_mask)

    final = cv2.bitwise_or(sampan, bg)
    cv2.imwrite('final{}.jpg'.format(i), final)

def random_resize(img, range=0.2):
    h, w = img.shape[:2]
    factor = np.random.uniform(1.0 - range, 1.0 + range)
    return cv2.resize(img, (int(w*factor),int(h*factor)))

def is_intersect(boxA, boxB):
    '''
    box : [xmin,ymin,xmax,ymax]
    '''
    a_xmin, a_ymin, a_xmax, a_ymax = boxA
    b_xmin, b_ymin, b_xmax, b_ymax = boxB
    return not (a_xmax < b_xmin or a_ymax < b_ymin or b_xmax < a_xmin or b_ymax < a_ymin)

def get_valid_zone(boxes, frame_size):
    '''
    boxes : list of ltrb
    frame_size : height, width
    ------
    Returns zone (ltrb)
    '''
    horizon = min(boxes, key = lambda x: x[-1])
    return [ 0, horizon, frame_size[1]-1, frame_size[0]-1 ]    

def get_random_valid_point(zone, boxes, size, max_reps=1000):
    '''
    zone : ltrb
    boxes : list of ltrb
    size : height, width
    -------
    Returns (x,y) or None
    '''
    zl,zt,zr,zb = zone
    h,w = size
    for _ in range(max_reps):
        x = np.random.randint(zl, zr+1)
        y = np.random.randint(zt, zb+1)
        l = x - (w + 1) / 2
        r = x + (w + 1) / 2
        t = y - (h + 1) / 2
        b = y + (h + 1) / 2 
        for box in boxes:
            if is_intersect(box, [l,t,r,b]):
                break
        else: # no intersection with any box
            return x,y
    return None

def overlap(bg, bg_boxes, fg_chip, polygon_mask, horizon=None):
    masked_chip = cv2.bitwise_or(fg_chip, fg_chip, mask=polygon_mask)
    masked_chip = random_resize(masked_chip, range=0.2)
    valid_zone = get_valid_zone(bg_boxes)
    x, y = get_random_valid_point(valid_zone, bg_boxes, masked_chip.shape[:2], max_reps=1000)

    inv_mask = cv2.bitwise_not(polygon_mask)
    bg = cv2.bitwise_or(backdrop_resized, backdrop_resized, mask=inv_mask)

    final = cv2.bitwise_or(sampan, bg)

if __name__ == '__main__':
    backdrop = cv2.imread(backdrop_fp)
    print(backdrop.shape)
    res = random_resize(backdrop)
    print(res.shape)
    cv2.imshow('', res)
    cv2.waitKey(0)