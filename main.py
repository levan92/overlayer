import cv2
import numpy as np

from utils.cvat_xml_reader import XML_reader 

xml_fp =  'test_images/15_sampan_img.xml' 
root = 'test_images/small/'
target_classes = ['sampan']
xml_reader = XML_reader(xml_fp, root)

polygons_points = xml_reader.get_polygons(target_classes)

for poly in polygons_points:
    imgpath, points = poly
    img = cv2.imread(imgpath)
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(points)], 255)
    cv2.imshow('', mask)
    cv2.waitKey(0)