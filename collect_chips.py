import cv2
import argparse
import numpy as np
from pathlib import Path

from utils.cvat_xml_reader import XML_reader

parser = argparse.ArgumentParser()
parser.add_argument('xml_file',help='Path to cvat xml (image) file')
parser.add_argument('root', help='Path to root dir containing the images')
parser.add_argument('--target', help='list of target classes seperated by commas, defaults to take all polygons')
parser.add_argument('--out', help='Path to output directory of polygon, defaults to ./out_chips', default='./out_chips')
args = parser.parse_args()

assert Path(args.xml_file).is_file()
assert Path(args.root).is_dir()
target_classes = None
if args.target:
    target_classes = args.target.split(',')

if args.out:
    Path(args.out).mkdir(parents=True, exist_ok=True)

reader = XML_reader(args.xml_file, args.root)
polygon_points = reader.get_polygons(target_classes=target_classes)

for idx, poly in enumerate(polygon_points):
    imgpath, points = poly
    x_min = min(points, key= lambda t: t[0])[0]
    x_max = max(points, key= lambda t: t[0])[0]
    y_min = min(points, key= lambda t: t[1])[1]
    y_max = max(points, key= lambda t: t[1])[1]
    points = [ (x - x_min ,y - y_min) for x,y in points ]

    img = cv2.imread(imgpath)
    cropped = img[y_min:y_max+1, x_min:x_max+1]
    mask = np.zeros((cropped.shape[0], cropped.shape[1]), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(points)], 255)
    chip = cv2.bitwise_or(cropped, cropped, mask=mask)
    print('{}_{}_{}'.format(idx+1, Path(args.xml_file).stem, Path(imgpath).stem))
    chip_name = '{}_{}_{}.png'.format(idx+1, Path(args.xml_file).stem, Path(imgpath).stem)
    mask_name = '{}_{}_{}.mask.npy'.format(idx+1, Path(args.xml_file).stem, Path(imgpath).stem)
    chip_path = str( Path(args.out) / chip_name )
    mask_path = str( Path(args.out) / mask_name )
    cv2.imwrite( chip_path, chip )
    np.save(mask_path, mask, allow_pickle=False)

    cv2.imwrite('{}_{}_{}.frame.jpg'.format(idx+1, Path(args.xml_file).stem, Path(imgpath).stem), img)