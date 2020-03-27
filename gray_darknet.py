import cv2
import random
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from shutil import copy
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('darknet_txt',help='Path to dataset txt file in darknet (aka yolo) annotation format')
parser.add_argument('--aug_chance', help='Chance of augmentation, defaults to 1.0', default=1.0, type=float)
args = parser.parse_args()

def coin_flip():
    return bool(random.getrandbits(1))

assert Path(args.darknet_txt).is_file()


with open(args.darknet_txt,'r') as f:
    lines = f.readlines()

dataset_root = Path(lines[0].strip()).parent.parent
augmented_root_dir = Path( str(dataset_root)+'_gray' )
augmented_images_dir = augmented_root_dir / 'images'
augmented_images_dir.mkdir(parents=True, exist_ok=True)
augmented_labels_dir = augmented_root_dir / 'labels'
augmented_labels_dir.mkdir(parents=True, exist_ok=True)
og_text_name = Path(args.darknet_txt).name
augmented_list_part_text_path = augmented_root_dir / og_text_name.replace('.txt', '_gray.part.txt')
augmented_list_text_path = augmented_root_dir / og_text_name.replace('.txt', '_gray.txt')

new_imgstems = []
new_imgnames = []
new_imgpaths = []
label_tups = []
for line in tqdm(lines):
    impath = line.strip()
    assert Path(impath).is_file(),'{}'.format(impath)

    labelpath = dataset_root / 'labels' / '{}.txt'.format(Path(impath).stem)

    img = cv2.imread(impath)
    to_gray = ( random.uniform(0,1) <= args.aug_chance )
    if to_gray:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        state = 'gray'
    else:
        gray_img = img
        state = 'og'
        print('NO GRAYING THIS ONE')

    new_imgname = '{}_{}{}'.format(Path(impath).stem, state, Path(impath).suffix)
    new_imgname_stem = '{}_{}'.format(Path(impath).stem, state)
    newpath = augmented_images_dir / new_imgname
    cv2.imwrite(str(newpath), gray_img)

    new_imgnames.append(new_imgname)
    new_imgpaths.append(newpath)

    newlabelname = '{}.txt'.format(new_imgname_stem)
    newlabelpath = augmented_labels_dir / newlabelname
    copy(str(labelpath), str(newlabelpath))

with augmented_list_part_text_path.open('w') as f:
    for imgname in new_imgnames:
        f.write(imgname+'\n')
with augmented_list_text_path.open('w') as f:
    for p in new_imgpaths:
        f.write(str(p)+'\n')


# for i, poly in enumerate(polygons_points):
#     imgpath, points = poly
#     img = cv2.imread(imgpath)

#     mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
#     cv2.fillPoly(mask, [np.array(points)], 255)
#     sampan = cv2.bitwise_or(img, img, mask=mask)

#     inv_mask = cv2.bitwise_not(mask)
#     backdrop_resized = cv2.resize(backdrop, (img.shape[1], img.shape[0]))
#     bg = cv2.bitwise_or(backdrop_resized, backdrop_resized, mask=inv_mask)

#     final = cv2.bitwise_or(sampan, bg)
#     cv2.imwrite('final{}.jpg'.format(i), final)


# if __name__ == '__main__':
#     backdrop = cv2.imread(backdrop_fp)
#     print(backdrop.shape)
#     res = random_resize(backdrop)
#     print(res.shape)
#     cv2.imshow('', res)
#     cv2.waitKey(0)