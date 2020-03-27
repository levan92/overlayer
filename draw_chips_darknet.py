import cv2
import random
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('darknet_txt',help='Path to dataset txt file in darknet (aka yolo) annotation format')
parser.add_argument('--chips',help='Path to dir of chips, defaults to ./out_chips', default='./out_chips')
parser.add_argument('--chip_class', help='Chip class label idx, defaults to 0', default=0, type=int)
parser.add_argument('--aug_chance', help='Chance of augmentation, defaults to 0.75', default=0.75, type=float)
args = parser.parse_args()

def random_resize(img, mask, range=0.2):
    h, w = img.shape[:2]
    factor = np.random.uniform(1.0 - range, 1.0 + range)
    rs_img = cv2.resize(img, (int(w*factor),int(h*factor)))
    rs_mask = cv2.resize(mask, (int(w*factor),int(h*factor)))
    return rs_img, rs_mask

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
    if len(boxes) > 0:
        horizon = min(boxes, key = lambda x: x[-1])[-1]
        frame_height, frame_width = frame_size
        return [ 0, horizon, frame_width-1, frame_height-1 ]    
    else:
        return None

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

def paste_chip(chip, bg, point):
    x, y = point
    h, w = chip.shape[:2]
    l = int(x - (w+1)/2)
    t = int(y - (h+1)/2) 

    bg_h, bg_w = bg.shape[:2]

    y_start = int( max(0, t) )
    y_end = int( min(bg_h, int(t+h)) )
    x_start = int( max(0, l) )
    x_end = int( min(bg_w, int(l+w)) )

    chip_y_start = int( y_start - t ) 
    chip_y_end = int( h - (int(t+h) -  y_end) )
    chip_x_start = int( x_start - l ) 
    chip_x_end = int( w - (int(l+w) -  x_end) )

    bg[y_start:y_end, x_start:x_end] = chip[chip_y_start:chip_y_end, chip_x_start:chip_x_end]

    chip_new_coord = [x_start, y_start, x_end-1, y_end-1]

    return bg, chip_new_coord

def overlap(bg, bg_boxes, fg_chip, mask, horizon=None):
    valid_zone = get_valid_zone(bg_boxes, bg.shape[:2])
    if valid_zone is None:
        return None, None
    fg_chip, mask = random_resize(fg_chip, mask, range=0.2)
    point = get_random_valid_point(valid_zone, bg_boxes, fg_chip.shape[:2], max_reps=1000)
    if point is None:
        return None, None

    fg_darkness = np.zeros((bg.shape[0], bg.shape[1], 3), dtype=np.uint8)
    fg, chip_new_coord = paste_chip(fg_chip, fg_darkness, point)
    
    bg_whiteness = np.full((bg.shape[0], bg.shape[1]), 255, dtype=np.uint8)
    inv_mask = cv2.bitwise_not(mask)
    bg_mask, _ = paste_chip(inv_mask, bg_whiteness, point)
    bg = cv2.bitwise_or(bg, bg, mask=bg_mask)

    final = cv2.bitwise_or(fg, bg)
    return final, chip_new_coord

def coin_flip():
    return bool(random.getrandbits(1))

assert Path(args.darknet_txt).is_file()
with open(args.darknet_txt,'r') as f:
    lines = f.readlines()
impath2bbs = {}
dataset_root = None
for line in lines:
    impath = line.strip()
    assert Path(impath).is_file(),'{}'.format(impath)
    iw, ih = Image.open(impath).size
    if dataset_root is None:
        dataset_root = Path(impath).parent.parent
    labelpath = dataset_root / 'labels' / '{}.txt'.format(Path(impath).stem)
    assert labelpath.is_file()
    with labelpath.open('r') as f:
        box_lines = f.readlines()
    
    bbs = []
    box_strs = []
    for box in box_lines:
        box = box.strip()
        cl, x, y, w, h = box.split(' ')
        x = int( iw * float(x) )
        y = int( ih * float(y) )
        w = int( iw * float(w) )
        h = int( ih * float(h) )
        l = int(x - (w+1)/2)
        r = int(x + (w+1)/2)
        t = int(y - (h+1)/2)
        b = int(y - (h+1)/2)
        bbs.append( (int(l),int(t),int(r),int(b)) )
        box_strs.append(box)
    
    impath2bbs[impath] = bbs, box_strs

augmented_root_dir = Path( str(dataset_root)+'_augmented' )
augmented_images_dir = augmented_root_dir / 'images'
augmented_images_dir.mkdir(parents=True, exist_ok=True)
augmented_labels_dir = augmented_root_dir / 'labels'
augmented_labels_dir.mkdir(parents=True, exist_ok=True)

og_text_name = Path(args.darknet_txt).name

augmented_list_part_text_path = augmented_root_dir / og_text_name.replace('.txt', '_augmented.part.txt')
augmented_list_text_path = augmented_root_dir / og_text_name.replace('.txt', '_augmented.txt')

assert Path(args.chips).is_dir()
chips_masks = []
for f in Path(args.chips).glob('*'):
    if f.suffix == '.png':
        mask_file = f.parent / '{}.mask.npy'.format(f.stem)
        assert mask_file.is_file(),'{}'.format(mask_file)
        chip = cv2.imread(str(f))
        mask = np.load(mask_file)
        chips_masks.append((chip, mask))

new_imgstem2annots = {}
new_imgnames = []
new_imgpaths = []
aug_count = 0
for impath, bbs_and_box_strs in tqdm(impath2bbs.items()):
# for impath, bbs in impath2bbs.items():
    bbs, box_strs = bbs_and_box_strs
    to_augment = ( random.uniform(0, 1) <= args.aug_chance )
    # to_augment = coin_flip()
    if to_augment:
    # if True:
        state = 'aug'
        draw_chip, draw_mask = random.choice(chips_masks)
    else:
        state = 'og'
        draw_chip, draw_mask = None, None
    backdrop = cv2.imread(impath)
    if draw_chip is not None:
        res, chip_new_coord = overlap(backdrop, bbs, draw_chip, draw_mask)
        if res is None:
            res = backdrop
            state = 'og'
    else:
        res = backdrop
        chip_new_coord = None
    # cv2.imshow('{}'.format(action), res)
    # cv2.waitKey(0)
    new_imgname = '{}_{}{}'.format(Path(impath).stem, state, Path(impath).suffix)
    new_imgname_stem = '{}_{}'.format(Path(impath).stem, state)
    newpath = augmented_images_dir / new_imgname
    cv2.imwrite(str(newpath), res)

    out_lines = []
    for box_str in box_strs:
        out_lines.append(box_str)
    if chip_new_coord:
        l,t,r,b = chip_new_coord
        ih, iw = res.shape[:2]

        w = r - l + 1
        h = b - t + 1
        x = l + (w+1)/2
        y = t + (h+1)/2

        x = x / iw
        y = y / ih
        w = w / iw
        h = h / ih
        out_lines.append('{} {} {} {} {}'.format(args.chip_class, x, y, w, h))

    new_imgstem2annots[new_imgname_stem] = out_lines 
    new_imgnames.append(new_imgname)
    new_imgpaths.append(newpath)

    if state=='aug':
        aug_count += 1

print('Augmented {} out of {} images'.format(aug_count, len(new_imgstem2annots.keys())))

for new_imgname_stem, annots in new_imgstem2annots.items():
    label_file = augmented_labels_dir / '{}.txt'.format(new_imgname_stem)
    with label_file.open('w') as f:
        for annot in annots:
            f.write(annot+'\n')

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