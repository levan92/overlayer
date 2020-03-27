from pathlib import Path
import xml.dom.minidom

class XML_reader(object):
    def __init__(self, xml_path, root_dir):
        self.doc = xml.dom.minidom.parse(xml_path)
        self.root_dir = root_dir

    def get_polygons(self, target_classes=None):
        polygons = []
        for image in self.doc.getElementsByTagName("image"):
            imname = image.getAttribute("name")
            impath = Path(self.root_dir) / imname
            assert impath.is_file(), '{}'.format(impath)
            imwidth = image.getAttribute("width")
            imheight = image.getAttribute("height")
            print('{},{},{}'.format(imname, imwidth, imheight))
            for polygon in image.getElementsByTagName("polygon"):
                label = polygon.getAttribute("label")
                if target_classes and label not in target_classes:
                    continue
                # if bool(int(polygon.getAttribute("outside"))):
                #     continue
                points_str = polygon.getAttribute("points")
                points = []
                for pairs in points_str.split(';'):
                    x, y = pairs.split(',')
                    points.append((int(float(x)), int(float(y))))
                if len(points) > 0:
                    polygons.append((str(impath), points))
        return polygons

if __name__ == '__main__':
    xml_fp =  'test_images/15_sampan_img.xml' 
    root = 'test_images/small/'
    xml_reader = XML_reader(xml_fp, root)
    polygons = xml_reader.get_polygons(['sampan'])
    print(polygons)