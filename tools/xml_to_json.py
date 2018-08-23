import xml.etree.ElementTree as ET
import json
import numpy as np
import argparse, cv2, os
import json

def parse_xml(annotations, xml_path, image_id, id):
    tree = ET.parse(xml_path)
    objs = tree.findall('object')
    for ix, obj in enumerate(objs):
        cls_name = obj.find('name').text
