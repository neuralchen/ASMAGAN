#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: selective_copy_imgs.py
# Created Date: Saturday April 18th 2020
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Saturday, 18th April 2020 7:24:59 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################
from pathlib import Path
import os
import shutil
from tqdm import tqdm

if __name__ == "__main__":
    categories_names = \
        ['a/abbey', 'a/arch',
        'a/amphitheater', 'a/aqueduct', 'a/arena/rodeo', 'a/athletic_field/outdoor',
         'b/badlands', 'b/balcony/exterior', 'b/bamboo_forest', 'b/barn', 'b/barndoor', 'b/baseball_field',
         'b/basilica', 'b/bayou', 'b/beach', 'b/beach_house', 'b/beer_garden', 'b/boardwalk', 'b/boathouse',
         'b/botanical_garden', 'b/bullring', 'b/butte', 'c/cabin/outdoor', 'c/campsite', 'c/campus',
         'c/canal/natural', 'c/canal/urban', 'c/canyon', 'c/castle', 'c/church/outdoor', 'c/chalet',
         'c/cliff', 'c/coast', 'c/corn_field', 'c/corral', 'c/cottage', 'c/courtyard', 'c/crevasse',
         'd/dam', 'd/desert/vegetation', 'd/desert_road', 'd/doorway/outdoor', 'f/farm', 'f/fairway',
         'f/field/cultivated', 'f/field/wild', 'f/field_road', 'f/fishpond', 'f/florist_shop/indoor',
         'f/forest/broadleaf', 'f/forest_path', 'f/forest_road', 'f/formal_garden', 'g/gazebo/exterior',
         'g/glacier', 'g/golf_course', 'g/greenhouse/indoor', 'g/greenhouse/outdoor', 'g/grotto', 'g/gorge',
         'h/hayfield', 'h/herb_garden', 'h/hot_spring', 'h/house', 'h/hunting_lodge/outdoor', 'i/ice_floe',
         'i/ice_shelf', 'i/iceberg', 'i/inn/outdoor', 'i/islet', 'j/japanese_garden', 'k/kasbah',
         'k/kennel/outdoor', 'l/lagoon', 'l/lake/natural', 'l/lawn', 'l/library/outdoor', 'l/lighthouse',
         'm/mansion', 'm/marsh', 'm/mausoleum', 'm/moat/water', 'm/mosque/outdoor', 'm/mountain',
         'm/mountain_path', 'm/mountain_snowy', 'o/oast_house', 'o/ocean', 'o/orchard', 'p/park',
         'p/pasture', 'p/pavilion', 'p/picnic_area', 'p/pier', 'p/pond', 'r/raft', 'r/railroad_track',
         'r/rainforest', 'r/rice_paddy', 'r/river', 'r/rock_arch', 'r/roof_garden', 'r/rope_bridge',
         'r/ruin', 's/schoolhouse', 's/sky', 's/snowfield', 's/swamp', 's/swimming_hole',
         's/synagogue/outdoor', 't/temple/asia', 't/topiary_garden', 't/tree_farm', 't/tree_house',
         'u/underwater/ocean_deep', 'u/utility_room', 'v/valley', 'v/vegetable_garden', 'v/viaduct',
         'v/village', 'v/vineyard', 'v/volcano', 'w/waterfall', 'w/watering_hole', 'w/wave',
         'w/wheat_field', 'z/zen_garden', 'a/alcove', 'a/apartment-building/outdoor', 'a/artists_loft',
         'b/building_facade', 'c/cemetery']
    root = "/mnt/harddisk/CXH/place365_data_large/"
    target = "/home/gdp/CXH/data_set/Place365_large"
    for item in categories_names:
        images = Path(root).glob('%s/*.%s'%(item, "jpg"))
        target_dir = Path(target, item.replace('/','_'))
        print("processing %s"%item,end='\r')
        join_path = Path(root,item)
        if join_path.exists():
            if not target_dir.exists():
                os.makedirs(target_dir)
            for img in images:
                shutil.copy(img,target_dir)
        else:
            print("%s does not exist!"%item)
            continue
        print("finish process %s\n"%item)