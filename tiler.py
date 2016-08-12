#!/usr/bin/env python
'''
Simple utility to tile images
'''

import numpy as np
import cv2
from matplotlib import pyplot as plt
import os,sys

if __name__ == '__main__':

    in_dir = sys.argv[1]
    out_img_name = sys.argv[2]

    ll = os.listdir(in_dir)
    img_names = []
    for i in ll:
        if os.path.splitext(i)[-1] in ['.jpg','.png','.ppm']:
            img_names.append(i)

    imgs = [cv2.imread(os.path.join(in_dir,i)) for i in img_names]
    num_imgs = len(img_names)
    side = int(np.sqrt(num_imgs))+1
    print(num_imgs)
    #full_img = np.zeros((side*imgs[0].shape[0],side*imgs[0].shape[1]))
    rows = []
    for i in xrange(side):
        cols = []
        for j in xrange(side):
            if i*side + j < num_imgs:
                cols.append(np.pad(imgs[i*side+j],((5,5),(5,5),(0,0)),mode='constant'))
                print cols[-1].shape
            else:
                cols.append(np.ones((imgs[0].shape[0]+10,imgs[0].shape[1]+10,3))*255)
        print len(cols)
        rows.append(np.hstack(cols))

    out_img = np.vstack(rows)

    cv2.imwrite(out_img_name,out_img)
    plt.imshow(out_img)
    plt.show()


