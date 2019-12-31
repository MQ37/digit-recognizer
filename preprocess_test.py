import numpy as np
import pandas as pd
import copy
from multiprocessing.dummy import Pool as ThreadPool
import os

threads = int(os.environ['NUMBER_OF_PROCESSORS'])

data = pd.read_csv("test.csv")
pixel_keys = [i for i in data.keys() if i.startswith("pixel")]
pixel_keys = sorted(pixel_keys)
    
def dfrow2img(row_id):
    _pixel_keys = copy.deepcopy(pixel_keys)
    row = data.loc[row_id]
    img = []
    for _ in range(28):
        im_row = []
        for _ in range(28):
            im_row.append( row[_pixel_keys.pop(0)] )
        img.append(im_row)
    return img

import time
s = time.time()

pool = ThreadPool()

ret = pool.map(dfrow2img, list(range(len(data))))
pool.close()
pool.join()

images= ret

images = np.array(images)

print(images.shape)

np.save("images_test.npy", images)

print(time.time() - s)