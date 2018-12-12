import numpy as np
from scipy import misc
import os
import glob
from PIL import Image
from keras.models import model_from_json
from random import randint
loadSize = 143
imageSize = 128
channel_first=False

def load_data(file_pattern):
    return glob.glob(file_pattern)

def read_image(fn):
    im = Image.open(fn).convert('RGB')
    im = im.resize((loadSize, loadSize), Image.BILINEAR)
    arr = np.array(im) / 255 * 2 - 1
    w1, w2 = (loadSize - imageSize) // 2, (loadSize + imageSize) // 2
    h1, h2 = w1, w2
    img = arr[h1:h2, w1:w2, :]
    if randint(0, 1):
        img = img[:, ::-1]
    if channel_first:
        img = np.moveaxis(img, 2, 0)
    return img

ls=os.listdir('input_data')
c=1
json_file = open('maps_model/netgb500.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
ga = model_from_json(loaded_model_json)
ga.load_weights("maps_model/netgbw_500.h5")
print("Loaded model from disk")
print(ls)

train_A = load_data('input_data/*'+'.jpg')
c=1
for im in train_A:
    im=read_image(im)
    im=im.reshape(1,128,128,3)
    op=ga.predict(im)
    op=op[0,:,:,:]
    im=im[0,:,:,:]
    im=misc.toimage(im)
    im.save('output_data/ip'+str(c)+'.jpg')
    op=misc.toimage(op)
    op.save('output_data/op'+str(c)+'.jpg')
    c=c+1


