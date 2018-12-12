import os
global cnt
os.environ['KERAS_BACKEND'] = 'tensorflow'  # can choose theano, tensorflow, cntk
import keras.backend as K
from generator import UNET_G
from discriminator import BASIC_D
from keras.optimizers import Adam
from PIL import Image
import numpy as np
import time
from IPython.display import clear_output
import glob
from IPython.display import display
from random import randint, shuffle


K.set_image_data_format('channels_last')
channel_axis = -1
channel_first = False


nc_in = 3
nc_out = 3
ngf = 64
ndf = 64
use_lsgan = True
λ = 10 if use_lsgan else 100

loadSize = 143
imageSize = 128
batchSize = 1
lrD = 2e-4
lrG = 2e-4

netDA = BASIC_D(nc_in, ndf, use_sigmoid=not use_lsgan)
netDB = BASIC_D(nc_out, ndf, use_sigmoid=not use_lsgan)
netDA.summary()

netGB = UNET_G(imageSize, nc_in, nc_out, ngf)
netGA = UNET_G(imageSize, nc_out, nc_in, ngf)
netGA.summary()



if use_lsgan:
    loss_fn = lambda output, target: K.mean(K.abs(K.square(output - target)))
else:
    loss_fn = lambda output, target: -K.mean(K.log(output + 1e-12) * target + K.log(1 - output + 1e-12) * (1 - target))


def cycle_variables(netG1, netG2):
    real_input = netG1.inputs[0]
    fake_output = netG1.outputs[0]
    rec_input = netG2([fake_output])
    fn_generate = K.function([real_input], [fake_output, rec_input])
    return real_input, fake_output, rec_input, fn_generate


real_A, fake_B, rec_A, cycleA_generate = cycle_variables(netGB, netGA)
real_B, fake_A, rec_B, cycleB_generate = cycle_variables(netGA, netGB)


def cal_loss(netD, real, fake, rec):
    output_real = netD([real])
    output_fake = netD([fake])
    loss_D_real = loss_fn(output_real, K.ones_like(output_real))
    loss_D_fake = loss_fn(output_fake, K.zeros_like(output_fake))
    loss_G = loss_fn(output_fake, K.ones_like(output_fake))
    loss_D = loss_D_real + loss_D_fake
    loss_cyc = K.mean(K.abs(rec - real))
    return loss_D, loss_G, loss_cyc


loss_DA, loss_GA, loss_cycA = cal_loss(netDA, real_A, fake_A, rec_A)
loss_DB, loss_GB, loss_cycB = cal_loss(netDB, real_B, fake_B, rec_B)
loss_cyc = loss_cycA + loss_cycB
loss_G = loss_GA + loss_GB + λ * loss_cyc
loss_D = loss_DA + loss_DB

weightsD = netDA.trainable_weights + netDB.trainable_weights
weightsG = netGA.trainable_weights + netGB.trainable_weights

training_updates = Adam(lr=lrD, beta_1=0.5).get_updates(weightsD, [], loss_D)
netD_train = K.function([real_A, real_B], [loss_DA / 2, loss_DB / 2], training_updates)
training_updates = Adam(lr=lrG, beta_1=0.5).get_updates(weightsG, [], loss_G)
netG_train = K.function([real_A, real_B], [loss_GA, loss_GB, loss_cyc], training_updates)



def load_data(file_pattern):
    return glob.glob(file_pattern)  #glob module finds all the pathnames matching a specified pattern


def read_image(fn):
    im = Image.open(fn).convert('RGB')
    im = im.resize((loadSize, loadSize), Image.BILINEAR)
    #Resize performance In most cases, convolution is more a
    # expensive algorithm for downscaling because it takes into
    # account all the pixels of source image.
    # Therefore BILINEAR and BICUBIC filters’ performance can be lower than before.
    arr = np.array(im) / 255 * 2 - 1 #map all values between 0 and 1
    w1, w2 = (loadSize - imageSize) // 2, (loadSize + imageSize) // 2 #w1 is 7 and w2 is 135
    h1, h2 = w1, w2
    img = arr[h1:h2, w1:w2, :]  #returns a 128x128 portion of image which is of size 143x143
    if randint(0, 1):  #produces 0 or 1 randomly
        img = img[:, ::-1]
    if channel_first:
        img = np.moveaxis(img, 2, 0)
    return img


# data = "edges2shoes"
data = "cityscapes"
train_A = load_data('cityscapes/trainA/*.jpg')
train_B = load_data('cityscapes/trainB/*.jpg')

print(len(train_A))
print(len(train_B))
assert len(train_A) and len(train_B)




def minibatch(data, batchsize):
    length = len(data)
    epoch = i = 0
    tmpsize = None
    while True:
        size = tmpsize if tmpsize else batchsize
        if i + size > length:
            shuffle(data)
            i = 0
            epoch += 1
        rtn = [read_image(data[j]) for j in range(i, i + size)]
        i += size
        tmpsize = yield epoch, np.float32(rtn)
        #print(tmpsize,epoch,size,i,length,batchSize,"***")




def minibatchAB(dataA, dataB, batchsize):
    batchA = minibatch(dataA, batchsize)
    batchB = minibatch(dataB, batchsize)
    tmpsize = None
    while True:
        ep1, A = batchA.send(tmpsize)
        ep2, B = batchB.send(tmpsize)
        tmpsize = yield max(ep1, ep2), A, B






cnt=0
def showX(X, rows=1):
    global cnt
    assert X.shape[0] % rows == 0
    int_X = ((X + 1) / 2 * 255).clip(0, 255).astype('uint8')
    if channel_first:
        int_X = np.moveaxis(int_X.reshape(-1, 3, imageSize, imageSize), 1, 3)
    else:
        int_X = int_X.reshape(-1, imageSize, imageSize, 3)
    int_X = int_X.reshape(rows, -1, imageSize, imageSize, 3).swapaxes(1, 2).reshape(rows * imageSize, -1, 3)
    display(Image.fromarray(int_X))
    arr = Image.fromarray(int_X)
    #if cnt%500==0:
        #arr.save('results_cityscapes/ig' + str(cnt) + '.jpeg', 'jpeg')
    cnt = cnt+ 1


train_batch = minibatchAB(train_A, train_B, 6)

_, A, B = next(train_batch)
showX(A)
showX(B)
del train_batch, A, B


def showG(A, B):
    assert A.shape == B.shape

    def G(fn_generate, X):
        r = np.array([fn_generate([X[i:i + 1]]) for i in range(X.shape[0])])
        return r.swapaxes(0, 1)[:, :, 0]

    rA = G(cycleA_generate, A)
    rB = G(cycleB_generate, B)
    arr = np.concatenate([A, B, rA[0], rB[0], rA[1], rB[1]])
    showX(arr, 3)




t0 = time.time()
niter = 500
gen_iterations = 0
epoch = 0
errCyc_sum = errGA_sum = errGB_sum = errDA_sum = errDB_sum = 0

display_iters = 1
# val_batch = minibatch(valAB, 6, direction)
train_batch = minibatchAB(train_A, train_B, batchSize)


prv=-1
fp = open('Loss cityscapes.txt', 'w+')
while epoch < niter:
    epoch, A, B = next(train_batch)
    errDA, errDB = netD_train([A, B])
    errDA_sum += errDA
    errDB_sum += errDB

    # epoch, trainA, trainB = next(train_batch)
    errGA, errGB, errCyc = netG_train([A, B])
    errGA_sum += errGA
    errGB_sum += errGB
    errCyc_sum += errCyc
    gen_iterations += 1
    print(epoch,gen_iterations,prv)

    '''if prv+1==epoch:
        prv=epoch
        if epoch%20==0 and prv==epoch:
            with open("model_cityscapes_ga/netga"+str(epoch)+".json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            netGA.save_weights("model_cityscapes_ga_weights/netgaw_"+str(epoch)+".h5")
            print("Saved model to disk")

            with open("model_cityscapes_gb/netgb"+str(epoch)+".json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            netGB.save_weights("model_cityscapes_gb_weights/netgbw_"+str(epoch)+".h5")
            print("Saved model to disk")

            with open("model_cityscapes_da/netda"+str(epoch)+".json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            netDA.save_weights("model_cityscapes_da_weights/netdaw_"+str(epoch)+".h5")
            print("Saved model to disk")

            with open("model_cityscapes_db/netdb"+str(epoch)+".json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            netDB.save_weights("model_cityscapes_db_weights/netdbw_"+str(epoch)+".h5")
            print("Saved model to disk")'''

    if gen_iterations % display_iters == 0:
        clear_output()
        #print('[%d/%d][%d] Loss_D: %f %f Loss_G: %f %f loss_cyc %f'%(epoch, niter, gen_iterations, errDA_sum / display_iters, errDB_sum / display_iters,errGA_sum / display_iters, errGB_sum / display_iters,errCyc_sum / display_iters), time.time() - t0,file=fp)
        print('[%d/%d][%d] Loss_D: %f %f Loss_G: %f %f loss_cyc %f' % (epoch, niter, gen_iterations, errDA_sum / display_iters, errDB_sum / display_iters, errGA_sum / display_iters,errGB_sum / display_iters, errCyc_sum / display_iters), time.time() - t0)
        #fp.write('\n')
        _, A, B = train_batch.send(4)
        showG(A, B)
        errCyc_sum = errGA_sum = errGB_sum = errDA_sum = errDB_sum = 0

fp.close()