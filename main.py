#coding:utf-8
"""
Author: Lyndon Lee
"""

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.local import LocallyConnected2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.recurrent import LSTM, GRU
from keras.optimizers import SGD,Adam, Adadelta, Adamax, RMSprop
from keras.utils import np_utils, generic_utils
import numpy as np
import scipy as sp
import scipy.io as sio
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.visualize_util import plot 
from keras.callbacks import ModelCheckpoint, EarlyStopping
import keras.backend.tensorflow_backend as K
import os
import pickle

from my_recurrent import TAGM

os.environ['CUDA_VISIBLE_DEVICES']="2"

#%%
print('Loading data...')
train_dir='/usr0/home/liandonl/Datasets/CL_train/'
test_dir='/usr0/home/liandonl/Datasets/CL_val/'

label_list=['N2H','N2S','N2D','N2A','N2C','N2Sur','S2N2H','H2N2S','H2N2D','H2N2A','H2N2C','D2N2Sur']
#emo_list=['ANGER','CONTENTMENT','DISGUST','HAPPINESS','SADNESS','SURPRISE']
emo_list=['HAPPINESS','SADNESS','DISGUST','ANGER','CONTENTMENT','SURPRISE']
PREDICTED_LABEL=['fake','true']

#0:白 1:黄 2:蓝 3:绿 4:红

IDs_val=[
    [1,3,2,2,0,0,4,3,4,1],
    [2,4,1,0,3,4,0,1,2,3],
    [1,3,2,2,4,1,4,3,0,0],
    [0,3,3,2,0,1,4,2,1,4],
    [3,2,2,1,3,1,4,0,0,4],
    [0,1,3,2,4,3,0,2,4,1],
]

IDs_val=np.array(IDs_val)

def get_mean_train():
    mean_data=np.zeros([40,4464])
    for i in xrange(40):
        hog_data=np.zeros([0,4464],dtype='float64')
        for j in xrange(6):
            mat_dir=train_dir+label_list[j]+'/'
            mat_name=mat_dir+str(i+1)+'.mat'
            matfile=sio.loadmat(mat_name)
            temp_data=matfile['hog_data']
            if np.shape(temp_data)[0]==0:
                temp_data=np.zeros([TIME_STEP,4464])
            #print np.shape(temp_data)
            #print np.shape(hog_data)
            hog_data=np.concatenate([hog_data, temp_data],axis=0)

        mean = np.mean(hog_data,axis=0)
        if np.sum(mean)==0:
            mean=np.zeros([4464,],dtype='float64')
        mean_data[i,:]=mean

    return mean_data

def get_mean_test():
    mean_data=np.zeros([5,4464])
    for i in xrange(5):
        hog_data=np.zeros([0,4464],dtype='float64')
        for j in xrange(6):
            mat_dir=test_dir+emo_list[j]+'/'
            mat_files=os.listdir(mat_dir)
            #print np.where(IDs_val[j,:]==i)
            idx=np.where(IDs_val[j,:]==i)
            for k in xrange(2):
                #print mat_files[0]
                #print idx[0]
                mat_name=mat_dir+mat_files[idx[0][k]]
                matfile=sio.loadmat(mat_name)
                temp_data=matfile['hog_data']
                if np.shape(temp_data)[0]==0:
                    temp_data=np.zeros([TIME_STEP,4464])
                #print np.shape(temp_data)
                #print np.shape(hog_data)
                hog_data=np.concatenate([hog_data, temp_data],axis=0)

        mean = np.mean(hog_data,axis=0)
        if np.sum(mean)==0:
            mean=np.zeros([4464,],dtype='float64')
        mean_data[i,:]=mean

    return mean_data

def get_data_train(idx, data_list):
    train_data=np.zeros([np.size(data_list),TIME_STEP,4464])
    mat_dir=train_dir+label_list[idx]+'/'
    for i in xrange(np.size(data_list)):

        mat_name=mat_dir+str(data_list[i]+1)+'.mat'
        matfile=sio.loadmat(mat_name)
        hog_data=matfile['hog_data']

        #print np.shape(hog_data)
        #print hog_data.dtype
        if np.shape(hog_data)[0]==0:
            hog_data=np.zeros([TIME_STEP,4464])

        #print np.shape(hog_data)
        mean = np.mean(hog_data,axis=0)
        if np.sum(mean)==0:
            mean=np.zeros([4464,],dtype='float64')
            hog_data=np.zeros([TIME_STEP,4464],dtype='float64')
        #print np.shape(mean)
        #print mean.dtype
        #print np.sum(mean)
        mean=mean_data_train[data_list[i],:]
        hog_data -= mean

        hog_data=hog_data[1::20,:]
        temp_data=np.zeros([TIME_STEP,4464])
        gap=TIME_STEP-np.shape(hog_data)[0]
        if gap<0:
            temp_data=hog_data[-gap:,:]
        else:
            temp_data[gap:,:]=hog_data
        train_data[i,:,:]=temp_data
    return train_data

def get_data_test(idx):
    test_data=np.zeros([10,TIME_STEP,4464])
    mat_dir=test_dir+emo_list[idx]+'/'
    mat_files=os.listdir(mat_dir)
    for i in xrange(np.size(mat_files)):
        mat_name=mat_dir+mat_files[i]
        matfile=sio.loadmat(mat_name)
        hog_data=matfile['hog_data']

        #print np.shape(hog_data)
        #print hog_data.dtype
        if np.shape(hog_data)[0]==0:
            hog_data=np.zeros([TIME_STEP,4464])

        #print np.shape(hog_data)
        mean = np.mean(hog_data,axis=0)
        if np.sum(mean)==0:
            mean=np.zeros([4464,],dtype='float64')
            hog_data=np.zeros([TIME_STEP,4464],dtype='float64')
        #print np.shape(mean)
        #print mean.dtype
        #print np.sum(mean)
        mean=mean_data_test[IDs_val[idx,i],:]
        hog_data -= mean

        hog_data=hog_data[1::20,:]
        temp_data=np.zeros([TIME_STEP,4464])
        gap=TIME_STEP-np.shape(hog_data)[0]
        if gap<0:
            temp_data=hog_data[-gap:,:]
        else:
            temp_data[gap:,:]=hog_data
        test_data[i,:,:]=temp_data
    return test_data,mat_files


#%%
print('Setting params...')
NB_CLASSES=2
BATCH_SIZE=16
nb_epoch=150
RES=48
TIME_STEP=5
NB_TRAIN=30
#mean:300, max:500

#%%
print('Build model...')

def create_model():
    model = Sequential()

    model.add(TAGM(32,consume_less='gpu',return_sequences=False,input_shape=(TIME_STEP,4464)))
    #model.add(Dropout(0.5))
    #model.add(LSTM(32,return_sequences=False,input_shape=(TIME_STEP,4464)))
    #model.add(LSTM(128,return_sequences=False))
    #model.add(Flatten())
    #model.add(Dense(1024,activation='relu'))
    model.add(Dropout(0.25))
    #model.add(Dense(128,activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(NB_CLASSES))
    model.add(Activation('softmax'))

    return model
    

model=create_model()
plot(model, show_shapes=True, to_file='modelCNN.png')
sgd = SGD(lr=0.0001, decay=0.0005, momentum=0.9, nesterov=True)
ada=Adadelta()
model.compile(loss='categorical_crossentropy',
              optimizer=ada,
              metrics=['accuracy'])

model.save_weights('CL.hdf5')

filepath='weights.{epoch:03d}-{val_loss:.3f}.hdf5'
CheckPoint=ModelCheckpoint(filepath=filepath, monitor='val_acc', save_best_only=True)
StopPoint=EarlyStopping(patience=3, monitor='val_acc')



ground_truth=np.array([1,1,1,1,1,0,0,0,0,0])
final_acc=np.zeros(6)
pre_label_test=np.zeros([6,10])
pkl_data={}

X_pre_train=np.zeros([NB_TRAIN*12,75,4464])
y_train_pos=np.ones([NB_TRAIN*6,1])
y_train_neg=np.zeros([NB_TRAIN*6,1])

y_train=np.concatenate((y_train_pos,y_train_neg),axis=0)
Y_pre_train=np_utils.to_categorical(y_train)

#mean_data_train=get_mean_train()
#np.save('mean.npy',mean_data_train)

#mean_data_test=get_mean_test()
#np.save('mean_test.npy',mean_data_test)

mean_data_train=np.load('mean_train.npy')
mean_data_test=np.load('mean_test.npy')

'''
for j in range(6):
    
    x_train_pos=get_data_train(j,train_list)
    x_train_neg=get_data_train(j+6,train_list)

    X_pre_train[NB_TRAIN*j:NB_TRAIN*(j+1),:,:]=x_train_pos
    X_pre_train[NB_TRAIN*(j+6):NB_TRAIN*(j+7),:,:]=x_train_neg

model.fit(X_pre_train, Y_pre_train,
            nb_epoch=3,
            shuffle=True,
            batch_size=BATCH_SIZE,
            validation_split=0.2)
'''


for j in range(6):

    rand_list=np.random.permutation(40)
    train_list=rand_list[0:NB_TRAIN]
    val_list=rand_list[NB_TRAIN:]

    filepath=str(j)+'.hdf5'
    CheckPoint=ModelCheckpoint(filepath=filepath, monitor='val_acc', save_best_only=True)

    [X_test, video_names]=get_data_test(j)

    x_train_pos=get_data_train(j,train_list)
    y_train_pos=np.ones([NB_TRAIN,1])
    x_train_neg=get_data_train(j+6,train_list)
    y_train_neg=np.zeros([NB_TRAIN,1])

    x_train=np.concatenate((x_train_pos,x_train_neg),axis=0)
    y_train=np.concatenate((y_train_pos,y_train_neg),axis=0)

    X_train=x_train
    Y_train=np_utils.to_categorical(y_train)

    x_val_pos=get_data_train(j,val_list)
    y_val_pos=np.ones([40-NB_TRAIN,1])
    x_val_neg=get_data_train(j+6,val_list)
    y_val_neg=np.zeros([40-NB_TRAIN,1])


    x_val=np.concatenate((x_val_pos,x_val_neg),axis=0)
    y_val=np.concatenate((y_val_pos,y_val_neg),axis=0)

    X_val=x_val
    Y_val=np_utils.to_categorical(y_val)


    model.load_weights('CL.hdf5')
    #ccc_point=CCC_callback_ccc_reg(model, X_val, BATCH_SIZE, y_val, j+20)
    StopPoint=EarlyStopping(patience=1, monitor='val_acc')

    model.fit(X_train, Y_train,
              nb_epoch=20,
              shuffle=True,
              batch_size=BATCH_SIZE,
              validation_data=(X_val, Y_val),
              callbacks=[CheckPoint])

    weight_file=str(j)+'.hdf5'
    model.load_weights(weight_file)
    
    pre_label=model.predict_classes(X_test,batch_size=BATCH_SIZE,verbose=1)
    print pre_label    
    for k in xrange(np.size(video_names)):
        file_name=video_names[k][:-4]+'.mp4'
        print file_name
        pkl_data[file_name]=PREDICTED_LABEL[pre_label[k]]

    pre_label_test[j,:]=pre_label
    pre_label=model.predict_classes(X_val,batch_size=BATCH_SIZE,verbose=1)
    print pre_label
    [val_loss,val_acc]=model.evaluate(X_val,Y_val,batch_size=BATCH_SIZE,verbose=1)
    print val_acc
    final_acc[j]=val_acc
print final_acc
print np.mean(final_acc)
print pre_label_test

print pkl_data

print val_list

output = open('valid_prediction.pkl', 'wb')

# Pickle dictionary using protocol 0.
pickle.dump(pkl_data, output)

output.close()

