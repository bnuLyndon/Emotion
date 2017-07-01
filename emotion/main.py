#coding:utf-8
"""
Author: Lyndon Lee
"""

from keras.layers import Input, LSTM, RepeatVector, merge
from keras.models import Sequential, Model
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
val_dir='/usr0/home/liandonl/Datasets/CL_val/'
test_dir='/usr0/home/liandonl/Datasets/CL_test/'

train_dir_LM='/usr0/home/liandonl/Datasets/CL_LandMarks_train/'
val_dir_LM='/usr0/home/liandonl/Datasets/CL_LandMarks_val/'
test_dir_LM='/usr0/home/liandonl/Datasets/CL_LandMarks_test/'

label_list=['N2H','N2S','N2D','N2A','N2C','N2Sur','S2N2H','H2N2S','H2N2D','H2N2A','H2N2C','D2N2Sur']
#emo_list=['ANGER','CONTENTMENT','DISGUST','HAPPINESS','SADNESS','SURPRISE']
emo_list=['HAPPINESS','SADNESS','DISGUST','ANGER','CONTENTMENT','SURPRISE']
PREDICTED_LABEL=['fake','true']

def get_LM_train(idx):
    train_data=np.zeros([40,TIME_STEP_LM,40])
    mat_dir=train_dir_LM+label_list[idx]+'/'
    for i in xrange(40):

        mat_name=mat_dir+str(i+1)+'.mat'
        matfile=sio.loadmat(mat_name)
        hog_data=matfile['shape_params']

        #print np.shape(hog_data)
        #print hog_data.dtype
        if np.shape(hog_data)[0]==0:
            hog_data=np.zeros([TIME_STEP_LM,40])

        #print np.shape(hog_data)
        mean = np.mean(hog_data,axis=0)
        if np.sum(mean)==0:
            mean=np.zeros([40,],dtype='float64')
            hog_data=np.zeros([TIME_STEP_LM,40],dtype='float64')
        #print np.shape(mean)
        #print mean.dtype
        #print np.sum(mean)
        #mean=mean_data_train[data_list[i],:]
        hog_data -= mean

        hog_data=hog_data[1::INT_LM,:]
        temp_data=np.zeros([TIME_STEP_LM,40])
        gap=TIME_STEP_LM-np.shape(hog_data)[0]
        if gap<0:
            temp_data=hog_data[-gap:,:]
        else:
            temp_data[gap:,:]=hog_data
        train_data[i,:,:]=temp_data
    return train_data

def get_LM_train_aug(idx, data_list):
    train_data=np.zeros([np.size(data_list)*NB_AUG,TIME_STEP_LM,40])
    mat_dir=train_dir_LM+label_list[idx]+'/'
    for i in xrange(np.size(data_list)):

        mat_name=mat_dir+str(data_list[i]+1)+'.mat'
        matfile=sio.loadmat(mat_name)
        hog_data=matfile['shape_params']

        #print np.shape(hog_data)
        #print hog_data.dtype
        if np.shape(hog_data)[0]==0:
            hog_data=np.zeros([TIME_STEP_LM*2,40])

        #print np.shape(hog_data)
        mean = np.mean(hog_data,axis=0)
        if np.sum(mean)==0:
            mean=np.zeros([40,],dtype='float64')
            hog_data=np.zeros([TIME_STEP_LM*2,40],dtype='float64')
        #print np.shape(mean)
        #print mean.dtype
        #print np.sum(mean)
        #mean=mean_data_train[data_list[i],:]
        hog_data -= mean

        temp_rand_list=np.random.permutation(INT_LM)
        for j in xrange(NB_AUG):
            temp_hog=hog_data[temp_rand_list[j]::INT_LM,:]
            temp_data=np.zeros([TIME_STEP_LM,40])
            gap=TIME_STEP_LM-np.shape(temp_hog)[0]
            if gap<0:
                temp_data=temp_hog[-gap:,:]
            else:
                temp_data[gap:,:]=temp_hog
    
            train_data[i*NB_AUG+j,:,:]=temp_data
        #print np.sort(temp_rand_list[0:TIME_STEP])
        
    return train_data

def get_LM_val(idx):
    test_data=np.zeros([10,TIME_STEP_LM,40])
    mat_dir=val_dir_LM+emo_list[idx]+'/'
    mat_files=os.listdir(mat_dir)
    for i in xrange(np.size(mat_files)):
        mat_name=mat_dir+mat_files[i]
        matfile=sio.loadmat(mat_name)
        hog_data=matfile['shape_params']

        #print np.shape(hog_data)
        #print hog_data.dtype
        if np.shape(hog_data)[0]==0:
            hog_data=np.zeros([TIME_STEP_LM,40])

        #print np.shape(hog_data)
        mean = np.mean(hog_data,axis=0)
        if np.sum(mean)==0:
            mean=np.zeros([40,],dtype='float64')
            hog_data=np.zeros([TIME_STEP_LM,40],dtype='float64')
        #print np.shape(mean)
        #print mean.dtype
        #print np.sum(mean)
        #mean=mean_data_test[IDs_val[idx,i],:]
        hog_data -= mean

        hog_data=hog_data[1::INT_LM,:]
        temp_data=np.zeros([TIME_STEP_LM,40])
        gap=TIME_STEP_LM-np.shape(hog_data)[0]
        if gap<0:
            temp_data=hog_data[-gap:,:]
        else:
            temp_data[gap:,:]=hog_data
        test_data[i,:,:]=temp_data
    return test_data,mat_files

def get_LM_test(idx):
    test_data=np.zeros([10,TIME_STEP_LM,40])
    mat_dir=test_dir_LM+emo_list[idx]+'/'
    mat_files=os.listdir(mat_dir)
    for i in xrange(np.size(mat_files)):
        mat_name=mat_dir+mat_files[i]
        matfile=sio.loadmat(mat_name)
        hog_data=matfile['shape_params']

        #print np.shape(hog_data)
        #print hog_data.dtype
        if np.shape(hog_data)[0]==0:
            hog_data=np.zeros([TIME_STEP_LM,40])

        #print np.shape(hog_data)
        mean = np.mean(hog_data,axis=0)
        if np.sum(mean)==0:
            mean=np.zeros([40,],dtype='float64')
            hog_data=np.zeros([TIME_STEP_LM,40],dtype='float64')
        #print np.shape(mean)
        #print mean.dtype
        #print np.sum(mean)
        #mean=mean_data_test[IDs_val[idx,i],:]
        hog_data -= mean

        hog_data=hog_data[1::INT_LM,:]
        temp_data=np.zeros([TIME_STEP_LM,40])
        gap=TIME_STEP_LM-np.shape(hog_data)[0]
        if gap<0:
            temp_data=hog_data[-gap:,:]
        else:
            temp_data[gap:,:]=hog_data
        test_data[i,:,:]=temp_data
    return test_data,mat_files

def get_mean_train():
    mean_data=np.zeros([40,INPUT_DIM])
    for i in xrange(40):
        hog_data=np.zeros([0,INPUT_DIM],dtype='float64')
        for j in xrange(6):
            mat_dir=train_dir+label_list[j]+'/'
            mat_name=mat_dir+str(i+1)+'.mat'
            matfile=sio.loadmat(mat_name)
            temp_data=matfile['hog_data']
            if np.shape(temp_data)[0]==0:
                temp_data=np.zeros([TIME_STEP,INPUT_DIM])
            #print np.shape(temp_data)
            #print np.shape(hog_data)
            hog_data=np.concatenate([hog_data, temp_data],axis=0)

        mean = np.mean(hog_data,axis=0)
        if np.sum(mean)==0:
            mean=np.zeros([INPUT_DIM,],dtype='float64')
        mean_data[i,:]=mean

    return mean_data

def get_mean_val():
    mean_data=np.zeros([5,INPUT_DIM])
    for i in xrange(5):
        hog_data=np.zeros([0,INPUT_DIM],dtype='float64')
        for j in xrange(6):
            mat_dir=val_dir+emo_list[j]+'/'
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
                    temp_data=np.zeros([TIME_STEP,INPUT_DIM])
                #print np.shape(temp_data)
                #print np.shape(hog_data)
                hog_data=np.concatenate([hog_data, temp_data],axis=0)

        mean = np.mean(hog_data,axis=0)
        if np.sum(mean)==0:
            mean=np.zeros([INPUT_DIM,],dtype='float64')
        mean_data[i,:]=mean

    return mean_data

def get_mean_test():
    mean_data=np.zeros([5,INPUT_DIM])
    for i in xrange(5):
        hog_data=np.zeros([0,INPUT_DIM],dtype='float64')
        for j in xrange(6):
            mat_dir=test_dir+emo_list[j]+'/'
            mat_files=os.listdir(mat_dir)
            #print np.where(IDs_val[j,:]==i)
            idx=np.where(IDs_test[j,:]==i)
            for k in xrange(2):
                #print mat_files[0]
                #print idx[0]
                mat_name=mat_dir+mat_files[idx[0][k]]
                matfile=sio.loadmat(mat_name)
                temp_data=matfile['hog_data']
                if np.shape(temp_data)[0]==0:
                    temp_data=np.zeros([TIME_STEP,INPUT_DIM])
                #print np.shape(temp_data)
                #print np.shape(hog_data)
                hog_data=np.concatenate([hog_data, temp_data],axis=0)

        mean = np.mean(hog_data,axis=0)
        if np.sum(mean)==0:
            mean=np.zeros([INPUT_DIM,],dtype='float64')
        mean_data[i,:]=mean

    return mean_data

def get_data_train(idx, data_list):
    train_data=np.zeros([np.size(data_list),TIME_STEP,INPUT_DIM])
    mat_dir=train_dir+label_list[idx]+'/'
    for i in xrange(np.size(data_list)):

        mat_name=mat_dir+str(data_list[i]+1)+'.mat'
        matfile=sio.loadmat(mat_name)
        hog_data=matfile['hog_data']

        #print np.shape(hog_data)
        #print hog_data.dtype
        if np.shape(hog_data)[0]==0:
            hog_data=np.zeros([TIME_STEP,INPUT_DIM])

        #print np.shape(hog_data)
        mean = np.mean(hog_data,axis=0)
        if np.sum(mean)==0:
            mean=np.zeros([INPUT_DIM,],dtype='float64')
            hog_data=np.zeros([TIME_STEP,INPUT_DIM],dtype='float64')
        #print np.shape(mean)
        #print mean.dtype
        #print np.sum(mean)
        mean=mean_data_train[data_list[i],:]
        hog_data -= mean

        hog_data=hog_data[1::INT,:]
        temp_data=np.zeros([TIME_STEP,INPUT_DIM])
        gap=TIME_STEP-np.shape(hog_data)[0]
        if gap<0:
            temp_data=hog_data[-gap:,:]
        else:
            temp_data[gap:,:]=hog_data
        train_data[i,:,:]=temp_data
    return train_data

def get_data_val(idx):
    test_data=np.zeros([10,TIME_STEP,INPUT_DIM])
    mat_dir=val_dir+emo_list[idx]+'/'
    mat_files=os.listdir(mat_dir)
    for i in xrange(np.size(mat_files)):
        mat_name=mat_dir+mat_files[i]
        matfile=sio.loadmat(mat_name)
        hog_data=matfile['hog_data']

        #print np.shape(hog_data)
        #print hog_data.dtype
        if np.shape(hog_data)[0]==0:
            hog_data=np.zeros([TIME_STEP,INPUT_DIM])

        #print np.shape(hog_data)
        mean = np.mean(hog_data,axis=0)
        if np.sum(mean)==0:
            mean=np.zeros([INPUT_DIM,],dtype='float64')
            hog_data=np.zeros([TIME_STEP,INPUT_DIM],dtype='float64')
        #print np.shape(mean)
        #print mean.dtype
        #print np.sum(mean)
        mean=mean_data_val[IDs_val[idx,i],:]
        hog_data -= mean

        hog_data=hog_data[1::INT,:]
        temp_data=np.zeros([TIME_STEP,INPUT_DIM])
        gap=TIME_STEP-np.shape(hog_data)[0]
        if gap<0:
            temp_data=hog_data[-gap:,:]
        else:
            temp_data[gap:,:]=hog_data
        test_data[i,:,:]=temp_data
    return test_data,mat_files

def get_data_test(idx):
    test_data=np.zeros([10,TIME_STEP,INPUT_DIM])
    mat_dir=test_dir+emo_list[idx]+'/'
    mat_files=os.listdir(mat_dir)
    for i in xrange(np.size(mat_files)):
        mat_name=mat_dir+mat_files[i]
        matfile=sio.loadmat(mat_name)
        hog_data=matfile['hog_data']

        #print np.shape(hog_data)
        #print hog_data.dtype
        if np.shape(hog_data)[0]==0:
            hog_data=np.zeros([TIME_STEP,INPUT_DIM])

        #print np.shape(hog_data)
        mean = np.mean(hog_data,axis=0)
        if np.sum(mean)==0:
            mean=np.zeros([INPUT_DIM,],dtype='float64')
            hog_data=np.zeros([TIME_STEP,INPUT_DIM],dtype='float64')
        #print np.shape(mean)
        #print mean.dtype
        #print np.sum(mean)
        mean=mean_data_test[IDs_val[idx,i],:]
        hog_data -= mean

        hog_data=hog_data[1::INT,:]
        temp_data=np.zeros([TIME_STEP,INPUT_DIM])
        gap=TIME_STEP-np.shape(hog_data)[0]
        if gap<0:
            temp_data=hog_data[-gap:,:]
        else:
            temp_data[gap:,:]=hog_data
        test_data[i,:,:]=temp_data
    return test_data,mat_files

def get_data_train_aug(idx, data_list):
    train_data=np.zeros([np.size(data_list)*NB_AUG,TIME_STEP,INPUT_DIM])
    mat_dir=train_dir+label_list[idx]+'/'
    for i in xrange(np.size(data_list)):

        mat_name=mat_dir+str(data_list[i]+1)+'.mat'
        matfile=sio.loadmat(mat_name)
        hog_data=matfile['hog_data']

        #print np.shape(hog_data)
        #print hog_data.dtype
        if np.shape(hog_data)[0]==0:
            hog_data=np.zeros([TIME_STEP*2,INPUT_DIM])

        #print np.shape(hog_data)
        mean = np.mean(hog_data,axis=0)
        if np.sum(mean)==0:
            mean=np.zeros([INPUT_DIM,],dtype='float64')
            hog_data=np.zeros([TIME_STEP*2,INPUT_DIM],dtype='float64')
        #print np.shape(mean)
        #print mean.dtype
        #print np.sum(mean)
        mean=mean_data_train[data_list[i],:]
        hog_data -= mean

        temp_rand_list=np.random.permutation(INT)
        for j in xrange(NB_AUG):
            temp_hog=hog_data[temp_rand_list[j]::INT,:]
            temp_data=np.zeros([TIME_STEP,INPUT_DIM])
            gap=TIME_STEP-np.shape(temp_hog)[0]
            if gap<0:
                temp_data=temp_hog[-gap:,:]
            else:
                temp_data[gap:,:]=temp_hog
    
            train_data[i*NB_AUG+j,:,:]=temp_data
        #print np.sort(temp_rand_list[0:TIME_STEP])
        
    return train_data

#%%
print('Setting params...')
NB_CLASSES=2
BATCH_SIZE=16
nb_epoch=150
RES=48
TIME_STEP=10
TIME_STEP_LM=10
NB_TRAIN=32
NB_AUG=5
INT=20
INT_LM=20
NB_ITER=1
INPUT_DIM = 4464
#mean:300, max:500

IDs_val=[
    [1,3,2,2,0,0,4,3,4,1],
    [2,4,1,0,3,4,0,1,2,3],
    [1,3,2,2,4,1,4,3,0,0],
    [0,3,3,2,0,1,4,2,1,4],
    [3,2,2,1,3,1,4,0,0,4],
    [0,1,3,2,4,3,0,2,4,1],
]

IDs_val=np.array(IDs_val)

IDs_test=[
    [1,2,3,3,1,4,2,0,0,4],
    [3,0,1,1,4,3,4,2,0,2],
    [2,3,2,0,0,4,4,1,1,3],
    [0,0,2,2,3,1,1,4,4,3],
    [1,3,1,4,2,4,3,2,0,0],
    [0,0,3,1,4,4,2,2,3,1],
]

IDs_test=np.array(IDs_test)

#%%
print('Build model...')

def create_model():
    inputs = Input(shape=(TIME_STEP,INPUT_DIM))

    x=TAGM(32,consume_less='gpu',return_sequences=False)(inputs)
    x=Dropout(0.25)(x)

    inputs_LM = Input(shape=(TIME_STEP_LM, 40))
    encoded = LSTM(1024, dropout_U=0.5)(inputs_LM)

    decoded = RepeatVector(TIME_STEP_LM)(encoded)
    decoded = LSTM(40, return_sequences=True, dropout_U=0.5)(decoded)
    
    dense=Dense(32,activation='relu')(encoded)

    merged=merge([x,dense],mode='concat')

    output=Dense(NB_CLASSES,activation='softmax')(merged)

    model=Model([inputs, inputs_LM], output)
    sequence_autoencoder = Model(inputs_LM, decoded)
    #encoder = Model(inputs_LM, dense)

    return model, sequence_autoencoder
    

[model, sequence_autoencoder]=create_model()
plot(model, show_shapes=True, to_file='modelCNN.png')
sgd = SGD(lr=0.0001, decay=0.0005, momentum=0.9, nesterov=True)
ada=Adadelta()
sequence_autoencoder.compile(loss='mean_absolute_error',
                            optimizer=ada)

train_lists=np.zeros([6,NB_TRAIN],dtype=int)
val_lists=np.zeros([6,40-NB_TRAIN],dtype=int)
for i in xrange(6):
    rand_list=np.random.permutation(40)
    train_lists[i,:]=rand_list[0:NB_TRAIN]
    val_lists[i,:]=rand_list[NB_TRAIN:]
print train_lists
print val_lists
ground_truth=np.array([1,1,1,1,1,0,0,0,0,0])
final_acc=np.zeros([NB_ITER,6])
pred_label_test=np.zeros([6,10])
pkl_data={}

#mean_data_train=get_mean_train()
#np.save('mean.npy',mean_data_train)

#mean_data_test=get_mean_test()
#np.save('mean_test.npy',mean_data_test)

mean_data_train=np.load('/usr0/home/liandonl/Documents/python/ChaLearn_old/mean_train.npy')
mean_data_val=np.load('/usr0/home/liandonl/Documents/python/ChaLearn_old/mean_test.npy')


X_pre_train=np.zeros([40*12,TIME_STEP_LM,40])

for j in range(6):
    
    x_train_pos=get_LM_train(j)
    x_train_neg=get_LM_train(j+6)

    X_pre_train[40*j:40*(j+1),:,:]=x_train_pos
    X_pre_train[40*(j+6):40*(j+7),:,:]=x_train_neg

sequence_autoencoder.fit(X_pre_train, X_pre_train[:,::-1,:],
            nb_epoch=20,
            shuffle=True,
            batch_size=BATCH_SIZE)

sequence_autoencoder.save_weights('CL_ae.hdf5')
y_ground=np.load('val_ground.npy')

for i in xrange(NB_ITER):

    K.clear_session()
    ada=Adadelta()
    [model, sequence_autoencoder]=create_model()
    model.compile(loss='categorical_crossentropy',
                  optimizer=ada,
                  metrics=['accuracy'])

    sequence_autoencoder.load_weights('CL_ae.hdf5')
    model.save_weights('CL.hdf5')

    for j in xrange(6):

        train_list=train_lists[j,:]
        val_list=val_lists[j,:]

        filepath=str(i)+str(j)+'.hdf5'
        CheckPoint=ModelCheckpoint(filepath=filepath, monitor='val_acc', save_best_only=True)
        
        #HOG
        #[X_test, video_names]=get_data_val(j)

        x_train_pos=get_data_train_aug(j,train_list)
        y_train_pos=np.ones([NB_TRAIN*NB_AUG,1])
        x_train_neg=get_data_train_aug(j+6,train_list)
        y_train_neg=np.zeros([NB_TRAIN*NB_AUG,1])

        x_train=np.concatenate((x_train_pos,x_train_neg),axis=0)
        y_train=np.concatenate((y_train_pos,y_train_neg),axis=0)

        X_train=x_train
        Y_train=np_utils.to_categorical(y_train)

        x_val_pos=get_data_train_aug(j,val_list)
        y_val_pos=np.ones([(40-NB_TRAIN)*NB_AUG,1])
        x_val_neg=get_data_train_aug(j+6,val_list)
        y_val_neg=np.zeros([(40-NB_TRAIN)*NB_AUG,1])


        x_val=np.concatenate((x_val_pos,x_val_neg),axis=0)
        y_val=np.concatenate((y_val_pos,y_val_neg),axis=0)

        X_val=x_val
        Y_val=np_utils.to_categorical(y_val)

        #LandMarks
        #[X_test_LM, temp1]=get_LM_val(j)

        x_train_pos=get_LM_train_aug(j,train_list)
        x_train_neg=get_LM_train_aug(j+6,train_list)

        x_train=np.concatenate((x_train_pos,x_train_neg),axis=0)
        X_train_LM=x_train

        x_val_pos=get_LM_train_aug(j,val_list)
        x_val_neg=get_LM_train_aug(j+6,val_list)

        x_val=np.concatenate((x_val_pos,x_val_neg),axis=0)
        X_val_LM=x_val


        model.load_weights('CL.hdf5')
        #ccc_point=CCC_callback_ccc_reg(model, X_val, BATCH_SIZE, y_val, j+20)
        StopPoint=EarlyStopping(patience=3, monitor='val_acc')

        model.fit([X_train,X_train_LM], Y_train,
                  nb_epoch=20,
                  shuffle=True,
                  batch_size=BATCH_SIZE,
                  validation_data=([X_val,X_val_LM], Y_val),
                  callbacks=[StopPoint, CheckPoint])

        weight_file=str(i)+str(j)+'.hdf5'
        model.load_weights(weight_file)

        pred_label=model.predict([X_val,X_val_LM],batch_size=BATCH_SIZE,verbose=1)
        pred_label=np.argmax(pred_label,axis=1)
        print pred_label
        [val_loss,val_acc]=model.evaluate([X_val,X_val_LM],Y_val,batch_size=BATCH_SIZE,verbose=1)
        print val_acc
        final_acc[i,j]=val_acc

eval_val=np.zeros(6)

for j in range(6):
    
    val_list=val_lists[j,:]

    temp=np.argmax(final_acc[:,j])
    filepath=str(temp)+str(j)+'.hdf5'
    print filepath
    model.load_weights(filepath)
    new_file_path=str(j+100)+'.hdf5'
    model.save_weights(new_file_path)

    #HOG
    x_val_pos=get_data_train_aug(j,val_list)
    y_val_pos=np.ones([(40-NB_TRAIN)*NB_AUG,1])
    x_val_neg=get_data_train_aug(j+6,val_list)
    y_val_neg=np.zeros([(40-NB_TRAIN)*NB_AUG,1])

    x_val=np.concatenate((x_val_pos,x_val_neg),axis=0)
    y_val=np.concatenate((y_val_pos,y_val_neg),axis=0)

    X_val=x_val
    Y_val=np_utils.to_categorical(y_val)

    #LandMarks
    x_val_pos=get_LM_train_aug(j,val_list)
    x_val_neg=get_LM_train_aug(j+6,val_list)

    x_val=np.concatenate((x_val_pos,x_val_neg),axis=0)
    X_val_LM=x_val

    [val_loss,val_acc]=model.evaluate([X_val,X_val_LM],Y_val,batch_size=BATCH_SIZE,verbose=1)
    eval_val[j]=val_acc
    
    [X_test, video_names]=get_data_val(j)
    [X_test_LM, temp]=get_LM_val(j)
    #pred_label=model.predict_classes(X_test,batch_size=BATCH_SIZE,verbose=1) 
    pred_label=model.predict([X_test,X_test_LM],batch_size=BATCH_SIZE,verbose=1)
    pred_label=np.argmax(pred_label,axis=1)

    print pred_label    
    for k in xrange(np.size(video_names)):
        file_name=video_names[k][:-4]+'.mp4'
        print file_name
        pkl_data[file_name]=PREDICTED_LABEL[pred_label[k]]
    pred_label_test[j,:]=pred_label

print eval_val
print np.mean(eval_val)
print final_acc
print np.mean(final_acc)
print pred_label_test

output = open('valid_prediction.pkl', 'wb')

# Pickle dictionary using protocol 0.
pickle.dump(pkl_data, output)

output.close()

test_acc=np.zeros(6)
y_ground=np.load('val_ground.npy')
print y_ground
for i in xrange(6):
    temp=y_ground[i,:]-pred_label_test[i,:]
    #print temp
    test_acc[i]=np.sum(temp==0)/10.
print test_acc
print np.sum(test_acc)/6


pkl_data={}
test_dir='/usr0/home/liandonl/Datasets/CL_test/'
mean_data_test=get_mean_test()
pred_label_test=np.zeros([6,10])
for j in range(6):
    model_name=str(j+100)+'.hdf5'
    model.load_weights(model_name)
    [X_test, video_names]=get_data_test(j)
    [X_test_LM, temp]=get_LM_test(j)
    #pred_label=model.predict_classes(X_test,batch_size=BATCH_SIZE,verbose=1) 
    pred_label=model.predict([X_test,X_test_LM],batch_size=BATCH_SIZE,verbose=1)
    pred_label=np.argmax(pred_label,axis=1)
    print pred_label      
    for k in xrange(np.size(video_names)):
        file_name=video_names[k][:-4]+'.mp4'
        print file_name
        pkl_data[file_name]=PREDICTED_LABEL[pred_label[k]]
    pred_label_test[j,:]=pred_label

print pred_label_test

output = open('test_prediction.pkl', 'wb')

# Pickle dictionary using protocol 0.
pickle.dump(pkl_data, output)

output.close()
