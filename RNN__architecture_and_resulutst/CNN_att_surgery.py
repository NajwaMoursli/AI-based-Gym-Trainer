import sys, os
#import json
#from keras.backend import tensorflow_backend as K
#from keras import Model
#import model_qual
#import data
import glob
from matplotlib import pyplot as plt
from statistics import mean
from math import *
import numpy as np
#from keras.utils import get_custom_objects
import scipy.io as sio
import keras.layers as kl
from keras import optimizers, losses, Model, Sequential
from keras.layers import Conv2DTranspose, Reshape, Dense, Activation, Flatten, Dropout, Convolution1D, MaxPooling1D, Input

import time
import shutil
from keras.optimizers import adam, SGD
from keras import regularizers
import keras.backend as K
from keras.optimizers import SGD

def affiche(history):
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()



def standardize(inputs):
	for i in range(inputs.shape[1]):
		mu = np.mean(inputs[:, i], axis=0, keepdims=True)
		sigma = np.std(inputs[:, i], axis=0, keepdims=True)
		if sigma != 0:
			inputs[:, i] = (inputs[:, i] - mu) / sigma
	return inputs


def read_data(mat_data, dims,fact_down):
	data = []

	for k in range(0, mat_data.shape[0]):
		temp = dict()
		names = mat_data[k]._fieldnames
		for n in names:
			elem = mat_data[k].__dict__[n]
			temp[n] = elem
		data.append(temp)
	X = []
	y = []
	for i in range(0, len(data)):
		x=data[i]['traj'][::fact_down, dims]
		x=standardize(x)
		X.append(x)
		y.append(data[i]['score_grs'])
	return X,y

def  split_data(X,y, ind_test):
	X_test = []
	y_test=[]
	X_train=[]
	y_train=[]
	for i in range(0, len(y)):
		if i in ind_test:
			X_test.append(X[i])
			y_test.append(y[i])
		else:
			X_train.append(X[i])
			y_train.append(y[i])
	X_test=np.array(X_test)
	X_train=np.array(X_train)
	return X_test , np.array(y_test), X_train, np.array(y_train)







def Aff_conf(conf_m):
	labels = ['-1', '0', '1']
	fig, ax = plt.subplots()
	# sum_matrix = np.sum(conf_matrix, axis=1)
	for i in range(3):
		for j in range(3):
			# conf_matrix[j, i] /= sum_matrix[j]
			c = conf_m[j, i]
			ax.text(i, j, str(round(c, 4)), va='center', ha='center')

	cax = ax.imshow(conf_m, cmap=plt.cm.Blues)

	fig.colorbar(cax)
	ax.set_xlabel('Predicted')
	ax.set_ylabel('Truth')
	ax.set_xticks(np.arange(len(labels)))
	ax.set_yticks(np.arange(len(labels)))

	ax.set_xticklabels(labels)
	ax.set_yticklabels(labels)
	plt.pause(0.1)


def indexe_base(size,nb_fold, y):
	nb_s = np.argsort(y)
	index=[]
	for i in range(nb_fold):
		temp=np.array(range(i,size,4))
		ind=np.zeros(len(temp))
		for j in range(len(temp)):
			ind[j]=nb_s[temp[j]]
		index.append(ind)

	return index


def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    TIME_STEPS = int(inputs.shape[1])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
#    if SINGLE_ATTENTION_VECTOR:
#        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
#        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
#    output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')

    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul


def define_model(dim1, dim2):

	inputs= Input(shape=(dim1,dim2))
	x= inputs
	x = attention_3d_block(x)

	x=Convolution1D(32,5, activation='relu',padding='same')(x)
	x=MaxPooling1D(pool_size=5, strides=3,padding='same')(x)

	x=Convolution1D(64,3, activation='relu',padding='same')(x)
	x=MaxPooling1D(pool_size=3, strides=3,padding='same')(x)

	x=Convolution1D(64,3, activation='relu',padding='same')(x)
	x=MaxPooling1D(pool_size=3, strides=3,padding='same')(x)

	x=Convolution1D(64,3, activation='relu',padding='same')(x)
	x=MaxPooling1D(pool_size=3, strides=3,padding='same')(x)

	x=Convolution1D(64,3, activation='relu',padding='same')(x)
	x=MaxPooling1D(pool_size=3, strides=3,padding='same')(x)

	x=Convolution1D(64,3, activation='relu',padding='same')(x)
	x=MaxPooling1D(pool_size=3, strides=3,padding='same')(x)

	x=Flatten()(x)
	x=Dense(100, activation='relu')(x)
	x=Dropout(0.5)(x)
	outputs=Dense(1, activation='linear')(x)
	model = Model(inputs, outputs)
	model.summary()

	return model



def padd(X):
	length = 0
	X1=[]
	for i in range(len(X)):
		if X[i].shape[0]>length:
			length=X[i].shape[0]
	for i in range(len(X)):
		if X[i].shape[0] < length :
			padding = length - X[i].shape[0]
			pad = np.zeros((padding, X[i].shape[1]))
			return_X = np.append(X[i], pad, axis=0)
		else:
			return_X = X[i]
		X1.append(return_X)
	X1 = np.array(X1)
	return X1


################################################
################################################
################################################


dims = [38, 39, 40, 50, 51, 52, 56, 57, 58, 59, 69, 70, 71, 75]
fact_down=1

name_data, G = 'C:/Users/Catherine/Desktop/tensorflow/Mégane/Data/Gestes_Suturing.mat', 'fd1Sutu/'
data_file = sio.loadmat(name_data, struct_as_record=False, squeeze_me=True)
X, y = read_data(data_file['Gestes'], dims,fact_down)
del data_file


nb_epochs = 40
aff=0
lr=0.0001

X = padd(X)
index=indexe_base(len(y),1,y)


#ind_test=index[0]
ind_test = [13., 27., 12., 15., 26., 38., 16., 37., 31., 22.]
X_test , y_test, X_train, y_train = split_data(X,y, ind_test)

model = define_model(X_train.shape[1], X_train.shape[2])

sgd1=SGD(lr=lr, momentum=0.0)
model.compile(loss='MSE',
              optimizer=sgd1)


#history =model.fit(X_train, y_train, epochs=nb_epochs, verbose=1,validation_data=(X_test, y_test))


affiche(history)
score_train = model.predict(X_train)
RMSE_train=np.linalg.norm(score_train-y_train)/len(y_train)
score_test = model.predict(X_test)
RMSE_test=np.linalg.norm(score_test-y_test)/len(y_test)
print("RMSE train", RMSE_train)
print("RMSE test", RMSE_test)


test=np.reshape(X_test[0,:,:],(1,9012, 14))
activations = get_activations(model,test,print_shape_only=True,layer_name='attention_vec')
attention_vector = np.mean(activations[0], axis=2).squeeze()



import matplotlib.pyplot as plt
plt.plot(np.reshape(activations[0], (9012, 14)))
plt.show()




