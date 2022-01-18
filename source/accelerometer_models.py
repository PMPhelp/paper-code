import tensorflow as tf
import numpy as np
from tensorflow import keras

from keras.models import Model
from keras.models import Sequential


def base_mlp_model(dim=60, num_classes=5):
	# create model
	model = Sequential()
	model.add(layers.Dense(dim, input_dim=dim, kernel_initializer='normal', activation='relu'))
	model.add(layers.Dense(num_classes, kernel_initializer='normal', activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def base_mlp_model2(dim=60, num_classes=5):
	# create model
	model = Sequential()
	model.add(layers.Dense(10, input_dim=dim, kernel_initializer='normal', activation='relu'))
	model.add(layers.Dense(10, kernel_initializer='normal', activation='relu'))
	model.add(layers.Dense(5, kernel_initializer='normal', activation='relu'))
	model.add(layers.Dense(num_classes, kernel_initializer='normal', activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

ef multilabel_mlp_model(dim=60, num_classes=45):
	# create model
	model = Sequential()
	model.add(layers.Dense(dim, input_dim=dim, kernel_initializer='normal', activation='relu'))
	model.add(layers.Dense(num_classes, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def multilabel_mlp_model2(dim=60, num_classes=45):
	# create model
	model = Sequential()
	model.add(layers.Dense(10, input_dim=dim, kernel_initializer='normal', activation='relu'))
	model.add(layers.Dense(10, kernel_initializer='normal', activation='relu'))
	model.add(layers.Dense(5, kernel_initializer='normal', activation='relu'))
	model.add(layers.Dense(num_classes, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def multilabel_mlp_model3(dim=60, num_classes=45):
  # create model
  # model = Sequential()
  limb_la = layers.Input(shape=(70,))
  dense_la = layers.Dense(10, kernel_initializer='normal', activation='relu')(limb_la)
  limb_ra = layers.Input(shape=(70,))
  dense_ra = layers.Dense(10, kernel_initializer='normal', activation='relu')(limb_ra)
  limb_lw = layers.Input(shape=(70,))
  dense_lw = layers.Dense(10, kernel_initializer='normal', activation='relu')(limb_lw)
  limb_rw = layers.Input(shape=(70,))
  dense_rw = layers.Dense(10, kernel_initializer='normal', activation='relu')(limb_rw)
  merge = keras_concatenate([dense_la, dense_ra, dense_lw, dense_rw])
  pre2_logs = layers.Dense(10, kernel_initializer='normal', activation='relu')(merge)
  pre1_logs = layers.Dense(5, kernel_initializer='normal', activation='relu')(pre2_logs)
  output = layers.Dense(num_classes, kernel_initializer='normal', activation='sigmoid')(pre1_logs)
  model = Model(inputs=[limb_la, limb_ra, limb_lw, limb_rw], outputs=output)
  # Compile model
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model