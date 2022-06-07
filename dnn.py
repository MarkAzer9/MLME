import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from utils.dataset import read_dataset,flatten
#from keras.utils import plot_model
checkpoint_filepath = './dnn'
callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
dim=48

test_data,test_labels=read_dataset('test',dim)
train_data,train_labels=read_dataset('train',dim)
test_data=test_data.reshape(-1,dim,dim,1)
train_data=train_data.reshape(-1,dim,dim,1)

dropout=0.45

model=tf.keras.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(2304,activation='relu'))
model.add(tf.keras.layers.Dropout(dropout))
model.add(tf.keras.layers.Dense(2304,activation='relu'))
model.add(tf.keras.layers.Dropout(dropout))
model.add(tf.keras.layers.Dense(999,activation='relu'))
model.add(tf.keras.layers.Dropout(dropout))
model.add(tf.keras.layers.Dense(10,activation='softmax'))
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001),callbacks=[callback], loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_data,train_labels,validation_data=(test_data,test_labels),epochs=1000)
#plot_model(model, to_file='dnn_plot.png', show_shapes=True)