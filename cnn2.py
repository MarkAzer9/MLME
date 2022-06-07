import tensorflow as tf
from tensorflow.keras.layers import Conv2D
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import KFold
from utils.dataset import read_dataset,flatten

checkpoint_filepath = './cnn'
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

model=tf.keras.Sequential()
model.add(Conv2D(9,3,activation='relu', padding='same'))
model.add(Conv2D(13,5,activation='relu', padding='same'))
model.add(Conv2D(17,5,activation='relu', padding='same'))
model.add(Conv2D(19,5,activation='relu', padding='same'))
model.add(Conv2D(19,3,activation='relu', padding='same'))
model.add(Conv2D(17,5,activation='relu', padding='same'))
model.add(tf.keras.layers.AveragePooling2D(pool_size=(5, 5)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(99,activation='relu'))
model.add(tf.keras.layers.Dense(99,activation='relu'))
model.add(tf.keras.layers.Dense(10,activation='softmax'))
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(train_data,train_labels,callbacks=[callback], validation_data=(test_data,test_labels),epochs=200)

model.load_weights(checkpoint_filepath)

scores=model.evaluate(test_data,test_labels)