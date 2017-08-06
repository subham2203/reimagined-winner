import numpy
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K
from keras.layers import Conv2D,Conv1D,LSTM
from keras.layers.core import Reshape
from keras.layers.pooling import MaxPooling1D
from Fractional_MAXPOOL import FractionalPooling2D
from keras.callbacks import ModelCheckpoint
from keras.layers.advanced_activations import LeakyReLU
K.set_image_dim_ordering('tf')

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train[0:49984]
y_train = y_train[0:49984]
X_test = X_test[0:9984]
y_test = y_test[0:9984]

# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# Create the model
model = Sequential()
# Block 1
model.add(Conv2D(64, (3, 3), batch_input_shape=(64, 32, 32, 3), padding='same'))
model.add(LeakyReLU(alpha = 0.3))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(LeakyReLU(alpha = 0.3))
model.add(FractionalPooling2D(pool_ratio=(1, 1.6, 1.6, 1),pseudo_random = True,overlap=True))
# Block 2
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(LeakyReLU(alpha = 0.3))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(LeakyReLU(alpha = 0.3))
model.add(FractionalPooling2D(pool_ratio=(1, 1.25, 1.25, 1),pseudo_random = True,overlap=True))
# Block 3
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(LeakyReLU(alpha = 0.3))
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(LeakyReLU(alpha = 0.3))
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(LeakyReLU(alpha = 0.3))
model.add(FractionalPooling2D(pool_ratio=(1, 1.6, 1.6, 1),pseudo_random = True,overlap=True))
# Block 4
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(LeakyReLU(alpha = 0.3))
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(LeakyReLU(alpha = 0.3))
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(LeakyReLU(alpha = 0.3))
model.add(FractionalPooling2D(pool_ratio=(1, 1.25, 1.25, 1),pseudo_random = True,overlap=True))
# Block 5
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(LeakyReLU(alpha = 0.3))
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(LeakyReLU(alpha = 0.3))
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(LeakyReLU(alpha = 0.3))
model.add(FractionalPooling2D(pool_ratio=(1, 1.6, 1.6, 1),pseudo_random = True,overlap=True))
# Block 6
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(LeakyReLU(alpha = 0.3))
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(LeakyReLU(alpha = 0.3))
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(LeakyReLU(alpha = 0.3))
model.add(FractionalPooling2D(pool_ratio=(1, 1.25, 1.25, 1),pseudo_random = True,overlap=True))
model.add(Reshape((16,512)))
# fc layer_1
model.add(Dense(4096, kernel_constraint=maxnorm(3)))
model.add(LeakyReLU(alpha = 0.3))
# fc_layer_2
model.add(Dense(4096, kernel_constraint=maxnorm(3)))
model.add(LeakyReLU(alpha = 0.3))

model.add(Dense(num_classes, activation='softmax'))

opt = keras.optimizers.Adadelta(0.1,decay=1e-4)

model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
print(model.summary())

checkpoint = ModelCheckpoint('Model.hdf5', monitor='val_loss', save_best_only = True, verbose=1, mode='min')

callbacks_list = [checkpoint]
#model.load_weights('Model.hdf5')
epochs = 1000
model.fit(X_train, y_train, validation_data = [X_test,y_test], nb_epoch=epochs, batch_size=64, callbacks=callbacks_list)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
