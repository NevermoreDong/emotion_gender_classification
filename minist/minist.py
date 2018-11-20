import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop
import keras
from model import mini_XCEPTION

batch_size = 128
num_classes = 10
epochs = 3
img_rows, img_cols = 28, 28

# 用于训练和测试的数据集，经过了筛选（清洗、数据样本顺序打乱）和分割（分割为训练和测试集）
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# 类别向量转为2分类矩阵   # 将整型标签转为onehot
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = mini_XCEPTION(input_shape=input_shape, num_classes=num_classes)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


model.summary()

model.fit(x_train,y_train,batch_size=batch_size,
          epochs=epochs,verbose=1,validation_data=(x_test,y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
