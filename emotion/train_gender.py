from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.callbacks import EarlyStopping,ReduceLROnPlateau

from Manger import DataManager
from Xception_CNN import Xception
from data_augmentation import ImageGenerator
from Manger import split_imdb_data

# parameters
batch_size = 32
num_epochs = 10
validation = .2
do_random_crop = False
patience = 100
num_classes = 2
dataset_name = 'imdb'
input_shape = (64,64,1)
if input_shape[2] == 1:
    grayscale = True
images_path = 'E:/Programming/project/face_classification/face_classification-master/datasets/imdb_crop'
log_file_path = ''
trained_models_path = ''





























