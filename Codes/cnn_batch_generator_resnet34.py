import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from keras.utils import np_utils
from keras.utils import multi_gpu_model
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, cohen_kappa_score
from sklearn.model_selection import train_test_split
from skll.metrics import kappa

import keras
from utils.datautil import DataGenerator
from ml import resnet


np.random.seed(1367)

root_dir = '../Data/sample/'

def split_data(X, y, test_data_size):
    """
    Split data into test and training datasets.

    INPUT
        X: NumPy array of arrays
        y: Pandas series, which are the labels for input array X
        test_data_size: size of test/train split. Value from 0 to 1

    OUPUT
        Four arrays: X_train, X_test, y_train, and y_test
    """
    return train_test_split(X, y, test_size=test_data_size, random_state=42)


def reshape_data(arr, img_rows, img_cols, channels):
    """
    Reshapes the data into format for CNN.

    INPUT
        arr: Array of NumPy arrays.
        img_rows: Image height
        img_cols: Image width
        channels: Specify if the image is grayscale (1) or RGB (3)

    OUTPUT
        Reshaped array of NumPy arrays.
    """
    return arr.reshape(arr.shape[0], img_rows, img_cols, channels)


def cnn_model(X_training, X_validation, y_training, y_validation, channels, nb_epoch, batch_size, nb_classes, nb_gpus):
    """
    Define and run the Convolutional Neural Network

    INPUT
        X_train: Array of NumPy arrays
        X_test: Array of NumPy arrays
        y_train: Array of labels
        y_test: Array of labels
        channels: Specify if the image is grayscale (1) or RGB (3)
        nb_epoch: Number of epochs
        batch_size: Batch size for the model
        nb_classes: Number of classes for classification

    OUTPUT
        Fitted CNN model
    """

    model = Sequential()

    # build resnet 34 layers
    model = resnet.ResnetBuilder.build_resnet_34((channels, img_rows, img_cols), nb_classes)

    #model = multi_gpu_model(model, gpus=nb_gpus)
    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.SGD(lr=0.0005, momentum=0.9, decay=0.25, nesterov=False),
                  metrics=['accuracy'])

    stop = EarlyStopping(monitor='val_acc',
                         min_delta=0.001,
                         patience=2,
                         verbose=0,
                         mode='auto')

    tensor_board = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

    # Parameters
    params = {'dim': (img_rows,img_cols),
              'batch_size': batch_size,
              'n_classes': nb_classes,
              'n_channels': channels,
              'shuffle': False}


    # Datasets
    partition_training = np.arange(X_training.shape[0])
    partition_validation = np.arange(X_validation.shape[0])

    print('---partition_training---')
    print(partition_training)
    print('len : ', len(partition_training))

    print('---partition_validation---')
    print(partition_validation)
    print('len : ', len(partition_validation))

    # Generators
    training_generator = DataGenerator(partition_training, X_training, y_training, **params)
    validation_generator = DataGenerator(partition_validation, X_validation, y_validation, **params)

    # Train model on dataset
    model.fit_generator(generator=training_generator, epochs=nb_epoch, verbose=1,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=1)

    return model


def save_model(model, score, model_name):
    """
    Saves Keras model to an h5 file, based on precision_score

    INPUT
        model: Keras model object to be saved
        score: Score to determine if model should be saved.
        model_name: name of model to be saved
    """

    if score >= 0.75:
        print("Saving Model")
        model.save(root_dir+"models/" + model_name + "_recall_" + str(round(score, 4)) + ".h5")
    else:
        print("Model Not Saved.  Score: ", score)


if __name__ == '__main__':
    # Specify parameters before model is run.
    batch_size = 2 # change this to bigger size on larger dataset e.g. 32
    nb_classes = 2
    nb_epoch = 30

    img_rows, img_cols = 256, 256
    channels = 3

    nb_gpus = 4

    # Import data
    labels = pd.read_csv(root_dir+'labels/trainLabels_master_256_v3.csv')

    X = np.load(root_dir+'data/X_train_256_v3.npy')
    y = np.array([1 if l >= 1 else 0 for l in labels['level']])

    # =============== start - splitting to train (4 institutions), validation and test dataset

    #train_size, val_size, test_size = 6000, 3000, 3000 # training, validation and test set for actual dataset
    train_size, val_size, test_size = 8, 2, 1 # training, validation and test set for sample dataset

    # extract the test and validation set
    trainfull_x, testval_x, trainfull_y, testval_y = split_data(X, y, val_size+test_size)

    #######
    #TRAIN SET
    #######
    # construct train set
    res_x, train_x, res_y, train_y = split_data(trainfull_x, trainfull_y, train_size)

    # divide to institutional size
    inst12_x, inst34_x, inst12_y, inst34_y = split_data(train_x, train_y, int(0.5*(train_x.shape[0])))

    # divide to institutional size
    inst1_x, inst2_x, inst1_y, inst2_y = split_data(inst12_x, inst12_y, int(0.5*(inst12_x.shape[0])))

    # divide to institutional size
    inst3_x, inst4_x, inst3_y, inst4_y = split_data(inst34_x, inst34_y, int(0.5*(inst34_x.shape[0])))

    #######
    #VALIDATION & TEST SET
    #######
    # construct test and validation set
    val_x, test_x, val_y, test_y = split_data(testval_x, testval_y, test_size)

    print('=======================================')
    print('\n----inst1---')
    print(inst1_x.shape)
    print('\n----inst2---')
    print(inst2_x.shape)
    print('\n----inst3---')
    print(inst3_x.shape)
    print('\n----inst4---')
    print(inst4_x.shape)
    print('\n----val---')
    print(val_x.shape)
    print('\n----test---')
    print(test_x.shape)

    # if all the 4 institutes
    X_training = train_x
    y_training = train_y

    X_validation = val_x
    y_validation = val_y

    X_test = test_x
    y_test = test_y

    # =============== end - splitting to train (4 institutions), validation and test dataset

    print("Reshaping Data")
    X_training = reshape_data(X_training, img_rows, img_cols, channels)
    X_validation = reshape_data(X_validation, img_rows, img_cols, channels)
    X_test = reshape_data(X_test, img_rows, img_cols, channels)

    print("X_training Shape: ", X_training.shape)
    print("X_validation Shape: ", X_validation.shape)
    print("X_test Shape: ", X_test.shape)

    input_shape = (img_rows, img_cols, channels)

    print("Normalizing Data")
    X_training = X_training.astype('float32')
    X_validation = X_validation.astype('float32')
    X_test = X_test.astype('float32')

    X_training /= 255
    X_validation /= 255
    X_test /= 255

    print("Training Model")

    model = cnn_model(X_training, X_validation, y_training, y_validation, channels, nb_epoch, batch_size,
                      nb_classes, nb_gpus=nb_gpus)


    print("Evaluate Model")

    # for evaluation, apply the to catregorical (2 classes). 
    # In the cnn_model function, the data generator is doing this categorical so the training data is not be applied this function on training.

    y_training = np_utils.to_categorical(y_training, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    print("y_training Shape: ", y_training.shape)
    print("y_test Shape: ", y_test.shape)

    train_score = model.evaluate(X_training, y_training, verbose=0)
    print('Train score:', train_score[0])
    print('Train accuracy:', train_score[1])

    print("Predicting")
    y_pred = model.predict(X_test)

    test_score = model.evaluate(X_test, y_test, verbose=0)
    print('Test score:', test_score[0])
    print('Test accuracy:', test_score[1])

    y_test = np.argmax(y_test, axis=1)
    y_pred = np.argmax(y_pred, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="micro")
    cohen_kappa = cohen_kappa_score(y_test, y_pred)
    quad_kappa = kappa(y_test, y_pred, weights='quadratic')

    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)
    print("Cohen Kappa Score", cohen_kappa)
    print("Quadratic Kappa: ", quad_kappa)

    save_model(model=model, score=recall, model_name="DR_Two_Classes")
    print("Completed")
