import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, data_arr, labels, batch_size=4, dim=(256,256), n_channels=3,
                 n_classes=2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.data_arr = data_arr
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        print('--len self id :', len(self.list_IDs) )
        print('--self.batch_size :', self.batch_size )
        print('---len : ', int(np.floor(len(self.list_IDs) / self.batch_size)))
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            #X[i,] = np.load('data/' + ID + '.npy')
            #print('===========')
            #print('---batch_size : ', self.batch_size)
            #print('---ID : ', ID)
            #print('---i : ', i)
            #print('---ID : ', self.data_arr[ID].shape)
            #print('===========')

            X[i,] = self.data_arr[ID]

            # Store class
            y[i] = self.labels[ID]

        #print(X.shape)

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)