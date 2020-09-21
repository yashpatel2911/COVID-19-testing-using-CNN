import numpy as np
import pandas as pd
import pickle
from os import path
from tensorflow.keras.preprocessing import image

PATH = ''                                  # Edit this path to locate the dataset
set = 'dataset/'
metadata = 'Chest_xray_Corona_Metadata.csv'
train = 'train/'
test = 'test/'


def load_dataset(image_size=64):

    # Loading the COVID-19 csv file
    COVID_19 = pd.read_csv(PATH + metadata)
    print('Dataset Loaded Successfully.')

    # Concerting the output labels into 0 for Normal and 1 for Not Normal Cases
    print('Preprocessing the Data for setting up output Label...')
    Label = COVID_19['Label'].to_numpy()
    mask = (Label == 'Normal')
    Label[mask] = 0
    Label[~mask] = 1

    # Selecting the train label data
    print('Loading the training Labels...')
    y_train_temp = COVID_19['Label'][COVID_19['Dataset_type'] == 'TRAIN'].to_numpy().astype('float32')
    y_train = np.expand_dims(y_train_temp, axis=1)

    # Selecting the test label data
    print('Loading the testing Labels...')
    y_test_temp = COVID_19['Label'][COVID_19['Dataset_type'] == 'TEST'].to_numpy().astype('float32')
    y_test = np.expand_dims(y_test_temp, axis=1)

    print('Loading the train and test input data....')

    train_dataset = set + train + 'X_TRAIN_DUMP_' + str(image_size)
    test_dataset = set + test + 'X_TEST_DUMP_' + str(image_size)

    if path.isfile(train_dataset):

        print('Loading the train input data.')

        # Loading Training Data
        x_train_temp = open(train_dataset, 'rb')
        x_train = pickle.load(x_train_temp)

        print('Successfully Loaded the train input data.')

    else:

        print('No train Pickle Object is Found.')
        print('Pickling the train dataset. So Please wait for some time to finish the job.')

        train_dataframe = COVID_19['X_ray_image_name'][COVID_19['Dataset_type'] == 'TRAIN'].to_numpy()

        # Pickling the Train Dataset
        x_train = dumping_dataset(train_dataframe, 'TRAIN', train, image_size)

        print('Successfully Loaded the train input data.')

    if path.isfile(test_dataset):

        print('Loading the test input data.')

        # Loading Testing Data
        x_test_temp = open(test_dataset, 'rb')
        x_test = pickle.load(x_test_temp)

        print('Successfully Loaded the test input data.')

    else:

        print('No test Pickle Object is Found.')
        print('Pickling the test dataset. So Please wait for some time to finish the job.')

        test_dataframe = COVID_19['X_ray_image_name'][COVID_19['Dataset_type'] == 'TEST'].to_numpy()

        # Pickling the Test Dataset
        x_test = dumping_dataset(test_dataframe, 'TEST', test, image_size)

        print('Successfully Loaded the test input data.')

    return  x_train, y_train, x_test, y_test


def dumping_dataset(dataframe, type, location, image_size = 64):

    # Dumping the large dataset for faster Loading
    data = []

    dump_location = set + location + 'X_' + type + '_DUMP_' + str(image_size)

    for imageFile in dataframe:

        picture = image.load_img(PATH + dataset + location + imageFile,
                                 target_size=(image_size, image_size))
        picture_array = image.img_to_array(picture)
        picture_array = picture_array / 255
        data.append(picture_array)
        
    data = np.asanyarray(data, dtype='float32')

    pickle_out = open(dump_location, 'wb')
    pickle.dump(data, pickle_out)
    pickle_out.close()

    return data


if __name__ == '__main__':

    x_train, y_train, x_test, y_test = load_dataset(image_size=150)

    print(x_train.shape)
    print(y_train.shape)
    print(type(x_test))
    print(x_test.shape)
    print(y_test.shape)
