import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split

root_dir = '../../Data/sample/'

dict_labels = {0:"No DR", 1:"Mild", 2:"Moderate", 3:"Severe", 4:"Proliferative DR"}

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


def change_labels(df, category):
    '''
    Changes the labels for a binary classification.
    Either the person has a degree of retinopathy, or they don't.

    INPUT
        df: Pandas DataFrame of the image name and labels
        category: column of the labels

    OUTPUT
        Column containing a binary classification of 0 or 1
    '''
    return [1 if l > 0 else 0 for l in df[category]]




def plot_classification_frequency(df, category, file_name, convert_labels = False):
    '''
    Plots the frequency at which labels occur

    INPUT
        df: Pandas DataFrame of the image name and labels
        category: category of labels, from 0 to 4
        file_name: file name of the image
        convert_labels: argument specified for converting to binary classification

    OUTPUT
        Image of plot, showing label frequency
    '''
    if convert_labels == True:
        labels['level'] = change_labels(labels, 'level')

    print(labels)
    print(labels.shape)

    #scale the dataset
    combined_df = pd.concat([labels, labels]).reset_index(drop=True)

    print(combined_df)
    print(combined_df.shape)

    X = combined_df
    Y = combined_df


    sns.set(style="whitegrid", color_codes=True)
    sns.countplot(x=category, data=labels)
    plt.title('Retinopathy vs Frequency')
    plt.savefig(file_name)



if __name__ == '__main__':
    labels = pd.read_csv(root_dir+'labels/trainLabels.csv')

    plot_classification_frequency(labels, "level", "Retinopathy_vs_Frequency_All")
    plot_classification_frequency(labels, "level", "Retinopathy_vs_Frequency_Binary", True)

    plt.show()
