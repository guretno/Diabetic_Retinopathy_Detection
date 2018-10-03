import os

import pandas as pd

root_dir = '../../Data/sample/'


if __name__ == '__main__':
    trainLabels = pd.read_csv(root_dir+'labels/trainLabels_master_256_v2.csv')

    # remove the mild diabetic retinopathy images(scale 1)
    trainLabels = trainLabels.loc[trainLabels['level'] != 1]

    # select only the left eye
    trainLabels = trainLabels[trainLabels['image'].str.contains("_left")==True]

    print(trainLabels.shape)

    print("Writing CSV")
    trainLabels.to_csv(root_dir+'labels/trainLabels_master_256_v3.csv', index=False, header=True)
