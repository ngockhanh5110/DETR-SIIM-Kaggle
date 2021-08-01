from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
import pandas as pd


def seperate_fold(train_df = None, n_splits = 8):
    """
    Input: 
      1. Dataframe of train.csv
      2. n_splits: number of fould
    Output:
      A dataframe with added fold columns
    """
    vin_train = train_df.copy()
    image_id_list = vin_train['id'].unique()
    vin_train['fold'] = -1
    
    kf = KFold(n_splits=n_splits, random_state=42, shuffle=True)
    for fold_num, indexes in enumerate(kf.split(image_id_list)):
        train_indexes, test_indexes = indexes
        image_id = image_id_list[test_indexes]        
        vin_train.loc[vin_train['id'].isin(image_id), 'fold'] = fold_num

    for i in range(n_splits):
        print(vin_train[vin_train["fold"]==i]["class"].value_counts())
        print("======================================================================")

    return vin_train