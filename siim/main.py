import pandas as pd
import numpy as np
from .coco_generater import make_json
from .kfold import seperate_fold

if __name__ == '__main__':
    img_df = pd.read_csv('./data/train_image_level.csv')

    # Adding `class` column
    img_df['class'] = np.where(img_df['label'].str.startswith('none'), 0, 1)
    

    img_df = seperate_fold(img_df, n_splits=5)

    make_json(img_df[img_df['fold'].isin([0,1,2,3])], image_set='train')
    make_json(img_df[img_df['fold'].isin([4])], image_set='val')
