import pandas as pd
import sklearn
import glob
import os

class DataHandler():
    
    def __init__(self, args, path_prefix='data/images'):
        self.args = args
        self.path_prefix = path_prefix
    
    
    def load_dataset(self, shuffle=True, verbose=1):

        meteors_files = glob.glob(os.path.join(self.path_prefix, 'meteor/*.jpg'))
        nonmeteors_files = glob.glob(os.path.join(self.path_prefix, 'nonmeteor/*.jpg'))

        meteors_files = list(map(lambda x: '/'.join(x.split('/')[2:4]), meteors_files))
        nonmeteors_files = list(map(lambda x: '/'.join(x.split('/')[2:4]), nonmeteors_files))

        df_meteors = pd.DataFrame({
            'id': meteors_files, 
            'label': ['meteor'] * len(meteors_files)
        })
        df_nonmeteors = pd.DataFrame({
            'id': nonmeteors_files, 
            'label': ['nonmeteor'] * len(nonmeteors_files)
        })

        df_images = pd.concat([df_meteors, df_nonmeteors], axis=0)
        if shuffle:
            df_images = sklearn.utils.shuffle(df_images, random_state=self.args.seed)

        if verbose > 0:
            print('Found {} validated image filenames.'.format(df_images.shape[0]))
            print('{:.0%} belonging to Meteor class.'.format(df_images["label"].value_counts(normalize=True)["meteor"]))
            print('{:.0%} belonging to Non-Meteor class.'.format(df_images["label"].value_counts(normalize=True)["nonmeteor"]))

        return df_images


    def generate_data(self, 
                      data_generator,
                      df_train,
                      df_valid,
                      df_test,
                      batch_size=8,
                      shuffle=True):

        train_gen = data_generator.flow_from_dataframe(
            df_train,
            self.path_prefix,
            x_col='id',
            y_col='label',
            target_size=self.args.image_size,
            batch_size=batch_size,
            classes=self.args.classes,
            shuffle=shuffle,
            seed=self.args.seed,
        )
        valid_gen = data_generator.flow_from_dataframe(
            df_valid,
            self.path_prefix,
            x_col='id',
            y_col='label',
            target_size=self.args.image_size,
            batch_size=batch_size,
            classes=self.args.classes,
            shuffle=shuffle,
            seed=self.args.seed,
        )
        test_gen = data_generator.flow_from_dataframe(
            df_test,
            self.path_prefix,
            x_col='id',
            y_col='label',
            target_size=self.args.image_size,
            batch_size=batch_size,
            classes=self.args.classes,
            shuffle=shuffle,
            seed=self.args.seed,
        )

        return train_gen, valid_gen, test_gen