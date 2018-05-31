from pathlib import Path


from sklearn.model_selection import train_test_split
import pandas as pd


def read_and_clean_dataset(path_to_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(str(path_to_csv), header=None, sep=' ')
    del df[7]
    df.columns = ['filename', 'class', 'train_sub_class', 'x1', 'y1', 'x2', 'y2']

    new_df = []
    for filename, objets in df.groupby('filename'):
        for unique_row, _ in objets.groupby(['class', 'x1', 'y1', 'x2', 'y2']):
            new_df.append([filename] + list(unique_row))
    new_df = pd.DataFrame(new_df, columns=['filename', 'class', 'x1', 'y1', 'x2', 'y2'])
    return new_df


def train_split(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    train_data = []
    val_data = []
    for _, class_df in df.groupby('class'):
        filenames = class_df['filename'].unique()
        filenames_train, filenames_val = train_test_split(filenames, test_size=0.2)
        class_df_train = class_df[class_df['filename'].isin(filenames_train)]
        class_df_val = class_df[class_df['filename'].isin(filenames_val)]
        train_data.append(class_df_train)
        val_data.append(class_df_val)
    train_data = pd.concat(train_data)
    val_data = pd.concat(val_data)
    return train_data, val_data


def create_test_set(path_to_test_file: Path) -> pd.DataFrame:
    df = pd.read_csv(path_to_test_file, header=None, sep='\t', na_values='none')
    df.columns = ['filename', 'class']
    fake_values = [None] * df.shape[0]
    df['x1'] = fake_values
    df['y1'] = fake_values
    df['x2'] = fake_values
    df['y2'] = fake_values
    return df


if __name__ == '__main__':
    main_path = Path('./data/flickr_logos_27_dataset/')
    data_path = main_path / 'flickr_logos_27_dataset_training_set_annotation.txt'
    test_data_path = main_path / 'flickr_logos_27_dataset_query_set_annotation.txt'
    dataset_path = Path('./datasets')

    df = read_and_clean_dataset(data_path)
    df_train, df_val = train_split(df)

    df_train.to_csv(dataset_path / 'train.csv', index=False)
    df_val.to_csv(dataset_path / 'val.csv', index=False)

    print(df_train['class'].value_counts())
    print(df_val['class'].value_counts())
    print(df_train.shape, df_train['filename'].unique().shape)
    print(df_val.shape, df_val['filename'].unique().shape)

    df_test = create_test_set(test_data_path)
    df_test.to_csv(dataset_path / 'test.csv', index=False)
    print(df_test.shape)
