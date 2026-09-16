import os
import pandas as pd

def read_clean_ids(file_paths:list):
    dfs = []

    # Read each file
    for path in file_paths:
        df = pd.read_csv(
            path,
            sep=' ',              # Split on whitespace
            quotechar='"',           # Fields are quoted with "
            engine='python'          # Required for regex separator
        )
        df['source_file'] = path.split('/')[-1][:-4]     # Optional: Keep track of source file
        dfs.append(df)

    # Combine all into one DataFrame
    final_df = pd.concat(dfs, ignore_index=True)
    return final_df

def get_all_clean_ids(data_clean_ids_dir):
    all_clean_ids = [os.path.join(data_clean_ids_dir,f) for f in os.listdir(data_clean_ids_dir) if f.endswith('.txt')]
    return all_clean_ids

def get_Valid_ids(data_clean_dir):
    all_clean_ids = get_all_clean_ids(data_clean_dir)
    clean_ids_df = read_clean_ids(all_clean_ids)
    return clean_ids_df

def read_MCH_metadata(meta_data_path):
    metadata = pd.read_excel(meta_data_path, sheet_name=None)
    metadata_merged = pd.merge(metadata['Plant'], metadata['Sample'], on='plant_ID', how='inner')[['plant_ID', 'genus', 'species', 'sample_name', 'sample_date']]
    return metadata_merged

def get_final_df_with_labels(data_clean_dir, df, meta_data_path):
    metadata_merged = read_MCH_metadata(meta_data_path)
    event_base_name = get_Valid_ids(data_clean_dir)['event_base_name'].values
    # if 'sample_name' not in df.columns:
    #     df.rename(columns={'class': 'sample_name'}, inplace=True)
    df = pd.merge(df, metadata_merged[['genus','sample_name']], left_on='class', right_on='sample_name', how='left')
    df['genus'] = df['genus'].where(df['event_name'].isin(event_base_name), 'Garbage')
    df = df[['event', 'images', 'genus']]
    df.rename(columns={'genus': 'class'}, inplace=True)
    print("Shape of data after getting rid of invalid events:",df[~df['class'].isin(['Garbage'])].shape)
    return df
