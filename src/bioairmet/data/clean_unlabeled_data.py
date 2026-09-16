'''
@file    :   clean_unlabeled_data.py
@create date : 2025-10-15 11:12:00
@modify date 2026-04-29 14:44:58
@author  :   Mansoor Nabawi
@version :   1.0
@contact :   mansoor.nabawi@gmail.com
@license :   (C)Copyright 2026, Physikalisch-Technische Bundesanstalt (PTB) - BioAirMet Project
@desc    :   [
    This script is designed to clean and preprocess data in the BioAirMet project. 
    It performs the following steps:
    1. Recursively scans directories to find relevant JSON and image files, excluding specific non-target types (e.g., phase images).
    2. Extracts event names, class labels, and associated image pairs, handling nested directory structures.
    3. Filters events to ensure each has exactly two associated images and valid holographic properties (area, solidity, and intensity delta).
    4. Extracts fluorescence spectra in a 3x5 format using an optimized parallel processing pipeline (multiprocessing) to handle large datasets.
    5. Explodes the image pairs into individual entries to prepare the dataset for SSL training routines (One image per row).
    6. Exploding can be skipped to keep both images in the same row, which is useful for certain types of models that can take paired inputs.
    7. Saves the final cleaned DataFrame to an HDF5 file, preserving complex data types like tensors and lists.
    ]
'''

import torch
import os
# from works_.POLENO.main.data.dataset import define_preprocess
import pandas as pd
import numpy as np
import json
# from torch.utils.data import Dataset
# from torchvision import transforms
from PIL import Image
# import matplotlib.image as mpimg
# import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
from collections import defaultdict
import multiprocessing as mp
from functools import partial
import gc
from typing import Tuple


# json_post: list = ['_event.json', '_ev.json']
# cam0: list = [
#     '_ev.computed_data.holography.image_pairs.0.0.rec_mag.png', '_rec0.png']
# cam1: list = [
#     '_ev.computed_data.holography.image_pairs.0.1.rec_mag.png', '_rec1.png']

def create_category_mapping(file_path):
    """
    Create a category mapping from a file.

    The file should contain lines in the format 'category: number'. The category
    mapping is a dictionary where the keys are the categories and the values are
    the corresponding numbers.

    Args:
        file_path (str): The path to the file containing the category mapping.

    Returns:
        dict: A dictionary mapping category names to their corresponding numbers.
    """
    category_mapping = {}
    with open(file_path, 'r') as file:
        next(file)  # Skip the first line
        for line in file:
            line = line.strip()
            if line:  # Ensure the line is not empty
                key_value = line.split('\t')
                if len(key_value) == 2:
                    category = key_value[0].strip()
                    number = int(key_value[1].strip())
                    category_mapping[category] = number
    return category_mapping

def get_category_mapping(cat_map_path):
    """
    Return the category mapping from a file.

    :param file_path: The path to the file containing the category mapping.
    :return: A dictionary mapping category names to their corresponding numbers.
    """
    return create_category_mapping(cat_map_path)

def get_directories_and_files(path):
    """
    Recursively finds all files in all directories under the given path.
    Returns a mapping of full directory paths to their files, preserving the complete structure.
    
    :param path: The directory path to scan.
    :return: A dictionary where keys are the full directory paths 
             and values are the lists of files within them.
    """
    directory_dict = {}
    EXCLUDED_SUFFIXES = ('.npy', 'rec_pha.png')

    try:
        if not os.path.isdir(path):
            print(f"Error: Path '{path}' is not a valid directory.")
            return {}

        for root, dirs, files in os.walk(path):
            # Filter files based on excluded suffixes
            filtered_files = [
                f for f in files 
                if not f.endswith(EXCLUDED_SUFFIXES)
            ]
            
            if filtered_files:
                # Use the full root path as the key to preserve directory structure
                directory_dict[root] = filtered_files
                        
    except (FileNotFoundError, PermissionError, OSError) as e:
        print(f"Error accessing directory '{path}': {e}")
        return {}
    
    return directory_dict

def get_event_name(data_dict, base_path):
    """
    Return a list of dictionaries suitable for creating a pandas DataFrame with columns for class name,
    JSON files, and images as a list. The file paths are converted to absolute paths based on the given directory paths.

    :param data_dict: Data given as a dictionary. Keys are the full directory paths,
                    and values are all JSON and image files in those directories.
    :param base_path: The base directory (used for relative class name extraction).
    :return: A list of dictionaries with keys 'class', 'event', and 'images'.
    """
    json_post_: list = ['_event.json', '_ev.json']
    cam0_: list = [
        '_ev.computed_data.holography.image_pairs.0.0.rec_mag.png', '_rec0.png']
    cam1: list = [
        '_ev.computed_data.holography.image_pairs.0.1.rec_mag.png', '_rec1.png']
    event_list = []
    
    for full_dir_path, files in data_dict.items():
        if len(files) == 0:
            print(f"Warning: No files found in directory '{full_dir_path}'")
            continue
            
        # Extract class name from the directory path relative to base_path
        try:
            rel_path = os.path.relpath(full_dir_path, base_path)
            # Use the immediate parent directory name as class, or the relative path if nested
            # class_name = os.path.basename(full_dir_path) if '/' not in rel_path else rel_path.replace('/', '_')
            class_name = os.path.basename(rel_path)
            
        except ValueError:
            # If paths are on different drives or can't be made relative
            class_name = os.path.basename(full_dir_path)
        
        # Separate JSON and image files
        json_files = [file for file in files if file.endswith('.json')]
        img_files = [file for file in files if file.endswith('.png')]
        
        if not json_files or not img_files:
            continue
            
        # Determine patterns
        cam0 = cam0_[0] if img_files[0].endswith(cam0_[0][-11:]) else cam0_[1]
        json_post = json_post_[0] if json_files[0].endswith(json_post_[0]) else json_post_[1]

        # Create a mapping from event prefix to image files
        img_files_dict = {}
        for img_file in img_files:
            event_prefix = img_file.split(cam0[:2])[0]
            # Use the full directory path to construct absolute path
            abs_img_file = os.path.join(full_dir_path, img_file)
            img_files_dict.setdefault(event_prefix, []).append(abs_img_file)

        # Build the event_list with dictionaries for DataFrame creation
        for json_file in json_files:
            event_prefix = json_file.split(json_post[:2])[0]
            # Use the full directory path to construct absolute path
            abs_json_file = os.path.join(full_dir_path, json_file)
            images = img_files_dict.get(event_prefix, [])
            if len(images) > 2:
                suffixes = (cam0_[0], cam1[0])
                images = [image for image in images if image.endswith(suffixes)]
            event_list.append({
                'class': class_name,
                'event': abs_json_file,
                'images': images
            })
    return event_list

def get_event_name_fast(data_dict, base_path):
    """
    High-performance version using defaultdict and reduced string operations.
    Handles nested directory structures properly.
    """
    JSON_SUFFIXES = ('_event.json', '_ev.json')
    IMG_PATTERNS = ('_rec0.png', '_ev.computed_data.holography.image_pairs.0.0.rec_mag.png')
    CAM1_SUFFIX = '_ev.computed_data.holography.image_pairs.0.1.rec_mag.png'
    
    event_list = []
    
    for full_dir_path, files in data_dict.items():
        if not files:
            continue
            
        # Extract class name from the directory path relative to base_path
        try:
            rel_path = os.path.relpath(full_dir_path, base_path)
            # Use the immediate parent directory name as class, or the relative path if nested
            # class_name = os.path.basename(full_dir_path) if '/' not in rel_path else rel_path.replace('/', '_')
            class_name = os.path.basename(full_dir_path)
            
        except ValueError:
            # If paths are on different drives or can't be made relative
            class_name = os.path.basename(full_dir_path)
            
        # Separate files in single pass
        json_files = [f for f in files if f.endswith('.json')]
        img_files = [f for f in files if f.endswith('.png')]
        
        if not json_files or not img_files:
            continue
        
        # Determine patterns
        cam0_pattern = IMG_PATTERNS[0] if img_files[0].endswith(IMG_PATTERNS[0]) else IMG_PATTERNS[1]
        json_suffix = JSON_SUFFIXES[0] if json_files[0].endswith(JSON_SUFFIXES[0]) else JSON_SUFFIXES[1]
        
        # Use defaultdict for automatic list creation
        img_files_dict = defaultdict(list)
        
        # Optimized prefix extraction using full directory path
        for img_file in img_files:
            event_prefix = img_file[:-len(cam0_pattern)] if img_file.endswith(cam0_pattern) else img_file.split(cam0_pattern[:2])[0]
            img_files_dict[event_prefix].append(os.path.join(full_dir_path, img_file))
        
        # Process JSON files using full directory path
        for json_file in json_files:
            event_prefix = json_file[:-len(json_suffix)] if json_file.endswith(json_suffix) else json_file.split(json_suffix[:2])[0]
            
            images = img_files_dict.get(event_prefix, [])
            
            # Filter if needed
            if len(images) > 2:
                cam_suffixes = (cam0_pattern, CAM1_SUFFIX)
                images = [img for img in images if img.endswith(cam_suffixes)]
            
            event_list.append({
                'class': class_name,
                'event': os.path.join(full_dir_path, json_file),
                'images': images
            })
    
    return event_list

def get_df(data) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Converts the dictionary of event names to a pandas DataFrame.

    Args:
        data (Dict[str, List[str]]): A dictionary where keys are the class names and
            values are lists of event names.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A DataFrame with event data and a DataFrame of event names.
    """
    # df = pd.DataFrame([(key, value) for key, values in data.items() for value in values], columns=['date', 'event'])
    # df['event'] = df.apply(lambda row: os.path.join(self.root_dir, str(row.iloc[0]), row['event']), axis=1)
    df = pd.DataFrame(data)
    df = df[df['images'].apply(lambda x: len(x) == 2)]
    print(df.shape)
    df['event_name'] = df['event'].apply(lambda x: x.split('/')[-1].split('_ev')[0])
    print('drop duplicates...')
    df = df.drop_duplicates(subset=['event_name'], keep='first', ignore_index=True)
    print(df.shape)

    return df, df[['event_name']]

def read_image(img_path: str) -> Image.Image:
    """
    Reads an image, normalizes it from its original bit-depth (e.g., 16-bit)
    to 8-bit, and returns it as a PIL Image in grayscale ('L') mode.

    Args:
        image_path (str): The file path to the image.

    Returns:
        Image.Image: A normalized 8-bit grayscale PIL Image.
    
    Raises:
        FileNotFoundError: If the image path does not exist.
        ValueError: If the image cannot be processed.
    """
    try:
        # Open the image using Pillow
        pil_img = Image.open(img_path)
        
        # Convert to a NumPy array to access full pixel data. Use float for calculations.
        img_array = np.array(pil_img, dtype=np.float32)

        # If the image is already 8-bit, just ensure it's in 'L' mode.
        if pil_img.mode == 'L' and np.max(img_array) <= 255:
            return pil_img

        # --- Normalization for high bit-depth images ---
        min_val = np.min(img_array)
        max_val = np.max(img_array)

        if max_val > min_val:
            # Scale pixel values to the [0, 1] range
            img_array = (img_array - min_val) / (max_val - min_val)
            # Scale to the [0, 255] range for 8-bit
            img_array = (img_array * 255).astype(np.uint8)
        else:
            # Handle flat images (all pixels have the same value)
            img_array = np.zeros_like(img_array, dtype=np.uint8)

        # Convert the normalized array back to a PIL Image
        normalized_img = Image.fromarray(img_array, mode='L')
        return normalized_img

    except FileNotFoundError:
        print(f"Error: Image file not found at {img_path}")
        raise
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
        raise ValueError(f"Could not process image {img_path}")

def check_eventNgetspectra(event_path, min_area=500, min_solidity=0.7, min_intensity_delta=0.1, skip_validation=False):
    """
    Optimized version: Reads an event JSON, checks for valid holographic properties, 
    and extracts fluorescence spectra in a 3x5 tensor format if available.
    
    Returns numpy arrays instead of torch tensors to avoid multiprocessing serialization issues.
    """
    try:
        # Single file read with error handling
        with open(event_path, 'r') as file:
            event_data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError, PermissionError):
        return -1
    
    # Early validation of basic structure
    if not isinstance(event_data, dict):
        return -1
    
    # Determine format and validate properties
    try:
        if 'computedData' in event_data:
            result = _process_legacy_format_np(event_data, min_area, min_solidity, min_intensity_delta, skip_validation=skip_validation)
        elif 'computed_data' in event_data:
            result = _process_new_format_np(event_data, min_area, min_solidity, min_intensity_delta, skip_validation=skip_validation)
        else:
            return -1
    except:
        return -1
    
    return result

def _validate_properties(props, min_area, min_solidity, min_intensity_delta):
    """Fast property validation with early exit"""
    try:
        return (props['area'] >= min_area and 
                props['solidity'] >= min_solidity and
                (props.get('maxIntensity', props.get('max_intensity', 0)) - 
                 props.get('minIntensity', props.get('min_intensity', 0))) >= min_intensity_delta and
                props.get('majorAxis', props.get('major_axis_length', 0)) != -1 and
                props.get('minorAxis', props.get('minor_axis_length', 0)) != -1)
    except (KeyError, TypeError):
        return False

def _process_legacy_format_np(event_data, min_area, min_solidity, min_intensity_delta, skip_validation=False):
    """Process legacy format returning numpy arrays instead of tensors"""
    computed_data = event_data['computedData']
    
    # Fast validation using list comprehension
    for i in range(2):
        prop_key = f'img{i}Properties'
        if prop_key not in computed_data:
            return -1
        if not skip_validation:
            if not _validate_properties(computed_data[prop_key], min_area, min_solidity, min_intensity_delta):
                return -1
    
    # Extract spectra with prioritized paths
    fluor_data = computed_data.get('fluorescenceSpectra', {})
    # Fast check for None or null or null-like values
    if fluor_data is None or len(fluor_data) == 0:
        print("Fluorescence data is None or empty.")
        return -1
    
    # Try relative_spectra first (most common)
    if 'relative_spectra' in fluor_data:
        try:
            return np.array(fluor_data['relative_spectra'], dtype=np.float32).reshape(3, 5)
        except (ValueError, RuntimeError):
            pass
    
    # Fallback to mean_average_scaled
    if 'mean_average_scaled' in fluor_data:
        try:
            mas = fluor_data['mean_average_scaled']
            data = [mas.get('mean_280', [0]*5), mas.get('mean_365', [0]*5), mas.get('mean_405', [0]*5)]
            return np.array(data, dtype=np.float32)
        except (ValueError, RuntimeError):
            pass
    
    return -1

def _process_new_format_np(event_data, min_area, min_solidity, min_intensity_delta, skip_validation=False):
    """Process new format returning numpy arrays instead of tensors"""
    try:
        image_pairs = event_data['computed_data']['holography']['image_pairs'][0]
    except (KeyError, IndexError):
        return -1
    
    # Fast validation with proper None check
    for i in range(2):
        if (i >= len(image_pairs) or 
            image_pairs[i] is None or 
            not isinstance(image_pairs[i], dict) or
            'rec_mag_properties' not in image_pairs[i]):
            return -1
        if not skip_validation:
            if not _validate_properties(image_pairs[i]['rec_mag_properties'], min_area, min_solidity, min_intensity_delta):
                return -1
    
    # Extract spectra with optimized path access
    try:
        spectra_data = event_data['computed_data']['fluorescence']['processed_data']['spectra']['relative_spectra']
        return np.array(spectra_data, dtype=np.float32).reshape(3, 5)
    except (KeyError, ValueError, RuntimeError):
        pass
    
    # Fallback path
    try:
        fluor_data = event_data['computed_data'].get('fluorescenceSpectra', {})
        if 'mean_average_scaled' in fluor_data:
            mas = fluor_data['mean_average_scaled']
            data = [mas.get('mean_280', [0]*5), mas.get('mean_365', [0]*5), mas.get('mean_405', [0]*5)]
            return np.array(data, dtype=np.float32)
    except (ValueError, RuntimeError):
        pass
    
    return -1

def save_df_to_hdf5(df, path):
    """
    Saves a DataFrame with all columns to an HDF5 file, handling various data types.

    Args:
        df (pd.DataFrame): The DataFrame to save with any columns
        path (str): The file path for the output HDF5 file.
    """
    try:
        with h5py.File(path, 'w') as hf:
            # Save each column based on its data type
            for column in df.columns:
                data = df[column]
                
                if column == 'relative_spectra':
                    # Handle tensor/numpy array data for spectra
                    tensors_data = np.array([
                        t.numpy() if hasattr(t, 'numpy') else t 
                        for t in data.tolist()
                    ])
                    hf.create_dataset(column, data=tensors_data)
                
                elif data.dtype == 'object':
                    # Handle object columns (lists, strings, etc.)
                    dt = h5py.special_dtype(vlen=str)
                    
                    # Check if it's a list of lists/arrays
                    first_item = data.iloc[0] if len(data) > 0 else None
                    if isinstance(first_item, (list, tuple, np.ndarray)):
                        # Convert lists to JSON strings
                        string_data = np.array([
                            json.dumps(item.tolist() if hasattr(item, 'tolist') else item) 
                            for item in data
                        ], dtype=dt)
                    else:
                        # Handle regular strings
                        string_data = data.to_numpy().astype(dt)
                    
                    hf.create_dataset(column, data=string_data)
                
                else:
                    # Handle numeric columns directly
                    hf.create_dataset(column, data=data.to_numpy())
            
            # Save metadata about column types for proper reconstruction
            column_info = {}
            for col in df.columns:
                if col == 'relative_spectra':
                    column_info[col] = 'tensor_array'
                elif df[col].dtype == 'object':
                    first_item = df[col].iloc[0] if len(df[col]) > 0 else None
                    if isinstance(first_item, (list, tuple, np.ndarray)):
                        column_info[col] = 'list_object'
                    else:
                        column_info[col] = 'string_object'
                else:
                    column_info[col] = str(df[col].dtype)
            
            hf.attrs['column_info'] = json.dumps(column_info)
            hf.attrs['columns'] = json.dumps(list(df.columns))

        print(f"✅ DataFrame with {len(df.columns)} columns successfully saved to '{path}'")
        print(f"   Columns saved: {list(df.columns)}")
        
    except Exception as e:
        print(f"❌ Error saving DataFrame to HDF5: {str(e)}")
        raise

def load_df_from_hdf5(path):
    """
    Loads data from an HDF5 file back into a Pandas DataFrame with all original columns.

    Args:
        path (str): The path to the HDF5 file.

    Returns:
        pd.DataFrame: DataFrame with all original columns and proper data types
    """
    try:
        with h5py.File(path, 'r') as hf:
            # Load metadata
            columns = json.loads(hf.attrs.get('columns', '[]'))
            column_info = json.loads(hf.attrs.get('column_info', '{}'))
            
            # If no metadata, fall back to dataset keys
            if not columns:
                columns = list(hf.keys())
                print("⚠️ No column metadata found, using dataset keys")
            
            data = {}
            
            for column in columns:
                if column not in hf:
                    print(f"⚠️ Column '{column}' not found in HDF5 file")
                    continue
                
                dataset = np.array(hf[column])
                col_type = column_info.get(column, 'unknown')
                
                if col_type == 'tensor_array':
                    # Convert back to list of arrays/tensors
                    data[column] = [arr for arr in dataset]
                
                elif col_type in ['list_object', 'string_object']:
                    # Handle object columns
                    try:
                        # Decode byte strings if necessary
                        if isinstance(dataset[0], bytes):
                            string_data = [item.decode('utf-8') for item in dataset]
                        else:
                            string_data = dataset.tolist()
                        
                        if col_type == 'list_object':
                            # Parse JSON strings back to lists
                            data[column] = [json.loads(item) for item in string_data]
                        else:
                            # Keep as strings
                            data[column] = string_data
                            
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        print(f"⚠️ Error decoding column '{column}': {e}")
                        # Fallback to raw data
                        data[column] = dataset.tolist()
                
                else:
                    # Handle numeric columns
                    data[column] = dataset
            
            df = pd.DataFrame(data)

        print(f"✅ DataFrame with {len(df.columns)} columns successfully loaded from '{path}'")
        print(f"   Columns loaded: {list(df.columns)}")
        print(f"   Shape: {df.shape}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading DataFrame from HDF5: {str(e)}")
        raise
    
# Add this function for parallel processing
def process_events_parallel(events, min_area=500, min_solidity=0.7, min_intensity_delta=0.1, n_processes=None, skip_validation=False):
    """
    Process events in parallel using multiprocessing with high resource utilization but preventing memory errors.
    
    Args:
        events (list): List of event paths to process
        min_area (int): Minimum area threshold
        min_solidity (float): Minimum solidity threshold
        min_intensity_delta (float): Minimum intensity delta threshold
        n_processes (int, optional): Number of processes to use.
        
    Returns:
        list: List of processed spectra results
    """
    # Use half of available CPUs (16 of 32 cores) for optimal performance
    if n_processes is None:
        n_processes = max(1, mp.cpu_count() // 2)
        print(f"Using {n_processes} processes (half of {mp.cpu_count()} available cores)")
    
    # Create a partial function with fixed parameters
    process_func = partial(check_eventNgetspectra, skip_validation=skip_validation,
                          min_area=min_area,
                          min_solidity=min_solidity, 
                          min_intensity_delta=min_intensity_delta)
    
    # Calculate optimal batch size based on total events and memory constraints
    # For ~62.7GB RAM, we can process more events per batch
    total_events = len(events)
    batch_size = min(50000, max(5000, total_events // 10))
    
    print(f"Processing {total_events} events in batches of {batch_size}")
    results = []
    
    # Process events in batches
    for i in range(0, total_events, batch_size):
        batch = events[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total_events - 1) // batch_size + 1
        print(f"\nBatch {batch_num}/{total_batches}: Processing {len(batch)} events")
        
        # Calculate optimal chunksize based on batch size and processes
        chunksize = max(10, min(1000, len(batch) // (n_processes * 4)))
        
        try:
            # Use 'fork' method for better memory efficiency
            with mp.Pool(processes=n_processes) as pool:
                batch_results = list(tqdm(
                    pool.imap(process_func, batch, chunksize=chunksize),
                    total=len(batch),
                    desc=f"Processing with {n_processes} processes"
                ))
                results.extend(batch_results)
                
        except Exception as e:
            print(f"Warning: Parallel processing failed ({e}), trying with reduced processes")
            
            try:
                # Try again with fewer processes
                reduced_n = max(1, n_processes // 2)
                with mp.Pool(processes=reduced_n) as pool:
                    batch_results = list(tqdm(
                        pool.imap(process_func, batch, chunksize=chunksize*2),
                        total=len(batch),
                        desc=f"Retry with {reduced_n} processes"
                    ))
                    results.extend(batch_results)
            except Exception as e2:
                print(f"Retry failed ({e2}), falling back to sequential processing")
                batch_results = []
                for event in tqdm(batch, desc="Sequential processing"):
                    batch_results.append(process_func(event))
                results.extend(batch_results)
        
        # Force garbage collection between batches
        gc.collect()
    
    # Convert numpy arrays to tensors at the end
    print("Converting results to tensors...")
    tensor_results = []
    for res in results:
        if isinstance(res, np.ndarray):
            tensor_results.append(torch.tensor(res, dtype=torch.float32))
        else:
            tensor_results.append(res)
    
    return tensor_results

if __name__ == "__main__":
    path = "/home/pre_data/downloaded_zips"
    
    print("Starting data processing...")
    print(f"1. Getting directories and files from: {path}")
    data = get_directories_and_files(path)
    
    print("2. Extracting event names and image paths...")
    data_fast = get_event_name_fast(data, path)
    
    print("3. Converting to DataFrame and filtering invalid entries...")
    data_final, df_all_events = get_df(data_fast)
    
    print("4. Extracting fluorescence spectra with memory-optimized parallel processing...")
    event_paths = data_final['event'].tolist()
    spectra_results = process_events_parallel(event_paths)

    data_final['relative_spectra'] = spectra_results

    print("5. Filtering out invalid spectra entries...")
    # vectorized boolean mask on DataFrame
    mask = data_final['relative_spectra'].apply(lambda x: not isinstance(x, int))
    num_valid_spectra = mask.sum()
    print(f"Number of valid spectra: {num_valid_spectra} out of {len(spectra_results)}")

    # Faster filtering by creating a new DataFrame directly from filtered data
    valid_spectra_df = data_final[mask].reset_index(drop=True)
    print(f"Shape of cleaned DataFrame with valid spectra: {valid_spectra_df.shape}")
    
    # --- Only for single ssl dataset ---, this will repeat each row but every time the image is different and relative fl is repeated.
    print("5.1 exploding images column for SSL unlabeled data...")
    valid_spectra_df = valid_spectra_df.explode('images').reset_index(drop=True)
    print(f"Shape after exploding images: {valid_spectra_df.shape}")

    output_path = '/home/SSL_old_data_cleaned.h5'
    print(f"6. Saving cleaned DataFrame to HDF5: {output_path}")
    save_df_to_hdf5(valid_spectra_df, output_path)
        
    print("Data processing complete!")
