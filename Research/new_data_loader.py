import pandas as pd
import numpy as np
import os
from skimage.transform import resize
from matplotlib import image as img
import progressbar
from sklearn.model_selection import train_test_split

def load_filtered_data_from_csv(csv_filename="filtered_dataset.csv", limit=None):
    """
    Loads image data and labels from the filtered CSV file.
    """
    script_dir = os.path.dirname(os.path.realpath(__file__))
    csv_path = os.path.join(script_dir, csv_filename)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"'{csv_filename}' not found. Please run the data cleaning script first.")

    df = pd.read_csv(csv_path)

    # Apply limit if specified
    if limit is not None:
        df = df.sample(n=limit, random_state=42).reset_index(drop=True)

    IMG_SIZE = 128
    
    filepaths = df['filepath'].values
    labels = df['label'].values.astype(np.float32)

    # Load and process images
    X = np.empty((len(filepaths), IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
    bar = progressbar.ProgressBar(maxval=len(filepaths), widgets=[
                                        progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    for i, path in enumerate(filepaths):
        full_path = os.path.join(script_dir, path) # Construct full path from script directory
        try:
            image_npy = img.imread(full_path)
            resized_image = resize(image_npy, (IMG_SIZE, IMG_SIZE, 1), anti_aliasing=True)
            X[i] = resized_image
        except Exception as e:
            print(f"\nWarning: Could not read image {full_path}. Skipping. Error: {e}")
            X[i] = np.zeros((IMG_SIZE, IMG_SIZE, 1)) # Use a blank image on failure
        bar.update(i+1)
        
    bar.finish()
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=1, stratify=labels)
        
    return X_train, X_test, y_train, y_test