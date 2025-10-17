# clean_and_filter_data.py
# Standard Libraries
import os
import time
import pandas as pd

# Third Party Imports
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from matplotlib import image as img
import progressbar

# Local Imports
from cnn import CNN  # Assuming cnn.py is in the same directory

def load_all_image_paths_and_labels():
    """
    Loads the full paths and labels for all images without resizing them yet.
    """
    script_dir = os.path.dirname(os.path.realpath(__file__))
    covid_path = os.path.join(script_dir, "data", "covid")
    non_covid_path = os.path.join(script_dir, "data", "noncovid")

    covid_images = [os.path.join(covid_path, f) for f in os.listdir(covid_path) if f.endswith('.png')]
    non_covid_images = [os.path.join(non_covid_path, f) for f in os.listdir(non_covid_path) if f.endswith('.png')]

    filepaths = covid_images + non_covid_images
    labels = [1] * len(covid_images) + [0] * len(non_covid_images)
    
    return filepaths, labels

def load_images_from_paths(filepaths, img_size=128):
    """
    Loads and resizes images from a list of filepaths.
    """
    image_array = np.empty((len(filepaths), img_size, img_size, 1), dtype=np.float32)
    
    bar = progressbar.ProgressBar(maxval=len(filepaths), widgets=[
                                    progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    for i, path in enumerate(filepaths):
        try:
            image_npy = img.imread(path)
            resized_img = resize(image_npy, (img_size, img_size, 1), anti_aliasing=True)
            image_array[i] = resized_img
        except Exception as e:
            print(f"Warning: Could not load image {path}. Skipping. Error: {e}")
            # Replace with a black image if loading fails
            image_array[i] = np.zeros((img_size, img_size, 1), dtype=np.float32)
        bar.update(i+1)
        
    bar.finish()
    return image_array


if __name__ == '__main__':
    print("Step 1: Loading all image file paths and labels...")
    filepaths, labels = load_all_image_paths_and_labels()
    
    # Ensure labels are numpy array for splitting
    labels = np.array(labels, dtype=np.float32)

    print("Step 2: Splitting data into training and validation sets...")
    # We split the *paths*, not the images themselves, to save memory
    X_train_paths, X_val_paths, y_train, y_val = train_test_split(
        filepaths, labels, test_size=0.25, random_state=42, stratify=labels)

    print(f"Training set size: {len(X_train_paths)}")
    print(f"Validation set size: {len(X_val_paths)}")

    print("\nStep 3: Loading and processing images for training and validation...")
    X_train = load_images_from_paths(X_train_paths)
    X_val = load_images_from_paths(X_val_paths)

    print("\nStep 4: Initializing and training the CNN model...")
    model = CNN(weight_decimals=8)
    model.set_initial_params()
    
    # Train the model (you can adjust epochs)
    model.fit(X_train, y_train, X_val, y_val, epochs=20)

    print("\nStep 5: Evaluating model on the validation set to find misclassified images...")
    y_pred_probs = model.model.predict(X_val)
    y_pred = np.argmax(y_pred_probs, axis=-1)

    # Find the indices of the correctly classified images
    correctly_classified_indices = np.where(y_pred == y_val)[0]
    misclassified_count = len(y_val) - len(correctly_classified_indices)
    
    print(f"Found {len(correctly_classified_indices)} correctly classified images in the validation set.")
    print(f"Found {misclassified_count} misclassified images that will be filtered out.")

    # Create a list of the file paths for the images that were classified correctly
    correct_val_paths = [X_val_paths[i] for i in correctly_classified_indices]
    correct_val_labels = [y_val[i] for i in correctly_classified_indices]
    
    # The final "clean" dataset includes all original training images plus the correctly identified validation images
    final_filtered_paths = X_train_paths + correct_val_paths
    final_filtered_labels = np.concatenate([y_train, correct_val_labels])

    print(f"\nOriginal total dataset size: {len(filepaths)}")
    print(f"New filtered dataset size: {len(final_filtered_paths)}")
    
    print("\nStep 6: Saving the filtered dataset to 'filtered_dataset.csv'...")
    df = pd.DataFrame({
        'filepath': final_filtered_paths,
        'label': final_filtered_labels.astype(int)
    })
    
    output_path = os.path.join(os.path.dirname(__file__), 'filtered_dataset.csv')
    df.to_csv(output_path, index=False)
    
    print(f"\nProcess complete! Your clean dataset is ready at: {output_path}")