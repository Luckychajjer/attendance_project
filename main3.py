import os
import imagehash
import pandas as pd
from PIL import Image

# Function to compute the hash of an image
def compute_image_hash(image_path):
    try:
        with Image.open(image_path) as img:
            hash_value = imagehash.average_hash(img)
            return hash_value
    except Exception as e:
        print(f"Error computing hash for {image_path}: {e}")
        return None

# Function to find duplicate images in two folders
def find_duplicate_images(folder1, folder2):
    # Get a list of image files in both folders
    files1 = [f for f in os.listdir(folder1) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    files2 = [f for f in os.listdir(folder2) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    # Compute hashes for images in the first folder
    hashes1 = {compute_image_hash(os.path.join(folder1, f)): f for f in files1}

    # Compare hashes of images in the second folder with the first folder
    duplicate_images = []
    for file2 in files2:
        hash2 = compute_image_hash(os.path.join(folder2, file2))
        if hash2 is not None:
            if hash2 in hashes1:
                duplicate_images.append((file2, hashes1[hash2]))

    return duplicate_images

# Function to update CSV file
def update_csv(csv_file, duplicates):
    if not duplicates:
        return

    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Iterate through the duplicate images and update the "ATTENDENCE" column
    for img1, img2 in duplicates:
        # Assuming the CSV file has a "NAME" column that matches image filenames
        img2_filename = os.path.splitext(os.path.basename(img2))[0]
        df.loc[df['NAME'] == img2_filename, 'ATTENDENCE'] = 'PRESENT'

    # Save the updated DataFrame back to the CSV file
    df.to_csv(csv_file, index=False)

if __name__ == "__main__":
    folder1 = r"C:\Users\asus\Desktop\ATTENDENCE\CAPTURED IMAGES"
    folder2 = r"C:\Users\asus\Desktop\ATTENDENCE\MAIN IMAGES"
    
    # Change this line to specify the correct CSV file path
    csv_file = r"C:\Users\asus\Desktop\ATTENDENCE\attendence.CSV"

    # Call the function and store the result in 'duplicates'
    duplicates = find_duplicate_images(folder1, folder2)

    if duplicates:
        print("Duplicate images found:")
        for img1, img2 in duplicates:
            print(f"{img1} in folder 2 is the same as {img2} in folder 1")

        # Update the CSV file with attendance information
        update_csv(csv_file, duplicates)
        print("CSV file updated.")
    else:
        print("No duplicate images found.")
