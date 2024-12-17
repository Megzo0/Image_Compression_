import numpy as np
from sklearn.cluster import KMeans
import cv2
import pickle

# Compression function
def compress_image(image_path, block_size, codebook_size, output_file):
    """
    Compress a grayscale image using vector quantization.

    Args:
        image_path (str): Path to the grayscale image file.
        block_size (int): Size of each block (e.g., 4 for 4x4 blocks).
        codebook_size (int): Number of codewords in the codebook.
        output_file (str): Path to save the compressed file.
    """
    # Read the grayscale image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Invalid image file.")

    height, width = image.shape
    padded_height = (height + block_size - 1) // block_size * block_size
    padded_width = (width + block_size - 1) // block_size * block_size

    # Pad the image to make it divisible by block size
    padded_image = np.zeros((padded_height, padded_width), dtype=np.uint8)
    padded_image[:height, :width] = image

    # Split the image into non-overlapping blocks
    blocks = []
    for i in range(0, padded_height, block_size):
        for j in range(0, padded_width, block_size):
            block = padded_image[i:i+block_size, j:j+block_size].flatten()
            blocks.append(block)

    blocks = np.array(blocks)

    # Perform K-Means clustering to generate the codebook
    kmeans = KMeans(n_clusters=codebook_size, random_state=0, n_init=10)
    kmeans.fit(blocks)

    codebook = kmeans.cluster_centers_
    labels = kmeans.labels_

    compressed_data = {
        "codebook": codebook,
        "labels": labels,
        "original_shape": (height, width),
        "block_size": block_size,
        "padded_shape": (padded_height, padded_width)
    }

    # Save the compressed data to a file
    with open(output_file, 'wb') as f:
        pickle.dump(compressed_data, f)

# Decompression function
def decompress_image(input_file, output_image_path):
    """
    Decompress an image using vector quantization.

    Args:
        input_file (str): Path to the compressed file.
        output_image_path (str): Path to save the decompressed image.
    """
    # Load the compressed data
    with open(input_file, 'rb') as f:
        compressed_data = pickle.load(f)

    codebook = compressed_data["codebook"]
    labels = compressed_data["labels"]
    original_shape = compressed_data["original_shape"]
    block_size = compressed_data["block_size"]
    padded_shape = compressed_data["padded_shape"]

    # Reconstruct the image from the labels and codebook
    blocks = codebook[labels]
    reconstructed_image = np.zeros(padded_shape, dtype=np.uint8)

    block_idx = 0
    for i in range(0, padded_shape[0], block_size):
        for j in range(0, padded_shape[1], block_size):
            block = blocks[block_idx].reshape((block_size, block_size))
            reconstructed_image[i:i+block_size, j:j+block_size] = block
            block_idx += 1

    # Crop the image to the original dimensions
    decompressed_image = reconstructed_image[:original_shape[0], :original_shape[1]]

    # Save the decompressed image
    cv2.imwrite(output_image_path, decompressed_image)

# Example Usage
# Compress
compress_image(r"CV2.jpg", block_size=4, codebook_size=16, output_file="compressed_data.pkl")

# Decompress
decompress_image("compressed_data.pkl", "decompressed_image.png")