import cv2
import numpy as np
import os
from stl import mesh

def generate_3d_model(image_path, output_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Get the dimensions of the image
    height, width = image.shape
    
    # Draw the largest possible circle in the image
    center = (width // 2, height // 2)
    radius = min(center)
    thickness = 2  # Increase thickness to ensure lines connect
    cv2.circle(image, center, radius, (255, 255, 255), thickness)  # White color
    
    # Create a mask for the circle
    mask = np.zeros_like(image)
    cv2.circle(mask, center, radius, (255, 255, 255), -1)
    
    # Set the outside of the circle to green (height 0)
    image[mask == 0] = 0
    
    # Apply Gaussian blur to smooth the image
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Normalize the pixel values to range from 0 to 1
    normalized_image = blurred_image / 255.0
    
    # Scale the normalized values to range from 0mm to 3mm
    z = normalized_image * 3.0
    
    # Get the dimensions of the image
    height, width = normalized_image.shape
    
    # Create a 3D mesh grid
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    
    # Create vertices
    vertices = np.zeros((height * width, 3))
    vertices[:, 0] = x.flatten()
    vertices[:, 1] = y.flatten()
    vertices[:, 2] = z.flatten()
    
    # Create faces
    faces = []
    for i in range(height - 1):
        for j in range(width - 1):
            idx = i * width + j
            if mask[i, j] == 255 and mask[i + 1, j] == 255 and mask[i, j + 1] == 255 and mask[i + 1, j + 1] == 255:
                faces.append([idx, idx + width, idx + 1])
                faces.append([idx + 1, idx + width, idx + width + 1])
    
    # Add a new set of vertices for the 1cm height
    outer_height = 10.0  # 1cm = 10mm
    top_vertices = vertices.copy()
    top_vertices[:, 2] = outer_height  # Set z to 1cm for top vertices
    vertices = np.vstack((vertices, top_vertices))
    
    num_vertices = height * width
    for i in range(height - 1):
        for j in range(width - 1):
            idx = i * width + j
            if mask[i, j] == 255 and mask[i + 1, j] == 255 and mask[i, j + 1] == 255 and mask[i + 1, j + 1] == 255:
                faces.append([idx + num_vertices, idx + 1 + num_vertices, idx + width + num_vertices])
                faces.append([idx + 1 + num_vertices, idx + width + 1 + num_vertices, idx + width + num_vertices])
    
    # Add side faces to connect top and bottom
    for i in range(height - 1):
        for j in range(width - 1):
            idx = i * width + j
            if mask[i, j] == 255:
                faces.append([idx, idx + num_vertices, idx + 1])
                faces.append([idx + 1, idx + num_vertices, idx + 1 + num_vertices])
                faces.append([idx + width, idx + width + num_vertices, idx + width + 1])
                faces.append([idx + width + 1, idx + width + num_vertices, idx + width + 1 + num_vertices])
    
    # Create the mesh
    faces = np.array(faces)
    model_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            model_mesh.vectors[i][j] = vertices[face[j], :]
    
    # Save the mesh to file
    try:
        model_mesh.save(output_path)
    except Exception as e:
        print(f"Failed to save the mesh: {e}")

# Example usage
image_path = 'test.png'
output_path = 'output_model.stl'
generate_3d_model(image_path, output_path)