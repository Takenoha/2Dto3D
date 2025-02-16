import numpy as np
from stl import mesh
import cv2
import os

def generate_3d_model(image_path, target_faces=1000):
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
    
    # Add diagonal faces to smooth the outer edge
    for i in range(height - 1):
        for j in range(width - 1):
            idx = i * width + j
            if mask[i, j] == 255 and (mask[i + 1, j] == 0 or mask[i, j + 1] == 0):
                if mask[i + 1, j] == 0 and mask[i, j + 1] == 255:
                    faces.append([idx, idx, idx + 1])
                if mask[i, j + 1] == 0 and mask[i + 1, j] == 255:
                    faces.append([idx, idx, idx + width])
    
    vertices = [v for v in vertices if v is not None]
    faces = [face for face in faces if len(set(face)) == 3]

    return np.array(vertices), np.array(faces)

def create_cylinder_with_custom_base(base_vertices, base_faces, height, output_path):
    num_base_vertices = len(base_vertices)
    
    # Create the top vertices by shifting the base vertices up by the height
    top_vertices = np.copy(base_vertices)
    top_vertices[:, 2] += height
    
    # Combine the base and top vertices
    vertices = np.vstack((base_vertices, top_vertices))
    
    # Create the side faces
    side_faces = []
    for i in range(num_base_vertices):
        next_i = (i + 1) % num_base_vertices
        side_faces.append([i, next_i, i + num_base_vertices])
        side_faces.append([next_i, next_i + num_base_vertices, i + num_base_vertices])
    
    # Combine all faces
    faces = np.vstack((base_faces, side_faces))
    
    # Create the mesh
    cylinder_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            cylinder_mesh.vectors[i][j] = vertices[face[j]]
    
    # Save the mesh to file
    try:
        cylinder_mesh.save(output_path)
        print(f"Mesh saved to {output_path}")
    except Exception as e:
        print(f"Failed to save the mesh: {e}")

# Example usage
image_path = 'test.png'
output_path = 'cylinder_with_custom_base.stl'
height = 10.0  # Height of the cylinder

base_vertices, base_faces = generate_3d_model(image_path)
create_cylinder_with_custom_base(base_vertices, base_faces, height, output_path)