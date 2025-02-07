import cv2
import numpy as np
from stl import mesh

def generate_3d_model(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply Gaussian blur to smooth the image
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Normalize the pixel values to range from 0 to 1
    normalized_image = blurred_image / 255.0
    
    # Get the dimensions of the image
    height, width = normalized_image.shape
    
    # Create a 3D mesh grid
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    x, y = np.meshgrid(x, y)
    z = normalized_image
    
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
            faces.append([idx, idx + 1, idx + width])
            faces.append([idx + 1, idx + width + 1, idx + width])
    
    faces = np.array(faces)
    
    # Create the mesh
    model = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            model.vectors[i][j] = vertices[f[j], :]
    
    # Save the mesh to file
    model.save(output_path)

if __name__ == "__main__":
    image_path = "path_to_your_image.jpg"
    output_path = "output_model.stl"
    generate_3d_model(image_path, output_path)