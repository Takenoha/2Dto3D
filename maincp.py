import cv2
import numpy as np
import os
from stl import mesh
import heapq
from collections import defaultdict

def generate_3d_model(image_path, output_path, target_faces=1000):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # 画像をRGBで読み込む
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # 画像をグレースケールに変換する
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 画像の寸法を取得する
    height, width = gray_image.shape
    # 画像に最大の円を描く
    center = (width // 2, height // 2)
    radius = min(center)
    thickness = 2  # 線がつながるように厚さを増やす
    cv2.circle(gray_image, center, radius, (0, 0, 0), thickness)  # 白色
    
    # 円のマスクを作成する
    mask = np.zeros_like(gray_image)
    cv2.circle(mask, center, radius, (255, 255, 255), -1)
    
    # 円の外側を緑色（高さ0）に設定する
    gray_image[mask == 0] = 0
    
    # 画像を平滑化するためにガウシアンブラーを適用する
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # ピクセル値を0から1の範囲に正規化する
    normalized_image = blurred_image / 255.0
    
    # 正規化された値を0mmから3mmの範囲にスケーリングする
    z = normalized_image * 3.0
    
    # 画像の寸法を取得する
    height, width = normalized_image.shape
    
    # 3Dメッシュグリッドを作成する
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    
    # 頂点を作成する
    vertices = np.zeros((height * width, 3))
    vertices[:, 0] = x.flatten()
    vertices[:, 1] = y.flatten()
    vertices[:, 2] = z.flatten()
    
    # 面を作成する
    faces = []
    for i in range(height - 1):
        for j in range(width - 1):
            idx = i * width + j
            if mask[i, j] == 255 and mask[i + 1, j] == 255 and mask[i, j + 1] == 255 and mask[i + 1, j + 1] == 255:
                faces.append([idx, idx + width, idx + 1])
                faces.append([idx + 1, idx + width, idx + width + 1])
    
    # エッジの出現回数を数える
    edge_count = defaultdict(int)
    for face in faces:
        for i in range(3):
            v1, v2 = sorted([face[i], face[(i + 1) % 3]])
            edge_count[(v1, v2)] += 1

    # 一度だけ出現するエッジを見つける
    boundary_edges = [edge for edge, count in edge_count.items() if count == 1] 
    
    # 境界エッジの頂点を使って新しい頂点を作って、新しい面を作る
    new_boundary_vertices = []
    used_vertices = set()
    height_offset = 10.0  # 1cm = 10mm
    boundary_edgescpy = boundary_edges.copy()
    boundary_array = [boundary_edges[0][0], boundary_edges[0][1]]
    for i in range(len(boundary_edges)-1):
        for j in range(len(boundary_edges)-1):
            if boundary_array[1+i] == boundary_edges[j+1][0]:
                boundary_array.append(boundary_edges[j+1][1])
                del boundary_edges[j+1]
                break
            elif boundary_array[1+i] == boundary_edges[j+1][1]:
                boundary_array.append(boundary_edges[j+1][0])
                del boundary_edges[j+1]
                break
    print(boundary_array)
    
    # 新しい頂点を使って新しい面を作る
    new_faces = []
    new_vertex_indices = []
    for i in range(len(boundary_array)):
        new_vertex = np.copy(vertices[boundary_array[i]])
        new_vertex[2] = height_offset
        new_boundary_vertices.append(new_vertex)
        new_vertex_indices.append(len(vertices) + i)
    print(new_vertex_indices)
    new_boundary_vertices.append([center[0], center[1], height_offset])
    new_vertex_indices.append(len(vertices) + len(boundary_array))
    vertices = np.vstack((vertices, new_boundary_vertices))
    
    for i in range(len(boundary_array) - 1):
        new_faces.append([boundary_array[i],  new_vertex_indices[i],boundary_array[i + 1]])
        new_faces.append([boundary_array[i + 1], new_vertex_indices[i], new_vertex_indices[i + 1]])
        new_faces.append([new_vertex_indices[i+1], new_vertex_indices[i],new_vertex_indices[-1]])
    
    # 古い面と新しい面を結合する
    all_faces = np.vstack((faces, new_faces))

    # 最終的なメッシュを作成する
    final_mesh = mesh.Mesh(np.zeros(all_faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, face in enumerate(all_faces):
        for j in range(3):
            final_mesh.vectors[i][j] = vertices[face[j]]

    # 最終的なメッシュをファイルに保存する
    try:
        final_mesh.save(output_path)
        print(f"Mesh saved to {output_path}")
    except Exception as e:
        print(f"Failed to save the mesh: {e}")

# 使用例
image_path = 'test.png'
output_path = 'output_model_with_boundary.stl'
target_faces = 1000  # 面の目標数を設定するためにこの値を調整する
generate_3d_model(image_path, output_path, target_faces)