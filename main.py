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
    
    # 画像の寸法を取得する
    height, width = image.shape[:2]
    
    # 画像に最大の円を描く
    center = (width // 2, height // 2)
    radius = min(center)
    thickness = 2  # 線がつながるように厚さを増やす
    cv2.circle(image, center, radius, (0, 255, 0), thickness)  # 緑色
    
    # 円のマスクを作成する
    mask = np.zeros_like(image[:, :, 0])
    cv2.circle(mask, center, radius, (255, 255, 255), -1)
    
    # 円の外側を緑色（高さ0）に設定する
    image[mask == 0] = [0, 255, 0]
    
    # 画像をグレースケールに変換する
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # ピクセル値を0から1の範囲に正規化する
    normalized_image = gray_image / 255.0
    
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
    
    # 外縁を滑らかにするために対角線の面を追加する
    for i in range(height - 1):
        for j in range(width - 1):
            idx = i * width + j
            if mask[i, j] == 255 and (mask[i + 1, j] == 0 or mask[i, j + 1] == 0):
                if mask[i + 1, j] == 0 and mask[i, j + 1] == 255:
                    faces.append([idx, idx, idx + 1])
                if mask[i, j + 1] == 0 and mask[i + 1, j] == 255:
                    faces.append([idx, idx, idx + width])
    
    # QEMアルゴリズム
    def compute_quadric(v1, v2, v3):
        normal = np.cross(v2 - v1, v3 - v1)
        norm = np.linalg.norm(normal)
        if norm == 0:
            return np.zeros((4, 4))
        normal = normal / norm
        d = -np.dot(normal, v1)
        q = np.outer(normal, normal)
        q = np.pad(q, ((0, 1), (0, 1)), mode='constant', constant_values=0)
        q[3, 3] = d * d
        return q

    def compute_error(q, v):
        v_hom = np.append(v, 1)
        return np.dot(v_hom, np.dot(q, v_hom))

    def collapse_edge(v1, v2, q1, q2):
        q = q1 + q2
        try:
            v_new = np.linalg.solve(q[:3, :3], -q[:3, 3])
        except np.linalg.LinAlgError:
            v_new = (v1 + v2) / 2
        return v_new

    quadrics = [np.zeros((4, 4)) for _ in range(len(vertices))]
    for face in faces:
        v1, v2, v3 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        q = compute_quadric(v1, v2, v3)
        quadrics[face[0]] += q
        quadrics[face[1]] += q
        quadrics[face[2]] += q

    edge_queue = []
    edge_map = {}
    for i, face in enumerate(faces):
        for j in range(3):
            v1, v2 = sorted([face[j], face[(j + 1) % 3]])
            if (v1, v2) not in edge_map:
                edge_map[(v1, v2)] = i
                error = compute_error(quadrics[v1] + quadrics[v2], (vertices[v1] + vertices[v2]) / 2)
                heapq.heappush(edge_queue, (error, v1, v2))

    while len(faces) > target_faces and edge_queue:
        error, v1, v2 = heapq.heappop(edge_queue)
        if v1 not in vertices or v2 not in vertices:
            continue
        v_new = collapse_edge(vertices[v1], vertices[v2], quadrics[v1], quadrics[v2])
        vertices[v1] = v_new
        quadrics[v1] += quadrics[v2]
        vertices[v2] = None
        for i, face in enumerate(faces):
            if v2 in face:
                face[face.index(v2)] = v1
            if len(set(face)) < 3:
                faces.pop(i)

    vertices = [v for v in vertices if v is not None]
    faces = [face for face in faces if len(set(face)) == 3]

    # メッシュを作成する
    faces = np.array(faces)
    model_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            model_mesh.vectors[i][j] = vertices[face[j]]

    # メッシュをファイルに保存する
    try:
        model_mesh.save(output_path)
    except Exception as e:
        print(f"Failed to save the mesh: {e}")

    # エッジの出現回数を数える
    edge_count = defaultdict(int)
    for face in faces:
        for i in range(3):
            v1, v2 = sorted([face[i], face[(i + 1) % 3]])
            edge_count[(v1, v2)] += 1

    # 一度だけ出現するエッジを見つける
    boundary_edges = [edge for edge, count in edge_count.items() if count == 1]

    # 境界エッジから頂点を抽出する
    boundary_vertices = set()
    for v1, v2 in boundary_edges:
        boundary_vertices.add(v1)
        boundary_vertices.add(v2)

    boundary_vertices = list(boundary_vertices)
    print("Boundary vertices:", boundary_vertices)

    # 境界頂点の1cm上に新しい頂点を作成する
    height_offset = 10.0  # 1cm = 10mm
    new_vertices = []
    for v in boundary_vertices:
        new_vertex = np.copy(vertices[v])
        new_vertex[2] += height_offset
        new_vertices.append(new_vertex)
    
    new_vertices = np.array(new_vertices)
    vertices = np.vstack((vertices, new_vertices))

    # 新しい頂点のリストを表示する
    print("New vertices:", new_vertices)

    # 新しい頂点とエッジのリストを保存する
    new_vertex_list = new_vertices.tolist()
    new_edges = [(i, (i + 1) % len(new_vertices)) for i in range(len(new_vertices))]
    print("New vertex list:", new_vertex_list)
    print("New edges:", new_edges)

    # 新しい頂点と元の頂点の間に面を作成する
    new_faces = []
    num_boundary_vertices = len(boundary_vertices)
    num_vertices = len(vertices) - num_boundary_vertices
    for i in range(num_boundary_vertices):
        v1 = boundary_vertices[i]
        v2 = boundary_vertices[(i + 1) % num_boundary_vertices]
        v1_new = num_vertices + i
        v2_new = num_vertices + (i + 1) % num_boundary_vertices
        new_faces.append([v1, v1_new, v2])
        new_faces.append([v1_new, v2_new, v2])

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