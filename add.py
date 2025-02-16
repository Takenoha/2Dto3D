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