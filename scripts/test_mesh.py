import trimesh

path = "data/0a3e50926ca64400be27f4316f746b29.obj"

mesh = trimesh.load(path)
print(mesh.vertices.shape)
print(mesh.faces.shape)
print(mesh.is_watertight)

mesh = mesh.split()

print(len(mesh))