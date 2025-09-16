import pymesh
import time
import json

with open('pymesh_config.json', 'rb') as f:
    config = json.load(f)

models = config["model path abs"]
thresh = config["maximum edge length"]

for model in models:

    mesh_i = pymesh.load_mesh(model)

    for thresh_i in thresh:
        time_in = time.time()
        mesh_o, info = pymesh.split_long_edges(mesh_i, thresh_i)
        pymesh.save_mesh(f'{model}_{thresh_i}.obj', mesh_o)
        print(f'{model}: face edge threshold {thresh_i}, done in {time.time() - time_in}s')