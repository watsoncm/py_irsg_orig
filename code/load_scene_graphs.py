import os
import pdb
import json

with open('config.json') as f:
    cfg_data = json.load(f)
    vg_path = cfg_data['file_paths']['vg_data']


def load_scene_graphs():
    json_path = os.path.join(vg_path, 'scene_graphs.json')
    with open(json_path) as f:
        scene_graphs = json.load(f, object_hook=json_callback)


if __name__ == '__main__':
    load_scene_graphs()
