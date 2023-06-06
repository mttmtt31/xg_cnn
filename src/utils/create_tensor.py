import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import raster_geometry as rg
import json
import math
from tqdm import tqdm

def area(x1, y1, x2, y2, x3, y3):
    return abs((x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))/2)

def full_triangle(a):
    a = [min(max_coord, math.floor(coord * 2)) for coord, max_coord in zip(a, [239, 159])]
    b = (239, 71)
    c = (239, 87)
    ab = rg.bresenham_line(a, b, endpoint=True)
    for x in set(ab):
        yield from rg.bresenham_line(c, x, endpoint=True)

def is_point_inside_triangle(ball, px, py):
    x1, y1 = ball
    x2, y2 = 120, 36
    x3, y3 = 120, 44
    triangle_area = area(x1, y1, x2, y2, x3, y3)
    area1 = area(px, py, x2, y2, x3, y3)
    area2 = area(x1, y1, px, py, x3, y3)
    area3 = area(x1, y1, x2, y2, px, py)

    if triangle_area - 0.01 <= area1 + area2 + area3 <= triangle_area + 0.01:
        return True
    else:
        return 
    
with open('data/shots.json', 'rb') as f:
    dict_shots = json.load(f) 

all_shots = []
labels = []

for shot_name, shot in tqdm(dict_shots.items(), total = len(dict_shots)):
    # initialise the pitch
    frame_array = np.zeros((2, 240, 160))
    # find the players
    players = {k:v for k, v in shot.items() if k.isdigit() and is_point_inside_triangle(shot['ball'], v['location'][0], v['location'][1])}
    for _, player in players.items(): 
        frame_array[int(player['teammate']), min(239, math.floor(player['location'][0] * 2)), min(159, math.floor(player['location'][1] * 2))] += 1

    coords = set(full_triangle(shot['ball']))
    arr = np.array(rg.render_at((240, 160), coords))
    arr = np.expand_dims(arr, axis = 0)
    shot_frame = np.concatenate((frame_array, arr), axis=0)[:, -40:, :]
    all_shots.append(shot_frame)
    labels.append(0 if shot_name.startswith('n') else 1)

all_shots = np.stack(all_shots)
labels = np.array(labels)

np.save('data/shots.npy', all_shots)
np.save('data/labels.npy', labels)