import json
import numpy as np
from tqdm import tqdm
import pandas as pd
import argparse
import warnings
import matplotlib.pyplot as plt
import itertools
from mplsoccer import VerticalPitch
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s', type=int, default=1)  
    parser.add_argument('--angle', action = 'store_true', help = 'If specified, plot the goal angle in transparency')  
    return parser.parse_args()

def area(x1, y1, x2, y2, x3, y3):
    return abs((x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))/2)

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
        return False

def main(s, plot_angle):
    with open('data/shots.json', 'rb') as f:
        dict_shots = json.load(f) 
    for shot_name, shot in tqdm(dict_shots.items(), total = len(dict_shots), desc = 'Saving shot frames.......'):
        # isolate ball location
        ball = np.array(shot['ball'])
        players_to_plot = {k:v for k,v in shot.items() if k not in ['ball', 'outcome', 'match_id', 'index'] and is_point_inside_triangle(ball, v['location'][0], v['location'][1])}
        # isolate teammates location
        teammates = np.array([player['location'] for key, player in players_to_plot.items() if player['teammate']]).reshape(-1, 2)
        # isolate opponents location
        opponents = np.array([player['location'] for key, player in players_to_plot.items() if not player['teammate'] and player['position']['name'] != 'Goalkeeper']).reshape(-1, 2)
        # isolate gk location
        gk = np.array([player['location'] for key, player in players_to_plot.items() if not player['teammate'] and player['position']['name'] == 'Goalkeeper']).reshape(-1, 2)
        
        # Setup the pitch
        pitch = VerticalPitch(line_alpha=0, goal_alpha=0, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, half=True, pitch_color='black')

        # We will use mplsoccer's grid function to plot a pitch with a title axis.
        fig, axs = pitch.draw()

        # plot the angle to the goal
        if plot_angle:
            pitch.goal_angle(ball[0], ball[1], ax=axs, alpha=0.2, color='#cb5a4c', goal='right')

        # Plot the players
        sc1 = pitch.scatter(teammates[:, 0], teammates[:, 1], c='red', label='Attacker',  ax=axs, s=s)
        sc2 = pitch.scatter(opponents[:, 0], opponents[:, 1], c='blue', label='Defender', ax=axs, s=s)
        sc4 = pitch.scatter(gk[:, 0], gk[:, 1], ax=axs, c='green', s=s)

        # plot the shot
        sc3 = pitch.scatter(ball[0], ball[1], c='black' if plot_angle else 'white', ax=axs, s=s)

        # crop to the last 40m
        plt.ylim(80, 120)

        # Set the figure size to 80x160 pixels
        fig.set_size_inches(2, 1) 

        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        # save picture with angle
        subfolder = 'angle' if plot_angle else 'white'
        plt.tight_layout()
        plt.savefig(f'images/{subfolder}/{shot_name}.png', dpi=80, pad_inches=0, facecolor = 'black')
        plt.close() 

if __name__ == '__main__':
    args = parse_args()
    main(s=args.s, plot_angle=args.angle)
    