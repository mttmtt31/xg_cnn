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
    parser.add_argument('--voronoi', action='store_true')
    parser.add_argument('--cones', action='store_true')
    parser.add_argument('--angle', action='store_true')
    
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

def main(s, plot_voronoi, plot_cones, plot_angle):
    with open('data/shots.json', 'rb') as f:
        dict_shots = json.load(f) 
    for shot_name, shot in tqdm(dict_shots.items(), total = len(dict_shots), desc = 'Saving shot frames.......'):
        # isolate ball location
        ball = np.array(shot['ball'])
        if plot_angle:
            players_to_plot = {k:v for k,v in shot.items() if k not in ['ball', 'outcome', 'match_id', 'index'] and is_point_inside_triangle(ball, v['location'][0], v['location'][1])}
            # isolate teammates location
            teammates = np.array([player['location'] for key, player in players_to_plot.items() if player['teammate']]).reshape(-1, 2)
            # isolate opponents location
            opponents = np.array([player['location'] for key, player in players_to_plot.items() if not player['teammate'] and player['position']['name'] != 'Goalkeeper']).reshape(-1, 2)
            # isolate gk location
            gk = np.array([player['location'] for key, player in players_to_plot.items() if not player['teammate'] and player['position']['name'] == 'Goalkeeper']).reshape(-1, 2)
        else:
            # isolate teammates location
            teammates = np.array([player['location'] for key, player in shot.items() if key not in ['ball', 'outcome'] and player['teammate']]).reshape(-1, 2)
            # isolate opponents location
            opponents = np.array([player['location'] for key, player in shot.items() if key not in ['ball', 'outcome'] and not player['teammate'] and player['position']['name'] != 'Goalkeeper']).reshape(-1, 2)
            # isolate gk location
            gk = np.array([player['location'] for key, player in shot.items() if key not in ['ball', 'outcome'] and not player['teammate'] and player['position']['name'] == 'Goalkeeper']).reshape(-1, 2)
        # put everything in a single dataframe with a column team which is a boolean for teammate or not. only for voronoi
        if plot_voronoi:
            opponents_gk = np.concatenate([opponents, gk])
            df_players = pd.DataFrame(np.concatenate([teammates, opponents_gk]))
            df_players['team'] = list(itertools.chain(*[[True]*len(teammates), [False]*(len(opponents_gk))]))

        # Setup the pitch
        pitch = VerticalPitch(line_alpha=0, goal_alpha=0, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, half=True, pitch_color='black')

        # We will use mplsoccer's grid function to plot a pitch with a title axis.
        fig, axs = pitch.draw()

        if plot_voronoi:
            # Plot Voronoi
            team1, team2 = pitch.voronoi(df_players[0], df_players[1], df_players['team'])
            t1 = pitch.polygon(team1, ax=axs, fc='red', alpha=0.2)
            t2 = pitch.polygon(team2, ax=axs, fc='blue', alpha=0.2)

        if plot_angle:
            # plot the angle to the goal
            pitch.goal_angle(ball[0], ball[1], ax=axs, alpha=0.2, color='#cb5a4c', goal='right')

        # Plot the players
        sc1 = pitch.scatter(teammates[:, 0], teammates[:, 1], c='red', label='Attacker',  ax=axs, s=s)
        sc2 = pitch.scatter(opponents[:, 0], opponents[:, 1], c='blue', label='Defender', ax=axs, s=s)
        sc4 = pitch.scatter(gk[:, 0], gk[:, 1], ax=axs, c='green', s=s)

        # plot the shot
        sc3 = pitch.scatter(ball[0], ball[1], c='black', ax=axs, s=s)

        if plot_cones:
        # Plotting circles, coloring the area, and filling between lines
            for players, color in zip([teammates, opponents], ["red", "blue"]):
                for i in range(len(players)):
                    center = (players[i, 0], players[i, 1])
                    radius = 8  # Adjust the radius as needed

                    # Calculate the angle based on the line connecting the current point to the fixed point
                    angle = np.degrees(np.arctan2(ball[1] - center[1], ball[0] - center[0]))
                    angle_range = [angle - 45 + 180, angle + 45 + 180]

                    # Plotting a portion of a circle within the desired angle range
                    theta = np.linspace(np.radians(angle_range[0]), np.radians(angle_range[1]), 100)
                    x_circle = center[0] + radius * np.cos(theta)
                    y_circle = center[1] + radius * np.sin(theta)

                    # Plotting the circle area
                    # plt.fill_between(x_circle, y_circle, center[1], color="salmon")

                    # Connecting lines
                    x_extreme_1 = center[0] + radius * np.cos(np.radians(angle_range[0]))
                    y_extreme_1 = center[1] + radius * np.sin(np.radians(angle_range[0]))
                    x_extreme_2 = center[0] + radius * np.cos(np.radians(angle_range[1]))
                    y_extreme_2 = center[1] + radius * np.sin(np.radians(angle_range[1]))

                    # Filling the area between the lines
                    plt.fill([players[i, 1], y_extreme_1, y_extreme_2], [players[i, 0], x_extreme_1, x_extreme_2], color=color, alpha = 0.2)

        # Set the figure size to 60x40 pixels
        fig.set_size_inches(1.5, 1.0)  # Inches are used for the size, so divide by the DPI

        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        # check if you need to plot voronoi for this goals
        if plot_voronoi:
            # save picture with all voronoi
            plt.savefig(f'images/all/{shot_name}.png', dpi=200, bbox_inches='tight', pad_inches=0, facecolor = 'black')
            # make the same picture but with only visible
            visible = pitch.polygon([np.array([[0,80.],[df_players[0].min() - 10,  80.], [df_players[0].min() - 10,  0], [0,0]])], color='white', ax=axs)
            plt.savefig(f'images/visible/{shot_name}.png', dpi=200, bbox_inches='tight', pad_inches=0, facecolor = 'black')
        elif plot_cones and plot_angle:
            # save picture with cones and angles
            plt.gca().set_aspect('equal')
            # Save the plot as a PNG image
            plt.savefig(f"images/cones+angle/{shot_name}.png", dpi=200, bbox_inches='tight', pad_inches=0, facecolor = 'black')            
        elif plot_cones:
            # Set the aspect ratio to 'equal' for circular circles
            plt.gca().set_aspect('equal')
            # Save the plot as a PNG image
            plt.savefig(f"images/cones/{shot_name}.png", dpi=200, bbox_inches='tight', pad_inches=0, facecolor = 'black')
        elif plot_angle:
            # save picture with angle
            plt.savefig(f'images/angle/{shot_name}.png', dpi=200, bbox_inches='tight', pad_inches=0, facecolor = 'black')
        else:
            # save picture with white background
            plt.savefig(f'images/white/{shot_name}.png', dpi=200, bbox_inches='tight', pad_inches=0, facecolor = 'black')
        # close the picture
        plt.close() 

if __name__ == '__main__':
    args = parse_args()
    if args.voronoi and (args.cones or args.angle):
        raise ValueError("Voronoi diagrams can only be plotted by themselves, without angles or cones.")
    main(s=args.s, plot_voronoi=args.voronoi, plot_cones=args.cones, plot_angle=args.angle)
    