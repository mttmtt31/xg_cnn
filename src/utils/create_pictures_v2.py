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
    parser.add_argument('--save', action='store_true')
    
    return parser.parse_args()

def compute_distance(ball):
    # compute the distance of the ball from the centre of the goal
    return round(np.sqrt((ball[0] - 120)**2 + (ball[1] - 40)**2), 3)

def compute_angle(ball):
    # Calculate the slopes of the lines connecting the ball with the posts
    slope1 = abs(45 - ball[1]) / (120 - ball[0])
    slope2 = abs(35 - ball[1]) / (120 - ball[0])

    # Calculate the angles using the arctan formula
    angle1 = np.arctan2(slope1, 1)  # Angle with the positive x-axis for line connecting (60, 60) and (120, 35)
    angle2 = np.arctan2(slope2, 1)  # Angle with the positive x-axis for line connecting (60, 60) and (120, 45)

    # Compute the angle difference
    return round(abs(angle1 - angle2), 3)

def crop_image(ball, teammates, opponents, gk):
    # Check if the arrays are empty and assign a default value if they are
    teammates_min = float('inf') if len(teammates) == 0 else min(teammates[:, 0])
    opponents_min = float('inf') if len(opponents) == 0 else min(opponents[:, 0])
    gk_min = float('inf') if len(gk) == 0 else min(gk[:, 0])
    ball_min = ball[0]

    return min(teammates_min, opponents_min, gk_min, ball_min)

def main(s, plot_voronoi, plot_cones, plot_angle, save):
    with open('data/shots.json', 'rb') as f:
        dict_shots = json.load(f) 
    for shot_name, shot in tqdm(dict_shots.items(), total = len(dict_shots), desc = 'Saving shot frames.......'):
        # isolate ball location
        ball = np.array(shot['ball'])
        # compute distance and angle
        shot_dist = compute_distance(ball)
        shot_angle = compute_angle(ball)
        # isolate teammates location
        teammates = np.array([player['location'] for key, player in shot.items() if key not in ['ball', 'outcome', 'match_id', 'index'] and player['teammate']]).reshape(-1, 2)
        # isolate opponents location
        opponents = np.array([player['location'] for key, player in shot.items() if key not in ['ball', 'outcome', 'match_id', 'index'] and not player['teammate'] and player['position']['name'] != 'Goalkeeper']).reshape(-1, 2)
        # isolate gk location
        gk = np.array([player['location'] for key, player in shot.items() if key not in ['ball', 'outcome', 'match_id', 'index'] and not player['teammate'] and player['position']['name'] == 'Goalkeeper']).reshape(-1, 2)
        # put everything in a single dataframe with a column team which is a boolean for teammate or not. only for voronoi
        if plot_voronoi:
            opponents_gk = np.concatenate([opponents, gk])
            df_players = pd.DataFrame(np.concatenate([teammates, opponents_gk]))
            df_players['team'] = list(itertools.chain(*[[True]*len(teammates), [False]*(len(opponents_gk))]))

        # We will use mplsoccer's grid function to plot a pitch with a title axis.
        fig, axs = plt.subplots(figsize=(100/80, 100/80), dpi=80) 

        # Setup the pitch
        pitch = VerticalPitch(line_alpha=0, goal_alpha=0, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0)

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
        sc3 = pitch.scatter(ball[0], ball[1], c='white', ax=axs, s=s)

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

        # crop the picture at the right dimension
        axs.set_ylim(crop_image(ball=ball, teammates=teammates, opponents=opponents, gk=gk), None)

        # turn off axis
        axs.axis('off')

        # Save the plot as a PNG image
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        # check if you need to plot voronoi for this goals
        if save:
            if plot_voronoi:
                # save picture with all voronoi
                plt.savefig(f'new_images/all/{shot_name}_{shot_dist}_{shot_angle}.png', dpi=80, bbox_inches='tight', pad_inches=0, facecolor = 'black')
                # make the same picture but with only visible
                visible = pitch.polygon([np.array([[0,80.],[df_players[0].min() - 10,  80.], [df_players[0].min() - 10,  0], [0,0]])], color='white', ax=axs)
                plt.savefig(f'new_images/visible/{shot_name}_{shot_dist}_{shot_angle}.png', dpi=80, bbox_inches='tight', pad_inches=0, facecolor = 'black')
            elif plot_cones and plot_angle:
                # save picture with cones and angles
                plt.gca().set_aspect('equal')
                # Save the plot as a PNG image
                plt.savefig(f"new_images/cones+angle/{shot_name}_{shot_dist}_{shot_angle}.png", dpi=80, bbox_inches='tight', pad_inches=0, facecolor = 'black')            
            elif plot_cones:
                # Set the aspect ratio to 'equal' for circular circles
                plt.gca().set_aspect('equal')
                # Save the plot as a PNG image
                plt.savefig(f"new_images/cones/{shot_name}_{shot_dist}_{shot_angle}.png", dpi=80, bbox_inches='tight', pad_inches=0, facecolor = 'black')
            elif plot_angle:
                # save picture with angle
                plt.savefig(f'new_images/angle/{shot_name}_{shot_dist}_{shot_angle}.png', dpi=80, bbox_inches='tight', pad_inches=0, facecolor = 'black')
            else:
                # save picture with white background
                plt.savefig(f'new_images/white/{shot_name}_{shot_dist}_{shot_angle}.png', dpi=80, bbox_inches='tight', pad_inches=0, facecolor = 'black')
        # close the picture
        plt.close() 

if __name__ == '__main__':
    args = parse_args()
    if args.voronoi and (args.cones or args.angle):
        raise ValueError("Voronoi diagrams can only be plotted by themselves, without angles or cones.")
    main(s=args.s, plot_voronoi=args.voronoi, plot_cones=args.cones, plot_angle=args.angle, save=args.save)
    