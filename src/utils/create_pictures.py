from statsbombpy import sb
import numpy as np
from tqdm import tqdm
import pandas as pd
import argparse
import warnings
import matplotlib.pyplot as plt
import itertools
from mplsoccer import Pitch
warnings.filterwarnings("ignore")

def boolean_string(s):
    if s.lower() not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s.lower() == 'true'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s', type=int, default=5)
    parser.add_argument('--voronoi', type=boolean_string, default=False)
    parser.add_argument('--cones', type=boolean_string, default=False)
    parser.add_argument('--output-path', type=str)
    
    return parser.parse_args()


def main(s, plot_voronoi, plot_cones, output_path):
    # read competitions
    df_competitions = sb.competitions()
    # get rid of barcelona and invincible season
    # TODO: maybe we can consider their opponents' shot, at least?
    df_competitions = df_competitions[~df_competitions['competition_name'].isin(['La Liga', 'Premier League'])]

    # find games
    all_matches = []
    for i, row in tqdm(df_competitions.iterrows(), total = len(df_competitions), desc = 'Scanning competitions to find games'):
        try:
            all_matches.extend(sb.matches(competition_id=row['competition_id'], season_id=row['season_id'])['match_id'].tolist())
        except:
            continue

    print('Games have been individuated.')

    # keep track of how many shots
    num_shots = 0
    num_goals = 0
    # for every game
    for match_id in tqdm(all_matches, total = len(all_matches), desc = 'Scanning games, looking for shots.'):
        # extract all shots
        df_shots = sb.events(match_id, split=True)['shots']
        # isolate open-play shots
        df_shots = df_shots.loc[(~df_shots['shot_freeze_frame'].isna()) & (df_shots['shot_type'] == 'Open Play'), ['location', 'shot_freeze_frame', 'shot_outcome']].reset_index(drop = True)
        # for every shot
        for _, shot in df_shots.iterrows():
            # isolate ball location
            ball = np.array(shot['location'])
            # isolate teammates location
            teammates = np.array([player['location'] for player in shot['shot_freeze_frame'] if player['teammate']]).reshape(-1, 2)
            # isolate opponents location
            opponents = np.array([player['location'] for player in shot['shot_freeze_frame'] if not player['teammate'] and player['position']['name'] != 'Goalkeeper']).reshape(-1, 2)
            # isolate gk location
            gk = np.array([player['location'] for player in shot['shot_freeze_frame'] if not player['teammate'] and player['position']['name'] == 'Goalkeeper']).reshape(-1, 2)
            # put everything in a single dataframe with a column team which is a boolean for teammate or not. only for voronoi
            if plot_voronoi:
                opponents_gk = np.concatenate([opponents, gk])
                df_players = pd.DataFrame(np.concatenate([teammates, opponents_gk]))
                df_players['team'] = list(itertools.chain(*[[True]*len(teammates), [False]*(len(opponents_gk))]))

            # Setup the pitch
            pitch = Pitch(line_alpha=0, goal_alpha=0, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0)

            # We will use mplsoccer's grid function to plot a pitch with a title axis.
            fig, axs = pitch.draw()

            if plot_voronoi:
                # Plot Voronoi
                team1, team2 = pitch.voronoi(df_players[0], df_players[1], df_players['team'])
                t1 = pitch.polygon(team1, ax=axs, fc='red', alpha=0.2)
                t2 = pitch.polygon(team2, ax=axs, fc='blue', alpha=0.2)

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
                        plt.fill([players[i, 0], x_extreme_1, x_extreme_2], [players[i, 1], y_extreme_1, y_extreme_2], color=color, alpha = 0.2)

            # Set the figure size to 60x40 pixels
            fig.set_size_inches(1.5, 1.0)  # Inches are used for the size, so divide by the DPI

            # Save the plot as a PNG image
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
            # check if this was a goal or not
            if shot["shot_outcome"].lower() == "goal":
                # check if you need to plot voronoi for this goals
                if plot_voronoi:
                    # save picture with all voronoi
                    plt.savefig(f'{output_path}/all/goals/{num_goals}.png', dpi=200, bbox_inches='tight', pad_inches=0)
                    # make the same picture but with only visible
                    visible = pitch.polygon([np.array([[0,80.],[df_players[0].min() - 10,  80.], [df_players[0].min() - 10,  0], [0,0]])], color='white', ax=axs)
                    plt.savefig(f'{output_path}/visible/goals/{num_goals}.png', dpi=200, bbox_inches='tight', pad_inches=0)
                elif plot_cones:
                    # Set the aspect ratio to 'equal' for circular circles
                    plt.gca().set_aspect('equal')
                    # Save the plot as a PNG image
                    plt.savefig(f"{output_path}/cones/goals/{num_goals}.png", dpi=200, bbox_inches='tight', pad_inches=0)
                else:
                    # save picture with white background
                    plt.savefig(f'{output_path}/white/goals/{num_goals}.png', dpi=200, bbox_inches='tight', pad_inches=0)
                # increment the number of goals
                num_goals = num_goals + 1
            else:
                # check if you need to plot voronoi for this shot
                if plot_voronoi:
                    # save picture with all voronoi
                    plt.savefig(f'{output_path}/all/non_goals/{num_shots}.png', dpi=200, bbox_inches='tight', pad_inches=0)
                    # make the same picture but with only visible
                    visible = pitch.polygon([np.array([[0,80.],[df_players[0].min() - 10,  80.], [df_players[0].min() - 10,  0], [0,0]])], color='white', ax=axs)
                    plt.savefig(f'{output_path}/visible/non_goals/{num_shots}.png', dpi=200, bbox_inches='tight', pad_inches=0)
                elif plot_cones:
                    # Set the aspect ratio to 'equal' for circular circles
                    plt.gca().set_aspect('equal')
                    # Save the plot as a PNG image
                    plt.savefig(f"{output_path}/cones/non_goals/{num_shots}.png", dpi=200, bbox_inches='tight', pad_inches=0)
                else:
                    # save picture with white background
                    plt.savefig(f'{output_path}/white/non_goals/{num_shots}.png', dpi=200, bbox_inches='tight', pad_inches=0)
                # increment the number of shots
                num_shots = num_shots + 1
            # close the picture
            plt.close() 

if __name__ == '__main__':
    args = parse_args()
    if args.voronoi and args.cones:
        raise ValueError("You are trying to plot both Voronoi and cones, but this is not possible. Please, select either or none.")
    main(s=args.s, plot_voronoi=args.voronoi, plot_cones=args.cones, output_path=args.output_path)
    