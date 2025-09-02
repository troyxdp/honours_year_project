import os
import csv
import shutil

import tables
import pandas as pd

from classes.song import Song
from classes.hdf5_utils.dataset_creator import create_track_file
from classes.trainer import Trainer
from classes.neural_network import NeuralNetwork

def create_test_dataset(dataset_path, csv_path, output_path):
    # Get dataset file paths
    trainer = Trainer(
        training_network=NeuralNetwork(202, 202),
        initial_lr=0.001,
        final_lr=0.0001,
        num_epochs=120,
        dataset_path=dataset_path,
        output_folder='/home/troyxdp/Documents/University Work/HYP/HYP Source Code/Back End/networks/experiment_2'
    )
    file_paths = trainer.get_file_paths()

    # Iterate through files
    print()
    for file_path in sorted(file_paths):
        # Create potential directory and file paths
        song_id = os.path.splitext(os.path.basename(file_path))[0]
        potential_dir_path = os.path.join(output_path, song_id[2], song_id[3], song_id[4])
        potential_file_name = f"{song_id}.csv"
        potential_file_path = os.path.join(potential_dir_path, potential_file_name)

        # Search for song ID in Echo Nest Taste Profile dataset
        found = False
        print(f"\nSearching for song with ID {song_id}...")
        for chunk in pd.read_csv(csv_path, delimiter=',', chunksize=1000):
            for row in chunk.itertuples(index=True):
                # Get track ID in Echo Nest Taste Profiles dataset and check if it is equal to the one of the track we are currently searching for
                row_track_id = row.track_id
                if row_track_id == song_id:
                    # Create a directory path corresponding to the path of the dataset item if not already existing
                    if not os.path.isdir(potential_dir_path):
                        os.makedirs(potential_dir_path)

                    # Create the file where user plays are listed if it does not exist
                    if not os.path.exists(potential_file_path):
                        with open(potential_file_path, 'w') as f:
                            print(f"Creating file for song with ID {row_track_id}...")
                            writer = csv.writer(f, delimiter=',')
                            writer.writerow(["user_id", "track_id", 'play_count']) # Write the column names

                    # Append data to CSV file
                    with open(potential_file_path, 'a') as f:
                        writer = csv.writer(f, delimiter=',')
                        writer.writerow([row.user_id, row_track_id, row.play_count])

                    found = True

        if not found:
            print(f"Could not find track with ID {song_id} in Echo Nest Taste Profiles dataset")

def move_songs_to_test_folder(full_dataset_path, entp_path, output_path):
    # Get dataset file paths
    trainer = Trainer(
        training_network=NeuralNetwork(202, 202),
        initial_lr=0.001,
        final_lr=0.0001,
        num_epochs=120,
        dataset_path=full_dataset_path,
        output_folder='/home/troyxdp/Documents/University Work/HYP/HYP Source Code/Back End/networks/experiment_2'
    )
    file_paths = trainer.get_file_paths()
    file_paths = sorted(file_paths)

    print()
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        potential_entp_path = os.path.join(entp_path, file_name[2], file_name[3], file_name[4], f"{os.path.splitext(file_name)[0]}.csv")
        if os.path.exists(potential_entp_path):
            test_dataset_dir = os.path.join(output_path, file_name[2], file_name[3], file_name[4])
            if not os.path.isdir(test_dataset_dir):
                os.makedirs(test_dataset_dir)
            shutil.move(file_path, test_dataset_dir)

def extract_linked_songs(test_dataset_path, output_path):
    # Get dataset file paths
    trainer = Trainer(
        training_network=NeuralNetwork(202, 202),
        initial_lr=0.001,
        final_lr=0.0001,
        num_epochs=120,
        dataset_path=test_dataset_path,
        output_folder='/home/troyxdp/Documents/University Work/HYP/HYP Source Code/Back End/networks/experiment_2'
    )
    file_paths = trainer.get_file_paths()
    file_paths = sorted(file_paths)

    # Loop through file paths - OUTER LOOP
    print()
    for o, file_path_o in enumerate(file_paths):
        print(f"Searching for links for file {file_path_o}...")
        df_o = pd.read_csv(file_path_o, delimiter=',')
        # Loop through all file paths after file_path_o - INNER LOOP
        for file_path_i in file_paths[o+1:]:
            df_i = pd.read_csv(file_path_i, delimiter=',')
            # Merge the dataframes
            merged_df = df_o.merge(df_i, left_on='user_id', right_on='user_id')
            # Check if there is any intersection
            if len(merged_df) > 0:
                # Get file and dir paths
                basename_o = os.path.basename(file_path_o)
                basename_i = os.path.basename(file_path_i)
                potential_dir_path_o = os.path.join(output_path, basename_o[2], basename_o[3], basename_o[4])
                potential_file_path_o = os.path.join(potential_dir_path_o, basename_o)
                potential_dir_path_i = os.path.join(output_path, basename_i[2], basename_i[3], basename_i[4])
                potential_file_path_i = os.path.join(potential_dir_path_i, basename_i)
                # Check if dir paths exist and make them if they don't
                if not os.path.isdir(potential_dir_path_o):
                    os.makedirs(potential_dir_path_o)
                if not os.path.isdir(potential_dir_path_i):
                    os.makedirs(potential_dir_path_i)
                # Copy files (if they don't exist)
                if not os.path.exists(potential_file_path_o):
                    print(f"Found link at least one link...")
                    shutil.copy(file_path_o, potential_dir_path_o)
                if not os.path.exists(potential_file_path_i):
                    shutil.copy(file_path_i, potential_dir_path_i)
        print()

def extract_song_links(test_dataset_path, output_path):
    # Get dataset file paths
    trainer = Trainer(
        training_network=NeuralNetwork(202, 202),
        initial_lr=0.001,
        final_lr=0.0001,
        num_epochs=120,
        dataset_path=test_dataset_path,
        output_folder='/home/troyxdp/Documents/University Work/HYP/HYP Source Code/Back End/networks/experiment_2'
    )
    file_paths = trainer.get_file_paths()
    file_paths = sorted(file_paths)

    # Create number of links tracker
    num_links_per_file = {}
    for file_path in file_paths:
        num_links_per_file[file_path] = 0

    # Loop through file paths - OUTER LOOP
    print()
    for o, file_path_o in enumerate(file_paths):
        print(f"Searching for links for file {file_path_o}...")
        df_o = pd.read_csv(file_path_o, delimiter=',')
        
        # Loop through all file paths after file_path_o - INNER LOOP
        for file_path_i in file_paths[o+1:]:
            df_i = pd.read_csv(file_path_i, delimiter=',')
            # Merge the dataframes
            merged_df = df_o.merge(df_i, left_on='user_id', right_on='user_id')
            # Check if there is any intersection
            if len(merged_df) > 0:
                # Get file and dir paths
                basename_o = os.path.basename(file_path_o)
                basename_i = os.path.basename(file_path_i)
                potential_dir_path_o = os.path.join(output_path, basename_o[2], basename_o[3], basename_o[4])
                potential_file_path_o = os.path.join(potential_dir_path_o, f'{os.path.splitext(basename_o)[0]}.txt')
                potential_dir_path_i = os.path.join(output_path, basename_i[2], basename_i[3], basename_i[4])
                potential_file_path_i = os.path.join(potential_dir_path_i, f'{os.path.splitext(basename_i)[0]}.txt')
                # Check if dir paths exist and make them if they don't
                if not os.path.isdir(potential_dir_path_o):
                    os.makedirs(potential_dir_path_o)
                if not os.path.isdir(potential_dir_path_i):
                    os.makedirs(potential_dir_path_i)
                # Copy files (if they don't exist)
                with open(potential_file_path_o, 'a') as f:
                    f.writelines([file_path_i, '\n'])
                    num_links_per_file[file_path_o] += 1
                with open(potential_file_path_i, 'a') as f:
                    f.writelines([file_path_o, '\n'])
                    num_links_per_file[file_path_i] += 1
        
        # Print the number of links found for a file
        print(f"Number of links found for file: {num_links_per_file[file_path_o]}")
        print()

    # Convert num_links to lists
    num_links_per_file_paths = list(num_links_per_file.keys())
    num_links_per_file_nums = list(num_links_per_file.values())
    num_links = {
        'file_paths': num_links_per_file_paths,
        'num_links': num_links_per_file_nums,
    }
    df = pd.DataFrame(num_links)
    df.to_csv(os.path.join(output_path, 'number_of_links_per_file.csv'))



if __name__ == '__main__':
    # Create test dataset
    if input("Would you like to create a test dataset? (y/n) ").lower() == 'y':
        dataset_path = '/home/troyxdp/Documents/University Work/HYP/HYP Source Code/Back End/MillionSongSpotifyTracksDataset'
        csv_path = '/home/troyxdp/Documents/University Work/HYP/HYP Source Code/Back End/test_dataset/plays_data/train_triplets.csv'
        output_path = '/home/troyxdp/Documents/University Work/HYP/HYP Source Code/Back End/test_dataset/songs'
        create_test_dataset(
            dataset_path=dataset_path,
            csv_path=csv_path,
            output_path=output_path
        )

    # Move song files with corresponding test CSV files
    if input("Would you like to move the test dataset files to a different folder? (y/n) ").lower() == 'y':
        full_dataset_path = '/home/troyxdp/Documents/University Work/HYP/HYP Source Code/Back End/ProjectDataset'
        entp_path = '/home/troyxdp/Documents/University Work/HYP/HYP Source Code/Back End/test_dataset/song_play_data'
        output_path = '/home/troyxdp/Documents/University Work/HYP/HYP Source Code/Back End/test_dataset/test_songs'
        move_songs_to_test_folder(
            full_dataset_path=full_dataset_path, 
            entp_path=entp_path, 
            output_path=output_path
        )

    # Extract songs with links to other songs
    if input("Would you like to extract songs with links to other songs? (y/n) ").lower() == 'y':
        test_dataset_path = '/home/troyxdp/Documents/University Work/HYP/HYP Source Code/Back End/test_dataset/song_play_data'
        output_path = '/home/troyxdp/Documents/University Work/HYP/HYP Source Code/Back End/test_dataset/test_songs_with_links'
        extract_linked_songs(
            test_dataset_path=test_dataset_path,
            output_path=output_path
        )

    # Create dataset of song links
    if input("Would you like to extract song links from a dataset like the one created using the previous question? (y/n) ").lower() == 'y':
        test_dataset_path = '/home/troyxdp/Documents/University Work/HYP/HYP Source Code/Back End/test_dataset/test_songs_with_links'
        output_path = '/home/troyxdp/Documents/University Work/HYP/HYP Source Code/Back End/test_dataset/song_links_dataset'
        extract_song_links(
            test_dataset_path=test_dataset_path,
            output_path=output_path
        )