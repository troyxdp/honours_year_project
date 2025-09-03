import os
import random
import time

import pandas as pd
import matplotlib.pyplot as plt

from classes.trainer import Trainer
from classes.neural_network import NeuralNetwork
from classes.recommender import Recommender

def test_network(
        nn: NeuralNetwork, 
        test_dataset_path: str, 
        song_plays_dataset_path: str, 
        song_links_test_dataset_path: str, 
        k_values: int,
        traversal_algorithm: str,
        debug=False,
    ):
    # Get test dataset files with song data
    trainer = Trainer(
        training_network=NeuralNetwork(202, 202),
        initial_lr=0.001,
        final_lr=0.0001,
        num_epochs=120,
        dataset_path=test_dataset_path,
        output_folder='/home/troyxdp/Documents/University Work/HYP/HYP Source Code/Back End/networks/experiment_2'
    )
    test_dataset_files = trainer.get_file_paths()
    test_dataset_files = sorted(test_dataset_files)

    file_names = [os.path.splitext(os.path.basename(track))[0] for track in test_dataset_files]
    song_plays_files = [os.path.join(song_plays_dataset_path, file_name[2], file_name[3], file_name[4], f'{file_name}.csv') for file_name in file_names]
    song_links_files = [os.path.join(song_links_test_dataset_path, file_name[2], file_name[3], file_name[4], f'{file_name}.txt') for file_name in file_names]

    # Test for each k value
    precision_scores = []
    for k in k_values:
        # List of k selected tracks and number of selected tracks
        selected_tracks = [] # List of song IDs selected
        num_selected = 1

        # Select a random track, get its index, and get its file path
        curr_track_file_path = random.choice(song_links_files)
        curr_track_song_id = os.path.splitext(os.path.basename(curr_track_file_path))[0]

        # Get list of possible first tracks for the first track if algorithm backtracks all the way to first track
        all_possible_first_tracks = [os.path.splitext(os.path.basename(track))[0] for track in song_links_files]
        all_possible_first_tracks.pop(all_possible_first_tracks.index(curr_track_song_id)) # Remove chosen first track - can not be reused for backtracking
        
        # Add file path to selected tracks
        selected_tracks.append(curr_track_song_id)

        # Get possible tracks for first selected track
        linked_tracks = []
        with open(curr_track_file_path, 'r') as f:
            linked_tracks = f.readlines()
        linked_tracks = [track.strip() for track in linked_tracks]
        possible_tracks = []
        for track in linked_tracks:
            if not track in selected_tracks:
                possible_tracks.append(track)

        # Get possible paths for each selected track
        possible_paths_for_selected_tracks = [] # Each index corresponds with the same index on selected_tracks
        possible_paths_for_selected_tracks.append(possible_tracks) # Add list of possible tracks that was just calculated

        # Generate a random path
        while num_selected < k:
            # Check if any of the tracks are not already chosen
            if len(possible_paths_for_selected_tracks[-1]) == 0:
                # There are no tracks that have not already been chosen
                selected_tracks.pop(-1)
                possible_paths_for_selected_tracks.pop(-1)

                # If the track removed was the first track, choose a new first track if there are any available, else throw an error
                if len(selected_tracks) == 0:
                    # Backtracked all the way back to the first track
                    if len(all_possible_first_tracks) > 0:
                        # There is still first tracks available - choose a track from all possible first tracks
                        if debug:
                            print("Replacing first track...")
                        curr_track_song_id = random.choice(all_possible_first_tracks)
                        curr_track_file_path = os.path.join(song_links_test_dataset_path, curr_track_song_id[2], curr_track_song_id[3], curr_track_song_id[4], f"{curr_track_song_id}.txt")

                        # Regenerate possible tracks
                        linked_tracks = []
                        with open(curr_track_file_path, 'r') as f:
                            linked_tracks = f.readlines()
                        linked_tracks = [track.strip() for track in linked_tracks]
                        possible_tracks = []
                        for track in linked_tracks:
                            if not track in selected_tracks:
                                possible_tracks.append(track)

                        # Update arrays
                        selected_tracks.append(curr_track_song_id)
                        possible_paths_for_selected_tracks.append(possible_tracks)
                    else:
                        # No possible choices for first track - throw an exception
                        raise Exception(f"Error: could not create path of length {k}")
                else:
                    # Didn't backtrack all the way back to first track
                    # Set curr_track to be previous track and update number selected counter
                    if debug:
                        print("Backtracking...")
                    curr_track_song_id = selected_tracks[-1]
                    curr_track_file_path = os.path.join(song_links_test_dataset_path, curr_track_song_id[2], curr_track_song_id[3], curr_track_song_id[4], f"{curr_track_song_id}.txt")

            else:
                # There is at least one track that has not already been chosen
                # Choose a random track
                if debug:
                    print("Adding new track...")
                curr_track_song_id = random.choice(possible_paths_for_selected_tracks[-1])
                curr_track_file_path = os.path.join(song_links_test_dataset_path, curr_track_song_id[2], curr_track_song_id[3], curr_track_song_id[4], f"{curr_track_song_id}.txt")

                # Remove that random track from the list of possible tracks for the previous track
                possible_paths_for_selected_tracks[-1].pop(possible_paths_for_selected_tracks[-1].index(curr_track_song_id))

                # Get possible tracks for song that was just selected
                linked_tracks = []
                with open(curr_track_file_path, 'r') as f:
                    linked_tracks = f.readlines()
                linked_tracks = [track.strip() for track in linked_tracks]
                possible_tracks = []
                for track in linked_tracks:
                    if not track in selected_tracks:
                        possible_tracks.append(track)

                # Add possible tracks for chosen track
                selected_tracks.append(curr_track_song_id)
                possible_paths_for_selected_tracks.append(possible_tracks)

            num_selected = len(selected_tracks)

        print()
        print(f"{selected_tracks[:8]} for k = {k}...")
        if debug:
            input("Press ENTER")

        # Get paths to song data for generated list of k songs
        k_test_song_paths = [os.path.join(test_dataset_path, track[2], track[3], track[4], f"{track}.h5") for track in selected_tracks]

        # Instantiate recommender object
        recommender = Recommender(traversal_algorithm=traversal_algorithm)

        # Get first (seed track) song and its embedding
        seed_track_id = selected_tracks[0]
        seed_song = trainer.get_song_data_from_file(k_test_song_paths[0])
        seed_song_nn_input = seed_song.get_nn_input()
        nn.set_input(seed_song_nn_input)
        nn.feed_forward()
        seed_song_embedding = nn.get_output()
        seed_track_info = (seed_track_id, seed_song_embedding)
        recommender.set_seed_track_info(seed_track_info)

        # Set recommender values
        selected_tracks_info = []
        for song_path in k_test_song_paths[1:]:
            song = trainer.get_song_data_from_file(song_path)
            song_nn_input = song.get_nn_input()
            nn.set_input(song_nn_input)
            nn.feed_forward()
            song_embedding = nn.get_output()
            track_info = (os.path.splitext(os.path.basename(song_path))[0], song_embedding)
            selected_tracks_info.append(track_info)
        recommender.init_selected_track_ids(selected_tracks_info=selected_tracks_info)

        # Generate recommendations
        start_time = time.time()
        recommender.init_recommendations()
        recommendations_gen_end_time = time.time() - start_time

        # Calculate precision
        true_positives = 0
        false_positives = 0
        curr_track_id = seed_track_id
        is_recommendations = True
        while is_recommendations:
            # Get positives
            curr_track_song_link_path = os.path.join(song_links_test_dataset_path, curr_track_id[2], curr_track_id[3], curr_track_id[4], f'{curr_track_id}.txt')
            positives = []
            with open(curr_track_song_link_path, 'r') as f:
                positives = f.readlines()
            positives = [pos.strip() for pos in positives]

            # Get next recommendation
            recommended_track_id = recommender.get_next_recommendation_track_id((curr_track_id, seed_song_embedding))
            # Check if the recommender has run out of recommendations
            if recommended_track_id == curr_track_id:
                # Run out
                is_recommendations = False
            else:
                # Not run out
                # See if recommendation is in list of positives
                if recommended_track_id in positives:
                    true_positives += 1
                else:
                    false_positives += 1

            # Update curr_track_id
            curr_track_id = recommended_track_id

        # Calculate final value
        precision = true_positives / (true_positives + false_positives)
        print(f"Precision@{k} = {precision}")

        


if __name__ == '__main__':
    # Load encoder neural network for testing
    nn_path = input("Please input the path to the neural network you would like to test: ")
    if not os.path.exists(nn_path):
        print("Error: could not find neural network")
        exit(0)
    nn = NeuralNetwork.load_network(nn_path)

    # Paths to data and hyperparameters to use for testing
    test_dataset_path = 'test_dataset/test_songs'
    song_plays_dataset_path = 'test_dataset/test_songs_with_links'
    song_links_test_dataset_path = 'test_dataset/song_links_dataset'
    low_k = 10
    high_k = 201 # above 801 and it takes way to long to generate a list. Up to this value is relatively quick
    k_stride = 10
    k_values = range(low_k, high_k, k_stride)
    
    # Test the network
    test_network(
        nn=nn,
        test_dataset_path=test_dataset_path,
        song_plays_dataset_path=song_plays_dataset_path,
        song_links_test_dataset_path=song_links_test_dataset_path,
        k_values=k_values,
        traversal_algorithm='greedy_nearest_neighbour',
        debug=False
    )