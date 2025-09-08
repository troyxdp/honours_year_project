import os
import random
import time
import itertools
import argparse

import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

from classes.trainer import Trainer
from classes.neural_network import NeuralNetwork
from classes.recommender import Recommender


# Calculate the ratio between the number of shared listeners and total number of listeners of two songs
def calculate_iou_in_song_plays(song_1_plays_file_path, song_2_plays_file_path):
    # Get dataframes from song files as well as the merge (intersection) between the two files
    df1 = pd.read_csv(song_1_plays_file_path)
    df2 = pd.read_csv(song_2_plays_file_path)
    intersection = df1.merge(df2, left_on='user_id', right_on='user_id')

    # Calculate the number of users in the intersection as well as in the union of the two files
    num_i = len(intersection) * 1.0
    num_u = len(df1) + len(df2) - num_i

    # Calculate iou and return
    iou = num_i / num_u
    return iou


# Generate subsets of given set
def get_subsets(set, length):
    set_list = list(set)
    return itertools.combinations(set_list, length)

# Made with assistance from https://en.wikipedia.org/wiki/Held%E2%80%93Karp_algorithm
# Made with assistance from https://stackoverflow.com/questions/69902373/can-you-help-explain-this-held-karp-tsp-pseudocode
def held_karp_maximimizer(dists):
    # This algorithm is similar to the standard Held-Karp Algorithm, except it tries to find the longest instead of the shortest path
    # Initialize values
    cities = [_ for _ in range(len(dists))]
    g = {} # keeps track of the G values for each "subpath" 
    parent = {} # keeps track of second-last city visited in each "subpath" 
    e = cities[0] # number of initial city

    # Initialize g with 1 step subpaths
    for k in range(1, len(dists)):
        g[((k,), k)] = dists[0][k]
        parent[((k,), k)] = 0

    # Add subpaths of increasing length s - this is the "dynamic programming loop"
    for s in range(2, len(dists)):
        # Get the subsets of cities excluding the starting city of length s
        subsets = get_subsets(cities[1:], s)
        for S in subsets:
            # Go through each end destination k
            for k in S:
                # Find the path of maximum cost (or rather set of vertices in the path) that ends at k with second last vertex m
                S_minus_k = [i for i in S if i != k]
                S_minus_k = tuple(S_minus_k)
                max_cost = -np.inf
                arg_max_m = -1
                for m in S:
                    if m == k:
                        continue
                    g_val = g[(S_minus_k, m)]
                    cost_val = dists[m][k]
                    cost = g_val + cost_val
                    if cost > max_cost:
                        max_cost = cost
                        arg_max_m = m
                g[(tuple(S), k)] = max_cost
                parent[(tuple(S), k)] = arg_max_m

    # Find optimal cost
    optimal_cost = -np.inf
    arg_max_k = -1
    for k in cities[1:]:
        cost = g[(tuple(cities[1:]), k)] + dists[k][e]
        if cost > optimal_cost:
            optimal_cost = cost
            arg_max_k = k

    # Find optimal path by backtracking through parents
    # Made with assistance from https://www.youtube.com/watch?v=-JjA4BLQyqE
    path = []
    subset = tuple(cities[1:])
    k = arg_max_k
    while len(subset) != 0:
        path.append(k)
        m = parent[(subset, k)]
        subset = tuple([city for city in subset if city != k])
        k = m
    path.append(0)
    path = list(reversed(path))

    return path, optimal_cost


# Generate a random path through songs that are linked of length k
def generate_random_traversal_of_linked_songs(song_links_files: list, k: int, song_links_test_dataset_path: str) -> list:
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
                curr_track_song_id = selected_tracks[-1]
                curr_track_file_path = os.path.join(song_links_test_dataset_path, curr_track_song_id[2], curr_track_song_id[3], curr_track_song_id[4], f"{curr_track_song_id}.txt")

        else:
            # There is at least one track that has not already been chosen
            # Choose a random track
            prev_track_song_id = curr_track_song_id
            prev_track_file_path = curr_track_file_path
            curr_track_song_id = random.choice(possible_paths_for_selected_tracks[-1])
            curr_track_file_path = os.path.join(song_links_test_dataset_path, curr_track_song_id[2], curr_track_song_id[3], curr_track_song_id[4], f"{curr_track_song_id}.txt")

            # Remove that random track from the list of possible tracks for the previous track
            possible_paths_for_selected_tracks[-1].pop(possible_paths_for_selected_tracks[-1].index(curr_track_song_id))

            # Try get possible tracks for song that was just selected (if the file exists)
            linked_tracks = []
            try:
                with open(curr_track_file_path, 'r') as f:
                    linked_tracks = f.readlines()
            except:
                # If track does not exist, reset values to previous value and continue. Track was already popped off but not added, so another possibility will be selected or backtracking will take place
                curr_track_song_id = prev_track_song_id
                curr_track_file_path = prev_track_file_path
                continue
            linked_tracks = [track.strip() for track in linked_tracks]
            possible_tracks = []
            for track in linked_tracks:
                if not track in selected_tracks:
                    possible_tracks.append(track)

            # Add possible tracks for chosen track
            selected_tracks.append(curr_track_song_id)
            possible_paths_for_selected_tracks.append(possible_tracks)

        num_selected = len(selected_tracks)

    # Return tracks that were selected
    return selected_tracks


# Test the NCG and generation times of a given neural network on a given set of song lists
def test_network_ncg(
        nn: NeuralNetwork, 
        test_dataset_path: str, 
        song_plays_dataset_path: str, 
        pregenerated_lists_dataset_path: str,
        traversal_algorithm: str,
        output_path='Statistics/Raw Data',
    ):
    # Create a trainer for fetching file paths - used later
    trainer = Trainer(
        training_network=NeuralNetwork(202, 202),
        initial_lr=0.001,
        final_lr=0.0001,
        num_epochs=120,
        dataset_path=test_dataset_path,
        output_folder=''
    )

    # Test for each k value
    ncg_scores = []
    recommendation_gen_times = []

    # Get all of the paths of the k files
    test_paths = os.listdir(pregenerated_lists_dataset_path)
    test_paths = sorted(test_paths, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    for k_txt_path in test_paths:
        # Get pregenerated path
        k = int(os.path.splitext(k_txt_path)[0])
        k_path = os.path.join(pregenerated_lists_dataset_path, k_txt_path)
        selected_tracks = []
        with open(k_path, 'r') as f:
            selected_tracks = f.readlines()
            selected_tracks = [song_id.strip() for song_id in selected_tracks]
        
        # Get the paths to each of the h5 files for the different song IDs
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
            try:
                song = trainer.get_song_data_from_file(song_path)
            except:
                print(song_path)
                exit(0)
            song_nn_input = song.get_nn_input()
            nn.set_input(song_nn_input)
            nn.feed_forward()
            song_embedding = nn.get_output()
            track_info = (os.path.splitext(os.path.basename(song_path))[0], song_embedding)
            selected_tracks_info.append(track_info)
        recommender.init_selected_track_ids(selected_tracks_info=selected_tracks_info)

        # Generate recommendations and record time taken
        start_time = time.time()
        recommender.init_recommendations()
        recommendations_gen_end_time = time.time() - start_time
        recommendation_gen_times.append(recommendations_gen_end_time)

        # Get "distances" between each song (which will be 1/iou because Held-Karp minimizes)
        ious = np.zeros((k, k))
        for i in range(len(selected_tracks)):
            ious[i][i] = 0
            for j in range(i + 1, len(selected_tracks)):
                # Get songs to calculate iou for
                song_a = selected_tracks[i]
                song_b = selected_tracks[j]

                # Get file paths for each song
                song_1_plays_file_path = os.path.join(song_plays_dataset_path, song_a[2], song_a[3], song_a[4], f'{song_a}.csv')
                song_2_plays_file_path = os.path.join(song_plays_dataset_path, song_b[2], song_b[3], song_b[4], f'{song_b}.csv')

                # Calculate IOU and score, which is 1/iou since we are trying to maximize iou and Held-Karp algorithm we are using minimizes
                iou = calculate_iou_in_song_plays(song_1_plays_file_path, song_2_plays_file_path)
                ious[i][j] = iou
                ious[j][i] = iou

        # Get path that maximizes total IOU (cumulative gain) using a maximization version of the Held Karp algorithm
        _, optimal_gain = held_karp_maximimizer(ious)

        # Calculate NCG
        cumulative_gain = 0
        curr_track_id = seed_track_id
        is_recommendations = True
        while is_recommendations:
            # Get next recommendation
            recommended_track_id = recommender.get_next_recommendation_track_id((curr_track_id, seed_song_embedding))
            # Check if the recommender has run out of recommendations
            if recommended_track_id is None:
                # Run out
                is_recommendations = False
            else:
                # Not run out
                # Update cumulative gain
                curr_track_plays_path = os.path.join(song_plays_dataset_path, curr_track_id[2], curr_track_id[3], curr_track_id[4], f'{curr_track_id}.csv')
                recommended_track_plays_path = os.path.join(song_plays_dataset_path, recommended_track_id[2], recommended_track_id[3], recommended_track_id[4], f'{recommended_track_id}.csv')
                iou = calculate_iou_in_song_plays(curr_track_plays_path, recommended_track_plays_path)
                cumulative_gain += iou

            # Update curr_track_id
            curr_track_id = recommended_track_id

        # Calculate NCG
        ncg = cumulative_gain / optimal_gain
        ncg_scores.append(ncg)
        print(f"NCG@{k} = {ncg}")

    # Do the tings for the NCG values (if they are there)
    ncg_k_values = [int(os.path.splitext(test_path)[0]) for test_path in test_paths]
    plt.plot(ncg_k_values, ncg_scores)
    plt.xlabel("k")
    plt.ylabel("NCG@k")
    plt.title("NCG@k for Different k Values")
    plt.show()

    # Save NCG scores to CSV
    ncg_dict = {
        "k_values": ncg_k_values,
        "ncg": ncg_scores
    }
    df = pd.DataFrame(ncg_dict)
    df.to_csv(os.path.join(output_path, 'ncg_scores.csv'))


# Test the precision and generation times of a given a neural network on a given set of song lists
def test_network_precision(
        nn: NeuralNetwork, 
        test_dataset_path: str, 
        song_links_test_dataset_path: str, 
        pregenerated_lists_dataset_path: str,
        traversal_algorithm: str,
        output_path='Statistics/Raw Data',
    ):
    # Create a trainer for fetching file paths - used later
    trainer = Trainer(
        training_network=NeuralNetwork(202, 202),
        initial_lr=0.001,
        final_lr=0.0001,
        num_epochs=120,
        dataset_path=test_dataset_path,
        output_folder=''
    )

    # Test for each k value
    precision_scores = []
    recommendation_gen_times = []

    # Get all of the paths of the k files
    test_paths = os.listdir(pregenerated_lists_dataset_path)
    test_paths = sorted(test_paths, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    for k_txt_path in test_paths:
        # Get pregenerated path
        k = int(os.path.splitext(k_txt_path)[0])
        k_path = os.path.join(pregenerated_lists_dataset_path, k_txt_path)
        selected_tracks = []
        with open(k_path, 'r') as f:
            selected_tracks = f.readlines()
            selected_tracks = [song_id.strip() for song_id in selected_tracks]
        
        # Get the paths to each of the h5 files for the different song IDs
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
            try:
                song = trainer.get_song_data_from_file(song_path)
            except:
                print(song_path)
                exit(0)
            song_nn_input = song.get_nn_input()
            nn.set_input(song_nn_input)
            nn.feed_forward()
            song_embedding = nn.get_output()
            track_info = (os.path.splitext(os.path.basename(song_path))[0], song_embedding)
            selected_tracks_info.append(track_info)
        recommender.init_selected_track_ids(selected_tracks_info=selected_tracks_info)

        # Generate recommendations and record time taken
        start_time = time.time()
        recommender.init_recommendations()
        recommendations_gen_end_time = time.time() - start_time
        recommendation_gen_times.append(recommendations_gen_end_time)

        # Calculate precision and "NCG" - recall is not appropriate for task
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
            if recommended_track_id is None:
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

        # Calculate precision
        precision = true_positives / (true_positives + false_positives)
        precision_scores.append(precision)
        print(f"Precision@{k} = {precision}")

    # Display the statistics for precision
    k_values = [int(os.path.splitext(test_path)[0]) for test_path in test_paths]
    plt.plot(k_values, precision_scores)
    plt.xlabel("k")
    plt.ylabel("Precision@k")
    plt.title("Precision@k for Different k Values")
    plt.show()

    # Save precision values to CSV
    precision_dict = {
        "k_values": k_values,
        "precision": precision_scores
    }
    df = pd.DataFrame(precision_dict)
    df.to_csv(os.path.join(output_path, 'precision_scores.csv'))

    # Display generation times for recommendations
    plt.plot(k_values, recommendation_gen_times)
    plt.xlabel("k")
    plt.ylabel("Time to Generate (seconds)")
    plt.title("Time Taken to Generate Recommendation Path for Different Path Lengths k")
    plt.show()

    # Save generation times to CSV
    gen_times_dict = {
        "k_values": k_values,
        "generation_time": recommendation_gen_times
    }
    df = pd.DataFrame(gen_times_dict)
    df.to_csv(os.path.join(output_path, 'generation_times.csv'))
        

# Test a given neural network - get precision, generation time, and NCG scores for different k values
def test_network_on_random_lists(
        nn: NeuralNetwork, 
        test_dataset_path: str, 
        song_plays_dataset_path: str, 
        song_links_test_dataset_path: str, 
        traversal_algorithm: str,
        debug=False,
        low_k=10,
        high_k=20,
        k_step=2,
        ncg_max_k=20,
        output_path='Statistics/Raw Data',
    ):
    # Get test dataset files with song data
    print("Loading test dataset...")
    trainer = Trainer(
        training_network=NeuralNetwork(202, 202),
        initial_lr=0.001,
        final_lr=0.0001,
        num_epochs=120,
        dataset_path=test_dataset_path,
        output_folder=''
    )
    initial_test_dataset_files = trainer.get_file_paths()
    initial_test_dataset_files = sorted(initial_test_dataset_files)

    print("Getting song links...")
    initial_song_ids = [os.path.splitext(os.path.basename(track))[0] for track in initial_test_dataset_files]

    # Check that the song files exist in each of the datasets
    song_ids = []
    for song_id in initial_song_ids:
        song_links_path = os.path.join(song_links_test_dataset_path, song_id[2], song_id[3], song_id[4], f'{song_id}.txt')
        test_songs_path = os.path.join(test_dataset_path, song_id[2], song_id[3], song_id[4], f'{song_id}.h5')
        song_plays_path = os.path.join(song_plays_dataset_path, song_id[2], song_id[3], song_id[4], f'{song_id}.csv')
        if os.path.exists(song_links_path) and os.path.exists(test_songs_path) and os.path.exists(song_plays_path):
            song_ids.append(song_id)
        else:
            print("Skipping song...")
    print("Number of songs:", len(song_ids))

    song_links_files = [os.path.join(song_links_test_dataset_path, file_name[2], file_name[3], file_name[4], f'{file_name}.txt') for file_name in song_ids]

    # Test for each k value
    precision_scores = []
    ncg_scores = []
    recommendation_gen_times = []
    k_values = range(low_k, high_k+1, k_step)
    for k in k_values:
        print("\nTesting for k =", k, "...")

        # Generate a set of linked tracks
        selected_tracks = generate_random_traversal_of_linked_songs(song_links_files, k, song_links_test_dataset_path)
        print(f"{selected_tracks[:8]} for k = {k}...")

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
            try:
                song = trainer.get_song_data_from_file(song_path)
            except:
                print(song_path)
                exit(0)
            song_nn_input = song.get_nn_input()
            nn.set_input(song_nn_input)
            nn.feed_forward()
            song_embedding = nn.get_output()
            track_info = (os.path.splitext(os.path.basename(song_path))[0], song_embedding)
            selected_tracks_info.append(track_info)
        recommender.init_selected_track_ids(selected_tracks_info=selected_tracks_info)

        # Generate recommendations and record time taken
        start_time = time.time()
        recommender.init_recommendations()
        recommendations_gen_end_time = time.time() - start_time
        recommendation_gen_times.append(recommendations_gen_end_time)

        # Calculate optimal path for "NCG" (not NDCG - discounting is not appropriate for task) - not recommended going above 20 else Held-Karp will take FOREVER since it is O(n^2 * 2^n)
        optimal_gain = 0
        if k <= ncg_max_k:
            # Get "distances" between each song (which will be 1/iou because Held-Karp minimizes)
            ious = np.zeros((k, k))
            for i in range(len(selected_tracks)):
                ious[i][i] = 0
                for j in range(i + 1, len(selected_tracks)):
                    # Get songs to calculate iou for
                    song_a = selected_tracks[i]
                    song_b = selected_tracks[j]

                    # Get file paths for each song
                    song_1_plays_file_path = os.path.join(song_plays_dataset_path, song_a[2], song_a[3], song_a[4], f'{song_a}.csv')
                    song_2_plays_file_path = os.path.join(song_plays_dataset_path, song_b[2], song_b[3], song_b[4], f'{song_b}.csv')

                    # Calculate IOU and score, which is 1/iou since we are trying to maximize iou and Held-Karp algorithm we are using minimizes
                    iou = calculate_iou_in_song_plays(song_1_plays_file_path, song_2_plays_file_path)
                    ious[i][j] = iou
                    ious[j][i] = iou

            if debug:
                is_symmetrical = True
                for i in range(len(ious)):
                    for j in range(len(ious[i])):
                        if ious[i][j] != ious[j][i]:
                            is_symmetrical = False
                            break
                    if not is_symmetrical:
                        break
                print("Is dists symmetrical: ", is_symmetrical)

            # Get path that maximizes total IOU (cumulative gain) using a maximization version of the Held Karp algorithm
            optimal_path, optimal_gain = held_karp_maximimizer(ious)
            if debug:
                print("Optimal path:", optimal_path)
                print("Optimal gain:", optimal_gain)

        # Calculate precision and "NCG" - recall is not appropriate for task
        true_positives = 0
        false_positives = 0
        cumulative_gain = 0
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
            if recommended_track_id is None:
                # Run out
                is_recommendations = False
            else:
                # Not run out
                # See if recommendation is in list of positives
                if recommended_track_id in positives:
                    true_positives += 1
                else:
                    false_positives += 1

                # Update cumulative gain
                if k <= ncg_max_k:
                    curr_track_plays_path = os.path.join(song_plays_dataset_path, curr_track_id[2], curr_track_id[3], curr_track_id[4], f'{curr_track_id}.csv')
                    recommended_track_plays_path = os.path.join(song_plays_dataset_path, recommended_track_id[2], recommended_track_id[3], recommended_track_id[4], f'{recommended_track_id}.csv')
                    iou = calculate_iou_in_song_plays(curr_track_plays_path, recommended_track_plays_path)
                    cumulative_gain += iou

            # Update curr_track_id
            curr_track_id = recommended_track_id

        # Calculate precision
        precision = true_positives / (true_positives + false_positives)
        precision_scores.append(precision)
        print(f"Precision@{k} = {precision}")

        # Calculate NCG (if k <= ncg_max_k)
        if k <= ncg_max_k:
            ncg = cumulative_gain / optimal_gain
            ncg_scores.append(ncg)
            print(f"NCG@{k} = {ncg}")

    # Display the statistics for precision
    k_values = list(range(low_k, high_k+1, k_step))
    plt.plot(k_values, precision_scores)
    plt.xlabel("k")
    plt.ylabel("Precision@k")
    plt.title("Precision@k for Different k Values")
    plt.show()

    # Save precision values to CSV
    precision_dict = {
        "k_values": k_values,
        "precision": precision_scores
    }
    df = pd.DataFrame(precision_dict)
    df.to_csv(os.path.join(output_path, 'precision_scores.csv'))

    # Display generation times for recommendations
    plt.plot(k_values, recommendation_gen_times)
    plt.xlabel("k")
    plt.ylabel("Time to Generate (seconds)")
    plt.title("Time Taken to Generate Recommendation Path for Different Path Lengths k")
    plt.show()

    # Save generation times to CSV
    gen_times_dict = {
        "k_values": k_values,
        "generation_time": recommendation_gen_times
    }
    df = pd.DataFrame(gen_times_dict)
    df.to_csv(os.path.join(output_path, 'generation_times.csv'))

    # Do the tings for the NCG values (if they are there)
    ncg_k_values = list(range(low_k, ncg_max_k+1, k_step))
    if ncg_k_values:
        # Display the statistics for NCG
        plt.plot(ncg_k_values, ncg_scores)
        plt.xlabel("k")
        plt.ylabel("NCG@k")
        plt.title("NCG@k for Different k Values")
        plt.show()

        # Save NCG scores to CSV
        ncg_dict = {
            "k_values": ncg_k_values,
            "ncg": ncg_scores
        }
        df = pd.DataFrame(ncg_dict)
        df.to_csv(os.path.join(output_path, 'ncg_scores.csv'))

    # Write the testing hyperparameters to a text file
    with open(os.path.join(output_path, 'testing_hyperparameters.txt'), 'w') as f:
        # K information
        f.write(f"Low K: {low_k}\n")
        f.write(f"High K: {high_k}\n")
        f.write(f"K Step Size: {k_step}\n\n")
        # NCG information
        f.write(f"Did NCG? {ncg_max_k >= low_k}\n")
        f.write(f"NCG Max K: {ncg_max_k}\n\n")
        # Traversal method
        f.write(f"Traversal method: {traversal_algorithm}")


# Generate lists
def generate_lists(
        test_dataset_path: str, 
        song_plays_dataset_path: str, 
        song_links_test_dataset_path: str, 
        output_path: str,
        low_k=10,
        high_k=20,
        k_step=2
    ):
    # Get test dataset files with song data
    print("Loading test dataset...")
    trainer = Trainer(
        training_network=NeuralNetwork(202, 202),
        initial_lr=0.001,
        final_lr=0.0001,
        num_epochs=120,
        dataset_path=test_dataset_path,
        output_folder=''
    )
    initial_test_dataset_files = trainer.get_file_paths(input_dim=202)
    initial_test_dataset_files = sorted(initial_test_dataset_files)

    print("Getting song links...")
    initial_song_ids = [os.path.splitext(os.path.basename(track))[0] for track in initial_test_dataset_files]

    # Check that the song files exist in each of the datasets
    song_ids = []
    for song_id in initial_song_ids:
        song_links_path = os.path.join(song_links_test_dataset_path, song_id[2], song_id[3], song_id[4], f'{song_id}.txt')
        test_songs_path = os.path.join(test_dataset_path, song_id[2], song_id[3], song_id[4], f'{song_id}.h5')
        song_plays_path = os.path.join(song_plays_dataset_path, song_id[2], song_id[3], song_id[4], f'{song_id}.csv')
        if os.path.exists(song_links_path) and os.path.exists(test_songs_path) and os.path.exists(song_plays_path):
            song_ids.append(song_id)
        else:
            print("Skipping song...")
    print("Number of songs:", len(song_ids))
    song_links_files = [os.path.join(song_links_test_dataset_path, file_name[2], file_name[3], file_name[4], f'{file_name}.txt') for file_name in song_ids]

    k_values = range(low_k, high_k + 1, k_step)
    for k in k_values:
        # Generate a set of linked tracks
        selected_tracks = generate_random_traversal_of_linked_songs(song_links_files, k, song_links_test_dataset_path)
        print(f"{selected_tracks[:8]} for k = {k}...")

        # Write the song IDs
        with open(os.path.join(output_path, f'{k}.txt'), 'w') as f:
            to_write = '\n'.join(selected_tracks)
            f.write(to_write)


# Main code
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Model Tester',
        description='Test a model created using the neural_network class',
    )
    parser.add_argument('--pregenerated', action='store_true', help='Specification of whether to run tests on pregenerated song lists')
    parser.add_argument('--test-precision', action='store_true', help='Specification of whether to test for precision or not')
    parser.add_argument('--test-dataset-path', type=str, help='Path to the test dataset of h5 songs')
    parser.add_argument('--song-plays-dataset-path', type=str, help='Path to the dataset showing which users played a song with a given song ID')
    parser.add_argument('--song-links-dataset-path', type=str, help='Path to the dataset showing which songs have common listeners for a given song ID')
    parser.add_argument('--pregenerated-lists-dataset-path', type=str, help='Path to the pregenerated lists of songs of different k lengths to test on')
    parser.add_argument('--output-path', type=str, help='The directory to output the CSV files containing the test results for each k value to')
    parser.add_argument('--nn-path', type=str, help='Path to neural network to test')
    parser.add_argument('--low-k', type=int, help='Lowest k value to test on')
    parser.add_argument('--high-k', type=int, help='Highest k value to test on')
    parser.add_argument('--k-step', type=int, help='The number of integers after a k value to next test on')
    parser.add_argument('--ncg-max-k', type=int, help='The maximum k value to calculate NCG for. Not recommended going above 20')
    parser.add_argument('--traversal-algorithm', type=str, choices=['greedy_nearest_neighbour', 'optimal_path'], help='The type of traversal you would like to perform: Greedy Nearest Neighbour (greedy_nearest_neighbour) or Held-Karp (optimal_path)')

    args = parser.parse_args()

    # Test on lists that are randomly generated during testing
    if not args.pregenerated:
        # Load encoder neural network for testing
        nn_path = args.nn_path
        if not os.path.exists(nn_path):
            print("Error: could not find neural network")
            exit(0)
        nn = NeuralNetwork.load_network(nn_path)

        # Paths to data and hyperparameters to use for testing
        low_k = args.low_k
        high_k = args.high_k # above 801 and it takes way to long to generate a list. Up to this value is relatively quick
        k_stride = args.k_step
        ncg_max_k = args.ncg_max_k
        output_path = args.output_path

        # Check if output_path exists
        if not os.path.isdir(output_path):
            print("Error: could not find output path")
            exit(0)
        
        # Test the network
        test_network_on_random_lists(
            nn=nn,
            test_dataset_path=args.test_dataset_path,
            song_plays_dataset_path=args.song_plays_dataset_path,
            song_links_test_dataset_path=args.song_links_test_dataset_path,
            traversal_algorithm='greedy_nearest_neighbour',
            high_k=high_k,
            low_k=low_k,
            k_step=k_stride,
            ncg_max_k=ncg_max_k,
            output_path=output_path
        )

    # Test on pregenerated lists
    else:
        pregenerated_lists_dataset_path = args.pregenerated_lists_dataset_path
        if not os.path.isdir(pregenerated_lists_dataset_path):
            print("Error: no such directory exists as the one you enterred")
            exit(0)

        # Load encoder neural network for testing
        nn_path = args.nn_path
        if not os.path.exists(nn_path):
            print("Error: could not find neural network")
            exit(0)
        nn = NeuralNetwork.load_network(nn_path)

        output_path = args.output_path
        # Check if output_path exists
        if not os.path.isdir(output_path):
            print("Error: could not find output path")
            exit(0)

        if args.test_precision:
            test_network_precision(
                nn=nn,
                test_dataset_path=args.test_dataset_path,
                song_links_test_dataset_path=args.song_links_dataset_path,
                pregenerated_lists_dataset_path=pregenerated_lists_dataset_path,
                traversal_algorithm='greedy_nearest_neighbour',
                output_path=output_path,
            )
        else:
            traversal_algorithm = input("Would you like to test using Greedy Nearest Neighbour or Held Karp traversal? (greedy_nearest_neighbour/optimal_path) ")
            test_network_ncg(
                nn=nn,
                test_dataset_path=args.test_dataset_path,
                song_plays_dataset_path=args.song_plays_dataset_path,
                pregenerated_lists_dataset_path=pregenerated_lists_dataset_path,
                traversal_algorithm=args.traversal_algorithm,
                output_path=output_path
            )

    # # Generate song lists to test on
    # if input("\nWould you like to generate lists of songs for tests? (y/n) ").lower() == 'y':
    #     high_k = int(input("Input High K: "))
    #     low_k = int(input("Input Low K: "))
    #     k_stride = int(input("Input K Stride: "))

    #     output_path = input("Please input the root directory for all of the lists: ")

    #     generate_lists(
    #         test_dataset_path=args.test_dataset_path,
    #         song_plays_dataset_path=args.song_plays_dataset_path,
    #         song_links_test_dataset_path=song_links_test_dataset_path,
    #         output_path=output_path,
    #         low_k=low_k,
    #         high_k=high_k,
    #         k_step=k_stride,
    #     )
