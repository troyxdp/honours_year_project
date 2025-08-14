import os
import csv
import json

import pandas as pd

def change_delimiter_in_csv(input_path, output_path, delimiter):
    count = 0
    with open(output_path, 'w') as f:
        writer = csv.writer(f, delimiter=delimiter, quotechar='|')
        writer.writerow(['user_id', 'track_id', 'play_count'])
        for chunk in pd.read_csv(input_path, delimiter='\t', chunksize=1000):
            for row in chunk.itertuples(index=False, name=None):
                user_id = row[0]
                song_id = row[1]
                play_count = row[2]
                writer.writerow([user_id, song_id, play_count])
                count += 1
                if count % 1000 == 0:
                    print(count)

def extract_from_track_features_csv(input_path, output_path):
    # Extract relevant data from first CSV
    counter = 0
    with open(output_path, 'a') as f:
        writer = csv.writer(f, delimiter=',', quotechar='|')
        for chunk_1 in pd.read_csv(input_path, chunksize=1000):
            for row in chunk_1.itertuples(index=True):
                # Search for track in other CSV file to remove duplicates
                track_id = str(row.track_id)
                
                # Extract relevant data and write to CSV
                track_name = str(row.track_name)
                artists_list = eval(row.artists)
                artists_list = [artist.strip() for artist in artists_list]
                artists = ';'.join(artists_list)
                # NB: need to get year from MSD

                # High-level descriptors
                # track_genre_1 = row.track_genre # excluding for now - might use supp dataset for this
                key = row.key
                mode = row.mode
                tempo = row.tempo
                time_signature = row.time_signature

                # Low-level descriptors/calculated features
                danceability = row.danceability
                energy = row.energy
                instrumentalness = row.instrumentalness
                loudness = row.loudness
                valence = row.valence

                # Write values to file
                writer.writerow([
                    track_id, track_name, artists, key, mode, 
                    tempo, time_signature, danceability, energy, loudness, 
                    valence, instrumentalness
                ])
                
                if counter % 100 == 0:
                    print(f"On row {counter + 1}")
                counter += 1

def extract_from_spotify_tracks_csv(input_path, output_path):
    # Extract relevant data from first CSV
    counter = 0
    with open(output_path, 'a') as f:
        writer = csv.writer(f, delimiter=',', quotechar='|')
        for chunk_1 in pd.read_csv(input_path, chunksize=1000):
            for row in chunk_1.itertuples(index=True):
                # Search for track in other CSV file to remove duplicates
                track_id = str(row.track_id)
                
                # Extract relevant data and write to CSV
                track_name = str(row.track_name)
                artists = str(row.artists)
                
                # NB: need to get year from MSD

                # High-level descriptors
                # track_genre_1 = row.track_genre # excluding for now - might use supp dataset for this
                key = row.key
                mode = row.mode
                tempo = row.tempo
                time_signature = row.time_signature

                # Low-level descriptors/calculated features
                danceability = row.danceability
                energy = row.energy
                instrumentalness = row.instrumentalness
                loudness = row.loudness
                valence = row.valence

                # Write values to file
                writer.writerow([
                    track_id, track_name, artists, key, mode, 
                    tempo, time_signature, danceability, energy, loudness, 
                    valence, instrumentalness
                ])
                
                if counter % 100 == 0:
                    print(f"On row {counter + 1}")
                counter += 1

def clean_data(input_path, output_path, required_comma_count):
    with open(input_path, 'r') as input_file:
        with open(output_path, 'w') as output_file:
            output_file.write('track_id,track_name,artists,key,mode,tempo,time_signature,danceability,energy,loudness,valence,instrumentalness\n')
            for line in input_file:
                comma_count = line.count(',')
                if comma_count == required_comma_count:
                    output_file.write(line)
                else:
                    print("Found invalid line")
        
if __name__ == '__main__':
    do_track_features_csv = input("Would you like to extract fields from Track Features dataset? (y/n) ")
    if do_track_features_csv.lower() == 'y':
        # Create file
        with open('/home/troyxdp/Documents/University Work/HYP/HYP Source Code/Back End/MergedDataset/extracted_data.csv', 'w') as f:
            writer = csv.writer(f, delimiter=',', quotechar='|')
            writer.writerow([
                'track_id', 'track_name', 'artists', 'key', 'mode', 
                'tempo', 'time_signature', 'danceability', 'energy', 'loudness', 
                'valence', 'instrumentalness'
            ]) 
        # Extract data
        extract_from_track_features_csv(
            '/home/troyxdp/Documents/University Work/HYP/HYP Source Code/Back End/SpotifyTracksDataset/tracks_features.csv',
            '/home/troyxdp/Documents/University Work/HYP/HYP Source Code/Back End/MergedDataset/extracted_data.csv'
        )
        # Remove rows with extra commas
        clean_data(
            '/home/troyxdp/Documents/University Work/HYP/HYP Source Code/Back End/MergedDataset/extracted_data.csv',
            '/home/troyxdp/Documents/University Work/HYP/HYP Source Code/Back End/TrackFeaturesDataset/track_features_cleaned_data.csv',
            11
        )

    do_spotify_tracks_csv = input("Would you like to extract fields from Spotify Tracks dataset? (y/n) ")
    if do_spotify_tracks_csv.lower() == 'y':
        # Create file
        with open('/home/troyxdp/Documents/University Work/HYP/HYP Source Code/Back End/MergedDataset/extracted_spotify_tracks_data.csv', 'w') as f:
            writer = csv.writer(f, delimiter=',', quotechar='|')
            writer.writerow([
                'track_id', 'track_name', 'artists', 'key', 'mode', 
                'tempo', 'time_signature', 'danceability', 'energy', 'loudness', 
                'valence', 'instrumentalness'
            ]) 
        # Extract data
        extract_from_spotify_tracks_csv(
            '/home/troyxdp/Documents/University Work/HYP/HYP Source Code/Back End/SpotifyTracksDataset/dataset.csv',
            '/home/troyxdp/Documents/University Work/HYP/HYP Source Code/Back End/SpotifyTracksDataset/extracted_data.csv'
        )
        # Remove rows with extra commas
        clean_data(
            '/home/troyxdp/Documents/University Work/HYP/HYP Source Code/Back End/SpotifyTracksDataset/extracted_data.csv',
            '/home/troyxdp/Documents/University Work/HYP/HYP Source Code/Back End/SpotifyTracksDataset/spotify_tracks_cleaned_data.csv',
            11
        )

    if input("Would you like to change the delimiter of a CSV file? ").lower() == 'y':
        change_delimiter_in_csv(
            '/home/troyxdp/Documents/University Work/HYP/HYP Source Code/Back End/test_dataset/plays_data/train_triplets.csv',
            '/home/troyxdp/Documents/University Work/HYP/HYP Source Code/Back End/test_dataset/plays_data/train_triplets_edit.csv',
            ','
        )