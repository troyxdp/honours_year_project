import os
import csv
import sys

import tables
import pandas as pd

from classes.song import Song
from classes.hdf5_utils.dataset_creator import create_track_file

# Adapted from Bertin-Mahieux, T. (2010) https://github.com/tbertinmahieux/MSongsDB/blob/master/PythonSrc/hdf5_getters.py
# specifically the parts for getting each value from the h5 file
def merge_msd_with_csv(msd_path, spotify_tracks_file_path, output_path):
    # Check MSD path provided is valid
    if not os.path.isdir(msd_path):
        raise FileNotFoundError(f"Error: could not find directory {msd_path}")
    
    # Check Spotify Tracks path provided is valid
    if not os.path.exists(spotify_tracks_file_path):
        raise FileNotFoundError(f"Error: could not find directory {msd_path}")

    # FOR DEBUGGING
    counter = 1

    # Get data
    if len(os.listdir(msd_path)) == 0:
        raise FileNotFoundError(f"Error: could not find dataset --- {msd_path} is empty")
    # Iterate through directory provided
    for path_dir_1 in sorted(os.listdir(msd_path)):
        dir_path_1 = os.path.join(msd_path, path_dir_1)
        if len(os.listdir(dir_path_1)) == 0:
            raise FileNotFoundError(f"Error: could not find dataset --- {dir_path_1} is empty")
        # Iterate through subdirectories
        for path_dir_2 in sorted(os.listdir(dir_path_1)):
            dir_path_2 = os.path.join(dir_path_1, path_dir_2)
            if len(os.listdir(dir_path_2)) == 0:
                raise FileNotFoundError(f"Error: could not find dataset --- {dir_path_2} is empty")
            # Iterate through subsubdirectories
            for path_dir_3 in sorted(os.listdir(dir_path_2)):
                dir_path_3 = os.path.join(dir_path_2, path_dir_3)
                # Get files
                for file_name in sorted(os.listdir(dir_path_3)):
                    file_path = os.path.join(dir_path_3, file_name)
                    with tables.open_file(file_path, mode='r') as h5:
                        # Get values for dataset
                        num_songs = h5.root.metadata.songs.nrows
                        for i in range(num_songs):
                            # Get values from MSD
                            track_id = str(h5.root.metadata.songs.cols.song_id[i])
                            track_name = str(h5.root.metadata.songs.cols.title[i])[2:-1]
                            artists = str(h5.root.metadata.songs.cols.artist_name[i])[2:-1]
                            if artists.__contains__(',') or artists.__contains__(';'):
                                input(f"Multiple artists {artists}... ")
                            year = h5.root.musicbrainz.songs.cols.year[i]
                            key = h5.root.analysis.songs.cols.key[i]
                            mode = h5.root.analysis.songs.cols.mode[i]
                            tempo = h5.root.analysis.songs.cols.tempo[i]
                            time_signature = h5.root.analysis.songs.cols.time_signature[i]
                            timbre_values = None
                            if h5.root.analysis.songs.nrows == i + 1:
                                timbre_values = h5.root.analysis.segments_timbre[h5.root.analysis.songs.cols.idx_segments_timbre[i] : , :]
                            else:
                                timbre_values = h5.root.analysis.songs.cols.idx_segments_timbre[h5.root.analysis.songs.cols.idx_segments_timbre[i] : h5.root.analysis.songs.cols.idx_segments_timbre[i+1], :]

                            # Get values from Spotify Tracks Dataset
                            # genre = None
                            danceability = None
                            energy = None
                            loudness = None
                            instrumentalness = None
                            valence = None

                            # Iterate through dataset
                            found = False
                            for chunk in pd.read_csv(spotify_tracks_file_path, chunksize=1000):
                                for row in chunk.itertuples(index=True):
                                    row_song_name = str(row.track_name).lower()
                                    row_song_artist = str(row.artists).lower()
                                    curr_song_name = track_name.lower()
                                    curr_song_artist = artists.lower()
                                    if row_song_name == curr_song_name: # TODO: consider using __contains__ both ways
                                        if row_song_artist == curr_song_artist:
                                            # Get values
                                            danceability = row.danceability
                                            energy = row.energy
                                            instrumentalness = row.instrumentalness
                                            loudness = row.loudness
                                            valence = row.valence
                                            # genre = row.track_genre

                                            # Update found and break
                                            found = True
                                            break
                                    if found:
                                        break

                            # Write to CSV if corresponding entry is found in Spotify Tracks Dataset
                            if found:
                                print(f"\n{counter}: Found entry for {track_name}\n")
                                with open(output_path, 'a') as f:
                                    writer = csv.writer(f, delimiter=',', quotechar='|')
                                    writer.writerow([
                                        track_id, track_name, artists, year, key, 
                                        mode, tempo, time_signature, danceability, energy, 
                                        loudness, valence, instrumentalness
                                    ])
                            else:
                                print(f"{counter}: No entry found for {track_name}")
                            counter += 1

def create_hdf5_extended_msd(msd_path, csv_path, output_path):
    # Check MSD path provided is valid
    if not os.path.isdir(msd_path):
        raise FileNotFoundError(f"Error: could not find directory {msd_path}")
    
    # Check Spotify Tracks path provided is valid
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Error: could not find directory {msd_path}")

    # FOR DEBUGGING
    counter = 1

    # Get data
    if len(os.listdir(msd_path)) == 0:
        raise FileNotFoundError(f"Error: could not find dataset --- {msd_path} is empty")
    # Iterate through directory provided
    for path_dir_1 in sorted(os.listdir(msd_path)):
        dir_path_1 = os.path.join(msd_path, path_dir_1)
        if len(os.listdir(dir_path_1)) == 0:
            raise FileNotFoundError(f"Error: could not find dataset --- {dir_path_1} is empty")
        # Iterate through subdirectories
        for path_dir_2 in sorted(os.listdir(dir_path_1)):
            dir_path_2 = os.path.join(dir_path_1, path_dir_2)
            if len(os.listdir(dir_path_2)) == 0:
                raise FileNotFoundError(f"Error: could not find dataset --- {dir_path_2} is empty")
            # Iterate through subsubdirectories
            for path_dir_3 in sorted(os.listdir(dir_path_2)):
                dir_path_3 = os.path.join(dir_path_2, path_dir_3)
                # Get files
                for file_name in sorted(os.listdir(dir_path_3)):
                    file_path = os.path.join(dir_path_3, file_name)
                    with tables.open_file(file_path, mode='r') as h5:
                        # Get values for dataset
                        num_songs = h5.root.metadata.songs.nrows
                        for i in range(num_songs):
                            # Get values from MSD
                            song_id = str(h5.root.metadata.songs.cols.song_id[i])[2:-1]
                            track_name = str(h5.root.metadata.songs.cols.title[i])[2:-1]
                            artists = str(h5.root.metadata.songs.cols.artist_name[i])[2:-1] # they ARE separated by semi-colons
                            year = h5.root.musicbrainz.songs.cols.year[i]
                            key = h5.root.analysis.songs.cols.key[i]
                            mode = h5.root.analysis.songs.cols.mode[i]
                            tempo = h5.root.analysis.songs.cols.tempo[i]
                            time_signature = h5.root.analysis.songs.cols.time_signature[i]
                            timbre_values = None
                            if h5.root.analysis.songs.nrows == i + 1:
                                timbre_values = h5.root.analysis.segments_timbre[h5.root.analysis.songs.cols.idx_segments_timbre[i] : , :]
                            else:
                                timbre_values = h5.root.analysis.songs.cols.idx_segments_timbre[h5.root.analysis.songs.cols.idx_segments_timbre[i] : h5.root.analysis.songs.cols.idx_segments_timbre[i+1], :]

                            # Get values from Spotify Tracks Dataset
                            # genre = None
                            danceability = None
                            energy = None
                            loudness = None
                            instrumentalness = None
                            valence = None

                            # Iterate through dataset
                            found = False
                            for chunk in pd.read_csv(csv_path, chunksize=1000):
                                for row in chunk.itertuples(index=True):
                                    row_song_name = str(row.track_name).lower()
                                    row_song_artist = str(row.artists).lower()
                                    curr_song_name = track_name.lower()
                                    curr_song_artist = artists.lower()
                                    if row_song_name == curr_song_name: # TODO: consider using __contains__ both ways
                                        if row_song_artist == curr_song_artist:
                                            # Get values
                                            danceability = row.danceability
                                            energy = row.energy
                                            instrumentalness = row.instrumentalness
                                            loudness = row.loudness
                                            valence = row.valence
                                            # genre = row.track_genre

                                            # Update found and break
                                            found = True
                                            break
                                    if found:
                                        break
                                if found:
                                    break

                            # Write to CSV if corresponding entry is found in Spotify Tracks Dataset
                            if found:
                                print(f"\n{counter}: Found entry for {track_name}\n")
                                track = Song(
                                    song_name=track_name,
                                    artist_name=artists,
                                    release_year=year,
                                    danceability=danceability,
                                    energy=energy,
                                    loudness=loudness,
                                    valence=valence,
                                    instrumentalness=instrumentalness,
                                    key=key,
                                    mode=mode,
                                    tempo=tempo,
                                    time_signature=time_signature,
                                    timbre_values=timbre_values,
                                    song_id=song_id,
                                )
                                create_track_file(output_path, track)
                            else:
                                print(f"{counter}: No entry found for {track_name}")
                            counter += 1

if __name__ == '__main__':
    create_csv_dataset = input("Would you like to create a CSV merged dataset? (y/n) ")
    if create_csv_dataset.lower() == 'y':
        rewrite = input("Would you like to write over merged_dataset (y/n)? ")
        if rewrite.lower() == 'y':
            with open('/home/troyxdp/Documents/University Work/HYP/HYP Source Code/Back End/MergedDataset/merged_dataset.csv', 'w') as f:
                writer = csv.writer(f, delimiter=',', quotechar='|')
                writer.writerow([
                                    'track_id', 'track_name', 'artists', 'year', 'key', 
                                    'mode', 'tempo', 'time_signature', 'danceability', 'energy', 
                                    'loudness', 'valence', 'instrumentalness'
                                ]) 
        dataset_gen = merge_msd_with_csv(
            msd_path='/home/troyxdp/Documents/University Work/HYP/HYP Source Code/Back End/MillionSongSubset',
            spotify_tracks_file_path='/home/troyxdp/Documents/University Work/HYP/HYP Source Code/Back End/MergedDataset/extracted_data.csv',
            output_path='/home/troyxdp/Documents/University Work/HYP/HYP Source Code/Back End/MergedDataset/merged_dataset.csv'
        )

    create_hdf5_dataset = input("Would you like to create a HDF5 merged dataset? (y/n) ")
    if create_hdf5_dataset.lower() == 'y':
        msd_path='/home/troyxdp/Documents/University Work/HYP/HYP Source Code/Back End/MillionSongSubset'
        csv_path = '/home/troyxdp/Documents/University Work/HYP/HYP Source Code/Back End/SpotifyTracksDataset/spotify_tracks_cleaned_data.csv'
        output_path = '/home/troyxdp/Documents/University Work/HYP/HYP Source Code/Back End/MillionSongSpotifyTracksDataset'

        create_hdf5_extended_msd(msd_path, csv_path, output_path)