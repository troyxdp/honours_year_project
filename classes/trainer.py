import os
import gc
import csv

import numpy as np
import pandas as pd
import tables

try:
    from neural_network import NeuralNetwork
    from song import Song
except:
    from .neural_network import NeuralNetwork
    from .song import Song



class EpochStatistics():

    def __init__(self, loss: float, epoch_time: float, map_score=None, precision=None, recall=None):
        self._loss = loss
        self._epoch_time = epoch_time
        self._map_score = map_score
        self._precision = precision
        self._recall = recall

    def get_loss(self):
        return self._loss
    
    def get_epoch_time(self):
        return self._epoch_time
    
    def get_map_score(self):
        return self._epoch_time
    
    def get_precision(self):
        return self._epoch_time
    
    def get_recall(self):
        return self._recall

class Trainer():

    dataset_yield_size = 1000

    def __init__(
        self,
        training_network: NeuralNetwork,
        initial_lr: float,
        final_lr: float,
        num_epochs: int,
        training_dataset_path=None,
        validation_dataset_path=None,
        momentum=None, 
        regularization_lambda=None,
        dropout_rate=None
    ):
        # Initialize network
        self.training_network = training_network
        self.best_network = training_network

        # Initialize hyperparameters
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.num_epochs = num_epochs
        self.momentum = momentum
        self.regularization_lambda = regularization_lambda
        self.dropout_rate = dropout_rate

        # Initialize training statistics
        self.model_statistics_per_epoch = []

        # Get dataset directories
        self.training_dataset_path = training_dataset_path
        self.validation_dataset_path = validation_dataset_path



    def merge_datasets(self, msd_path, spotify_tracks_file_path, output_path):
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
                                song_name = str(h5.root.metadata.songs.cols.title[i])[2:-1]
                                artist_name = str(h5.root.metadata.songs.cols.artist_name[i])[2:-1]
                                year = h5.root.musicbrainz.songs.cols.year[i]
                                key = h5.root.analysis.songs.cols.key[i]
                                mode = h5.root.analysis.songs.cols.mode[i]
                                bpm = h5.root.analysis.songs.cols.tempo[i]
                                time_signature = h5.root.analysis.songs.cols.time_signature[i]
                                mfcc_values = None
                                if h5.root.analysis.songs.nrows == i + 1:
                                    mfcc_values = h5.root.analysis.segments_timbre[h5.root.analysis.songs.cols.idx_segments_timbre[i] : , :]
                                else:
                                    mfcc_values = h5.root.analysis.songs.cols.idx_segments_timbre[h5.root.analysis.songs.cols.idx_segments_timbre[i] : h5.root.analysis.songs.cols.idx_segments_timbre[i+1], :]

                                # Get values from Spotify Tracks Dataset
                                genre = None
                                danceability = None
                                energy = None
                                loudness = None
                                valence = None

                                # Iterate through dataset
                                found = False
                                for chunk in pd.read_csv(spotify_tracks_file_path, chunksize=1000):
                                    for row in chunk.itertuples(index=True):
                                        row_song_name = str(row.track_name).lower()
                                        row_song_artist = str(row.artists).lower()
                                        curr_song_name = song_name.lower()
                                        curr_song_artist = artist_name.lower()
                                        if row_song_name == curr_song_name: # TODO: consider using __contains__ both ways
                                            if row_song_artist == curr_song_artist:
                                                # Get values
                                                danceability = row.danceability
                                                energy = row.energy
                                                genre = row.track_genre
                                                instrumentalness = row.instrumentalness
                                                loudness = row.loudness
                                                valence = row.valence
                                                # Update found and break
                                                found = True
                                                break
                                        if found:
                                            break

                                # Write to CSV if corresponding entry is found in Spotify Tracks Dataset
                                if found:
                                    print(f"\n{counter}: Found entry for {song_name}\n")
                                    with open(output_path, 'a') as f:
                                        writer = csv.writer(f, delimiter=',', quotechar='|')
                                        writer.writerow([song_name, artist_name, year, key, mode, bpm, time_signature, genre, danceability, energy, loudness, valence, instrumentalness])
                                else:
                                    print(f"{counter}: No entry found for {song_name}")
                                counter += 1



    def get_data(self, msd_path: str, csv_path: str):
        # Check path provided is valid
        if not os.path.isdir(msd_path):
            raise FileNotFoundError(f"Error: could not find directory {msd_path}")

        # FOR DEBUGGING
        num_loaded = 0

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
                                song_name = str(h5.root.metadata.songs.cols.title[i])[2:-1]
                                artist_name = str(h5.root.metadata.songs.cols.artist_name[i])[2:-1]
                                year = h5.root.musicbrainz.songs.cols.year[i]
                                key = h5.root.analysis.songs.cols.key[i]
                                mode = h5.root.analysis.songs.cols.mode[i]
                                bpm = h5.root.analysis.songs.cols.tempo[i]
                                time_signature = h5.root.analysis.songs.cols.time_signature[i]
                                mfcc_values = None
                                if h5.root.analysis.songs.nrows == i + 1:
                                    mfcc_values = h5.root.analysis.segments_timbre[h5.root.analysis.songs.cols.idx_segments_timbre[i] : , :]
                                else:
                                    mfcc_values = h5.root.analysis.songs.cols.idx_segments_timbre[h5.root.analysis.songs.cols.idx_segments_timbre[i] : h5.root.analysis.songs.cols.idx_segments_timbre[i+1], :]
                                
                                # Get values from Spotify Tracks Dataset
                                genre = None
                                danceability = None
                                energy = None
                                loudness = None
                                valence = None

                                # Iterate through dataset
                                found = False
                                for chunk in pd.read_csv(csv_path, chunksize=1000):
                                    for row in chunk.itertuples(index=True):
                                        row_song_name = str(row.track_name).lower()
                                        row_song_artist = str(row.artists).lower()
                                        curr_song_name = song_name.lower()
                                        curr_song_artist = artist_name.lower()
                                        if row_song_name == curr_song_name:
                                            if row_song_artist == curr_song_artist:
                                                # Get values
                                                danceability = row.danceability
                                                energy = row.energy
                                                genre = row.track_genre
                                                instrumentalness = row.instrumentalness
                                                loudness = row.loudness
                                                valence = row.valence
                                                # Update found and break
                                                found = True
                                                break
                                        if found:
                                            break

                                # Write to CSV if corresponding entry is found in Spotify Tracks Dataset
                                if found:
                                    song = Song(
                                        song_name=song_name,
                                        artist_name=artist_name,
                                        release_year=year,
                                        genre=genre,
                                        danceability=danceability,
                                        energy=energy,
                                        loudness=loudness,
                                        key=key,
                                        mode=mode,
                                        bpm=bpm,
                                        time_signature=time_signature,
                                        mfcc_values=mfcc_values,
                                        valence=valence,
                                        instrumentalness=instrumentalness
                                    )

                                    # FOR DEBUGGING
                                    num_loaded += 1
                                    # print(f"Number of songs loaded = {num_loaded}")

                                    yield song



    def normalize_data(self, dataset):
        pass

    def set_training_network(self, nn: NeuralNetwork):
        pass

    def train_model(self):
        pass

    def validate_model(self):
        pass

    def set_hyperparameters(
        self,
        initial_lr: float,
        final_lr: float,
        num_epochs: int,
        momentum=None,
        reg_const=None,
        dropout=None
    ):
        # Do validity checks
        if initial_lr is None or final_lr is None or num_epochs is None:
            raise AttributeError("Error: initial learning rate, final learning rate, or number of training epochs were not provided")
        if not type(initial_lr) == float or not type(final_lr) == float or not type(num_epochs) == int:
            raise TypeError("Error: incorrect type provided for initial_lr, final_lr, or num_epochs")
        if not momentum is None and not type(momentum) == float:
            raise TypeError("Error: momentum value was provided that was not a float")
        if not reg_const is None and not type(reg_const) == float:
            raise TypeError("Error: reg_const value was provided that was not a float")
        if not dropout is None and not type(dropout) == float:
            raise TypeError("Error: dropout was provided that was not a float")
        if not dropout is None and not (dropout >= 0 and dropout < 1):
            raise ValueError("Error: dropout value was provided that is not in range [0, 1)")

        # Set values
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.num_epochs = num_epochs
        self.momentum = momentum
        self.reg_const = reg_const
        self.dropout = dropout

    def get_curr_training_statistics(self):
        return self.model_statistics_per_epoch



if __name__ == '__main__':
    nn = NeuralNetwork()
    trainer = Trainer(
        training_network=nn,
        initial_lr=0.001,
        final_lr=0.0001,
        num_epochs=120,
        training_dataset_path='/home/troyxdp/Documents/University Work/HYP/HYP Source Code/MillionSongSubset'
    )

    rewrite = input("Would you like to write over merged_dataset (y/n)? ")
    if rewrite.lower() == 'y':
        with open('/home/troyxdp/Documents/University Work/HYP/HYP Source Code/MergedDataset/merged_dataset.csv', 'w') as f:
            writer = csv.writer(f, delimiter=',', quotechar='|')
            writer.writerow(['track_name', 'artists', 'year', 'key', 'mode', 'tempo', 'time_signature', 'track_genre', 'danceability', 'energy', 'loudness', 'valence', 'instrumentalness']) 

    dataset_gen = trainer.merge_datasets(
        msd_path='/home/troyxdp/Documents/University Work/HYP/HYP Source Code/MillionSongSubset',
        spotify_tracks_file_path='/home/troyxdp/Documents/University Work/HYP/HYP Source Code/SpotifyTracksDataset/dataset.csv',
        output_path='/home/troyxdp/Documents/University Work/HYP/HYP Source Code/MergedDataset/merged_dataset.csv'
    )