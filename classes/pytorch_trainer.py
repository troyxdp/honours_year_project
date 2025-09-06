import os
import time
import math
import csv

import numpy as np
import pandas as pd
import tables
from tqdm.auto import tqdm
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch

from classes.autoencoder import Autoencoder
from classes.song import Song



class EpochStatistics():

    def __init__(self, loss: float, epoch_time: float):
        self._loss = loss
        self._epoch_time = epoch_time

    def get_loss(self):
        return self._loss
    
    def get_epoch_time(self):
        return self._epoch_time
    
class PyTorchTrainer():

    def __init__(
        self,
        training_network: Autoencoder,
        initial_lr: float,
        final_lr: float,
        num_epochs: int,
        output_folder: str,
        early_stop_threshold=20,
        checkpoint_epoch=10,
        train_percentage=80,
        val_percentage_of_train=20,
        momentum=None, 
        l2_regularization_lambda=None,
        dropout_rate=None,
        clip_score=1.0,
        dataset_path=None,
    ):
        # Initialize network
        self.training_network = training_network
        self.best_network = training_network

        # Initialize hyperparameters
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.num_epochs = num_epochs
        self.momentum = momentum
        self.l2_regularization_lambda = l2_regularization_lambda
        self.dropout_rate = dropout_rate
        self.clip_score = clip_score

        # Initialize training statistics
        self.model_statistics_per_epoch = []

        # Set "non-technical" hyperparameters
        self.dataset_path = dataset_path
        self.train_percentage = train_percentage
        self.test_percentage = 100 - train_percentage
        self.val_percentage_of_train = val_percentage_of_train
        self.output_folder = output_folder
        self.checkpoint_epoch = checkpoint_epoch
        self.early_stop_threshold = early_stop_threshold

    def get_file_paths(self, input_dim: int = None, num_files: int = None):
        file_paths = []
        num_fetched = 0
        for path_dir_1 in os.listdir(self.dataset_path):
            dir_path_1 = os.path.join(self.dataset_path, path_dir_1)
            if not os.path.isdir(dir_path_1):
                continue
            if len(os.listdir(dir_path_1)) == 0:
                raise FileNotFoundError(f"Error: could not find dataset --- {dir_path_1} is empty")
            # Iterate through subdirectories
            for path_dir_2 in os.listdir(dir_path_1):
                dir_path_2 = os.path.join(dir_path_1, path_dir_2)
                if not os.path.isdir(dir_path_2):
                    continue
                if len(os.listdir(dir_path_2)) == 0:
                    raise FileNotFoundError(f"Error: could not find dataset --- {dir_path_2} is empty")
                # Iterate through subsubdirectories
                for path_dir_3 in os.listdir(dir_path_2):
                    dir_path_3 = os.path.join(dir_path_2, path_dir_3)
                    if not os.path.isdir(dir_path_3):
                        continue
                    # Get files
                    for file_name in os.listdir(dir_path_3):
                        if os.path.splitext(file_name)[1] == '.h5':
                            song = self.get_song_data_from_file(os.path.join(dir_path_3, file_name))

                            # Check if song is missing values
                            input = None
                            try:
                                input = song.get_nn_input()
                            except:
                                continue
                            
                            # Check if input dimensions retrieved match those of the model
                            if not input_dim is None:
                                if input_dim != len(input):
                                    continue
                                
                        # Add to training set if everything is valid
                        file_paths.append(os.path.join(dir_path_3, file_name))
                        num_fetched += 1
                        
                        if not num_files is None:
                            if num_fetched >= num_files:
                                return file_paths

        return file_paths
    
    # Adapted from Bertin-Mahieux, T. (2010) https://github.com/tbertinmahieux/MSongsDB/blob/master/PythonSrc/hdf5_getters.py
    # specifically the parts for getting each value from the h5 file
    def get_song_data_from_file(self, file_path):
        # Check if file exists. If not, raise exception
        if not os.path.exists(file_path):
            raise FileNotFoundError
        
        # Get file data
        with tables.open_file(file_path, mode='r') as h5:
            # Get values for dataset
            song_name = str(h5.root.metadata.songs.cols.title[0])[2:-1]
            artist_name = str(h5.root.metadata.songs.cols.artist_name[0])[2:-1]
            year = h5.root.musicbrainz.songs.cols.year[0]
            key = h5.root.analysis.songs.cols.key[0]
            mode = h5.root.analysis.songs.cols.mode[0]
            bpm = h5.root.analysis.songs.cols.tempo[0]
            time_signature = h5.root.analysis.songs.cols.time_signature[0]

            timbre_values = None
            if h5.root.analysis.songs.nrows == 0 + 1:
                timbre_values = h5.root.analysis.segments_timbre[h5.root.analysis.songs.cols.idx_segments_timbre[0] : , :]
            else:
                timbre_values = h5.root.analysis.songs.cols.idx_segments_timbre[h5.root.analysis.songs.cols.idx_segments_pitches[0] : h5.root.analysis.songs.cols.idx_segments_pitches[0+1], :]
            
            chroma_values = None
            if h5.root.analysis.songs.nrows == 0 + 1:
                chroma_values = h5.root.analysis.segments_pitches[h5.root.analysis.songs.cols.idx_segments_timbre[0] : , :]
            else:
                chroma_values = h5.root.analysis.songs.cols.idx_segments_pitches[h5.root.analysis.songs.cols.idx_segments_pitches[0] : h5.root.analysis.songs.cols.idx_segments_pitches[0+1], :]
            
            # Get values from Spotify Tracks Dataset
            danceability = h5.root.analysis.songs.cols.danceability[0]
            energy = h5.root.analysis.songs.cols.energy[0]
            loudness = h5.root.analysis.songs.cols.loudness[0]
            valence = h5.root.analysis.songs.cols.valence[0]
            instrumentalness = h5.root.analysis.songs.cols.instrumentalness[0]

            # Write to CSV if corresponding entry is found in Spotify Tracks Dataset
            song = Song(
                song_name=song_name,
                artist_name=artist_name,
                danceability=danceability,
                energy=energy,
                loudness=loudness,
                key=key,
                mode=mode,
                tempo=bpm,
                time_signature=time_signature,
                timbre_values=timbre_values,
                chroma_values=chroma_values,
                valence=valence,
                instrumentalness=instrumentalness
            )

            # Return fetched data
            return song

    # Adapted from Bertin-Mahieux, T. (2010) https://github.com/tbertinmahieux/MSongsDB/blob/master/PythonSrc/hdf5_getters.py
    # specifically the parts for getting each value from the h5 file
    def get_track_data(dataset_path):
        # Check path provided is valid
        if not os.path.isdir(dataset_path):
            raise FileNotFoundError(f"Error: could not find directory {dataset_path}")

        # Get data
        if len(os.listdir(dataset_path)) == 0:
            raise FileNotFoundError(f"Error: could not find dataset --- {dataset_path} is empty")
        # Iterate through directory provided
        for path_dir_1 in sorted(os.listdir(dataset_path)):
            dir_path_1 = os.path.join(dataset_path, path_dir_1)
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
                                song_id = str(h5.root.metadata.songs.cols.song_id[i])[2:-1]
                                song_name = str(h5.root.metadata.songs.cols.title[i])[2:-1]
                                artist_name = str(h5.root.metadata.songs.cols.artist_name[i])[2:-1]
                                # year = h5.root.musicbrainz.songs.cols.year[i]
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
                                danceability = h5.root.analysis.songs.cols.danceability[i]
                                energy = h5.root.analysis.songs.cols.energy[i]
                                loudness = h5.root.analysis.songs.cols.loudness[i]
                                valence = h5.root.analysis.songs.cols.valence[i]
                                instrumentalness = h5.root.analysis.songs.cols.instrumentalness[i]

                                # return Song object with all of the data
                                song = Song(
                                    song_id=song_id,
                                    song_name=song_name,
                                    artist_name=artist_name,
                                    key=key,
                                    mode=mode,
                                    tempo=bpm,
                                    danceability=danceability,
                                    energy=energy,
                                    loudness=loudness,
                                    instrumentalness=instrumentalness,
                                    valence=valence,
                                    time_signature=time_signature,
                                    timbre_values=mfcc_values,
                                )
                                yield song

    def normalize_song(self, song: Song):
        # Normalize danceability
        if song.danceability < 0:
            raise ValueError("Error: danceability cannot be less than 0")
        if song.danceability > 1:
            song.danceability /= 100
            if song.danceability > 1:
                raise ValueError("Error: invalid value provided for danceability")
            
        # Normalize energy
        if song.energy < 0:
            raise ValueError("Error: energy cannot be less than 0")
        if song.energy > 1:
            song.energy /= 100
            if song.energy > 1:
                raise ValueError("Error: invalid value provided for danceability")
            
        # Normalize valence
        if song.valence < 0:
            raise ValueError("Error: valence cannot be less than 0")
        if song.valence > 1:
            song.valence /= 100
            if song.valence > 1:
                raise ValueError("Error: invalid value provided for valence")
            
        # Normalize instrumentalness
        if song.instrumentalness < 0:
            raise ValueError("Error: instrumentalness cannot be less than 0")
        if song.instrumentalness > 1:
            song.instrumentalness /= 100
            if song.instrumentalness > 1:
                raise ValueError("Error: invalid value provided for instrumentalness")

    def is_missing_values(self, song: Song):
        # Check if missing BPM
        if song.tempo == 0:
            return True
        # Check if missing timbre values
        if len(song.timbre_values) < 16:
            return True
        # Check if missing danceability
        if song.danceability == 0:
            return True
        # Check if missing energy
        if song.energy == 0:
            return True
        # Check if missing valence
        if song.valence == 0:
            return True
        # Not missing anything --- return False
        return False

    def train_model(self):
        # get file paths for dataset as well as number of training, validation, and testing items
        print("\nFetching training and validation data...")
        file_paths = self.get_file_paths(202)
        num_train_val_files = int(len(file_paths) * (self.train_percentage / 100.0))
        num_val_files = int(num_train_val_files * (self.val_percentage_of_train / 100.0))
        num_train_files = num_train_val_files - num_val_files
        print(f"Number of training items: {num_train_files}")
        print(f"Number of validation items: {num_val_files}\n")

        # Create output folder (if it does not already exist)
        if not os.path.isdir(self.output_folder):
            os.makedirs(self.output_folder)

        # lists to store statistics for each epoch
        train_stats = []
        val_stats = []
        best_train_loss = math.inf
        best_val_loss = math.inf
        best_val_loss_epoch = -1

        # Make training parameters
        loss_fn = torch.nn.MSELoss(reduction="sum")

        # start training
        train_start_time = time.time()
        print("Starting training...\n")
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch + 1}...")
            # get learning rate
            lr = self._determine_epoch_learning_rate(epoch, self.num_epochs, self.initial_lr, self.final_lr)
            optimizer = torch.optim.SGD(
                params=self.training_network.parameters(),
                lr=lr,
                momentum=self.momentum,
                weight_decay=self.l2_regularization_lambda,
            )

            # train cycle
            train_cycle_start_time = time.time()
            train_error_this_epoch = 0
            num_train_samples = 0
            self.training_network.train()
            for file_path in tqdm(file_paths[:num_train_files], desc="Training cycle progress: ", ncols=150):
                # get song and feed it forward through network
                song = self.get_song_data_from_file(file_path)
                try:
                    input_value = song.get_nn_input()
                except ValueError:
                    continue
                x = torch.from_numpy(input_value).to(dtype=torch.float32)
                
                # Zero the gradients of the optimizer
                optimizer.zero_grad()

                # Get the output of the network
                outputs = self.training_network.forward(x)

                # Calculate the loss and backpropagate
                loss = loss_fn(outputs, x)
                loss.backward()

                # Update the weights using the gradients obtained from backpropagation
                optimizer.step()

                # Update statistics
                train_error_this_epoch += loss.item()
                num_train_samples += 1
            
            # save statistics
            train_cycle_time = time.time() - train_cycle_start_time
            train_stat = EpochStatistics(train_error_this_epoch, train_cycle_time)
            train_stats.append(train_stat)
            print(f"Training loss: {float(train_error_this_epoch)}")
            print(f"Average training loss: {float(train_error_this_epoch) / num_train_samples}")
            # print(f"Training time: {train_cycle_time}s") # commenting out because tqdm shows time

            # validation cycle
            val_error_this_epoch = 0
            val_cycle_start_time = time.time()
            num_val_samples = 0
            self.training_network.eval()
            with torch.no_grad():
                for file_path in tqdm(file_paths[num_train_files:num_train_files+num_val_files], desc="Validation cycle progress: ", ncols=150):
                    # get song and feed it forward through network
                    song = self.get_song_data_from_file(file_path)
                    try:
                        input_value = song.get_nn_input()
                    except ValueError:
                        continue
                    x = torch.from_numpy(input_value).to(dtype=torch.float32)
                    
                    # Get the output from the network
                    outputs = self.training_network.forward(x)
                    loss = loss_fn(outputs, x)
                    
                    # Update statistics
                    val_error_this_epoch += loss.item()
                    num_val_samples += 1

            # save statistics 
            val_cycle_time = time.time() - val_cycle_start_time
            val_stat = EpochStatistics(val_error_this_epoch, val_cycle_time)
            val_stats.append(val_stat)
            print(f"Validation loss: {float(val_error_this_epoch)}")
            print(f"Average validation loss: {float(val_error_this_epoch) / num_val_samples}")
            # print(f"Validation time: {val_cycle_time}s")  # commenting out because tqdm shows time

            # save model
            if train_error_this_epoch < best_train_loss:
                print("New best train loss!")
                torch.save(self.training_network.state_dict(), os.path.join(self.output_folder, 'best_train_loss_network.pt'))
                best_train_loss = train_error_this_epoch
            if val_error_this_epoch < best_val_loss:
                print("New best val loss!")
                torch.save(self.training_network.state_dict(), os.path.join(self.output_folder, 'best_val_loss_network.pt'))
                best_val_loss_epoch = epoch
            if (epoch + 1) % self.checkpoint_epoch == 0:
                print(f"Checkpoint save at epoch {epoch + 1}")
                torch.save(self.training_network.state_dict(), os.path.join(self.output_folder, f"checkpoint_epoch_{epoch + 1}_save.pt"))
            # early termination if no improvement has been seen in validation loss for a set number of epochs
            if epoch - self.early_stop_threshold > best_val_loss_epoch:
                print(f"\nTerminating training early - no improvement seen in {self.early_stop_threshold} epochs")
                break

            print()

        # save last network
        torch.save(self.training_network.state_dict(), os.path.join(self.output_folder, "last.pt"))
                
        # get total training time
        train_end_time = time.time() - train_start_time
        print(f"Training completed in {train_end_time / 3600} hours")
        
        print("Displaying stats graphs...")
            
        # get epoch numbers for plotting purposes
        epoch_nums = range(len(train_stats))

        # plot training loss per epoch
        train_loss_values = [epoch_stat.get_loss() for epoch_stat in train_stats]
        plt.plot(epoch_nums, train_loss_values)
        plt.xlabel("Epochs")
        plt.ylabel("Training Loss")
        plt.title("Training Loss per Epoch")
        plt.show()

        # plot validation loss per epoch
        val_loss_values = [epoch_stat.get_loss() for epoch_stat in val_stats]
        plt.plot(epoch_nums, val_loss_values)
        plt.xlabel("Epochs")
        plt.ylabel("Validation Loss")
        plt.title("Validation Loss per Epoch")
        plt.show()

        # plot training time per epoch
        train_time_values = [epoch_stat.get_epoch_time() for epoch_stat in train_stats]
        plt.plot(epoch_nums, train_time_values)
        plt.xlabel("Epochs")
        plt.ylabel("Training Time (seconds)")
        plt.title("Training Time Elapsed per Epoch")
        plt.show()

        # plot validation time per epoch
        val_time_values = [epoch_stat.get_epoch_time() for epoch_stat in val_stats]
        plt.plot(epoch_nums, val_time_values)
        plt.xlabel("Epochs")
        plt.ylabel("Validation Time (seconds)")
        plt.title("Validation Time Elapsed per Epoch")
        plt.show()

        print("Writing stats to CSV...")

        # write train stats to a csv file
        with open(os.path.join(self.output_folder, 'train_stats.csv'), 'w') as f:
            writer = csv.writer(f, delimiter=',', quotechar='|')
            writer.writerow(['epoch_number', 'loss', 'epoch_time'])
            for i, train_stat in enumerate(train_stats):
                writer.writerow([i+1, train_stat.get_loss(), train_stat.get_epoch_time()])

        # write val stats to a csv file
        with open(os.path.join(self.output_folder, 'val_stats.csv'), 'w') as f:
            writer = csv.writer(f, delimiter=',', quotechar='|')
            writer.writerow(['epoch_number', 'loss', 'epoch_time'])
            for i, val_stat in enumerate(val_stats):
                writer.writerow([i+1, val_stat.get_loss(), val_stat.get_epoch_time()])

    def _determine_epoch_learning_rate(self, epoch, num_epochs, initial_lr, final_lr):
        return initial_lr + epoch * ((final_lr - initial_lr) / (num_epochs - 1)) # num_epochs - 1 so that it cancels with epoch on the largest value of epoch