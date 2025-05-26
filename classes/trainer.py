import numpy as np
import os
import tables
import requests
import gc

from neural_network import NeuralNetwork
from song import Song

class Trainer():

    # Temporary placeholder for genre values
    genres = {
        'electronic': [0, 0, 0],
        'hip_hop': [0, 0, 1],
        'pop': [0, 1, 0],
        'rock': [0, 1, 1],
        'country': [1, 0, 0],
        'blues': [1, 1, 0],
        'jazz': [1, 1, 1]
    }
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

    def get_data(self, path: str):
        # Check path provided is valid
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Error: could not find directory {path}")

        # FOR DEBUGGING
        num_loaded = 0

        # Get data
        dataset = []
        if len(os.listdir(path)) == 0:
            raise FileNotFoundError(f"Error: could not find dataset --- {path} is empty")
        # Iterate through directory provided
        for path_dir_1 in sorted(os.listdir(path)):
            dir_path_1 = os.path.join(path, path_dir_1)
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
                                genre = Trainer.genres['electronic']
                                danceability = h5.root.analysis.songs.cols.danceability[i]
                                energy = h5.root.analysis.songs.cols.energy[i]
                                loudness = h5.root.analysis.songs.cols.loudness[i]
                                key = h5.root.analysis.songs.cols.key[i]
                                mode = h5.root.analysis.songs.cols.mode[i]
                                bpm = h5.root.analysis.songs.cols.tempo[i]
                                time_signature = h5.root.analysis.songs.cols.time_signature[i]
                                mfcc_values = None
                                if h5.root.analysis.songs.nrows == i + 1:
                                    mfcc_values = h5.root.analysis.segments_timbre[h5.root.analysis.songs.cols.idx_segments_timbre[i] : , :]
                                else:
                                    mfcc_values = h5.root.analysis.songs.cols.idx_segments_timbre[h5.root.analysis.songs.cols.idx_segments_timbre[i] : h5.root.analysis.songs.cols.idx_segments_timbre[i+1], :]
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
                                )
                                dataset.append(song)

                                # FOR DEBUGGING
                                num_loaded += 1
                                # print(f"Number of songs loaded = {num_loaded}")

                                yield song



    def _get_access_token(self):
        url = "https://accounts.spotify.com/api/token"
        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }
        # TO-DO: hide the client_id and client_secret
        data = {
            "grant_type": "client_credentials",
            "client_id": "6f6b166bffdc4596b1a5bcd8d53be857",
            "client_secret": "392a29b2f0dd4954a311993b163d72ac"
        }

        response = requests.post(url, headers=headers, data=data)

        if response.status_code >= 400:
            raise Exception
        return response.json()

    def _get_song_id(self, song_name, artist_name, access_token):
        input("Press ENTER to search for song ID")

        if access_token is None:
            access_token = self._get_access_token()['access_token']

        url = 'https://api.spotify.com/v1/search'
        headers = {
            'Authorization': f'Bearer {access_token}'
        }
        params = {
            'q': f'track:{song_name} artist:{artist_name}',
            'type': 'track'
        }
        response = requests.get(url, headers=headers, params=params)

        num_retries = 0
        while response.status_code == 401 and num_retries < 5:
            print("Retrying...")
            num_retries += 1
            access_token = self._get_access_token()
            headers['Authorization'] = f'Bearer {access_token}'
            response = requests.get(url, headers=headers, params=params)
            if response.status_code > 400 and response.status_code != 401:
                raise Exception

        if num_retries == 5:
            raise Exception
        
        track_data = response.json()
        track_id = track_data['tracks']['items'][0]['id']
        
        return access_token, track_id

    def _get_audio_features(self, spotify_id, access_token):
        input("Press ENTER to search for song audio features")

        if access_token is None:
            access_token = self._get_access_token()['access_token']

        url = f'https://api.spotify.com/v1/audio-features/{spotify_id}'
        headers = {
            'Authorization': f'Bearer {access_token}'
        }
        response = requests.get(url, headers=headers)

        num_retries = 0
        while response.status_code >= 400 and num_retries < 5:
            if response.status_code != 401:
                raise Exception
            print("Retrying...")
            num_retries += 1
            access_token = self._get_access_token()
            headers['Authorization'] = f'Bearer {access_token}'
            response = requests.get(url, headers=headers, params=params)

        if num_retries == 5:
            raise Exception

        audio_features_data = response.json()
        return audio_features_data
        
    def clean_data(self, dataset):
        print("Cleaning dataset...")
        to_ret_dataset = []
        access_token = None
        for item in dataset:
            if not isinstance(item, Song) or item is None:
                continue
            if item.danceability == 0.0 or item.energy == 0.0:
                # # Get track ID
                # track_id = None
                # try:
                #     access_token, track_id = self._get_song_id(item.song_name, item.artist_name, access_token)
                # except Exception:
                #     print(f"Cannot find track '{item.song_name}' by {item.artist_name} --- skipping")
                #     continue
                # # Get missing features
                # try:
                #     audio_features = self._get_audio_features(track_id, access_token)
                # except Exception:
                #     print(f"Cannot find audio features for '{item.song_name}' by {item.artist_name} with track ID {track_id} --- skipping")
                #     continue
                # # Set missing values
                # if item.energy == 0.0:
                #     item.energy = audio_features['energy']
                # if item.danceability == 0.0:
                #     item.danceability = audio_features['danceability']
                if item.danceability == 0 and item.energy != 0:
                    print("Found one!")
                if item.danceability != 0 and item.energy == 0:
                    print("Found one!")
                continue
            to_ret_dataset.append(item)
        # Raise exception if there are no items to return                
        # if len(to_ret_dataset) == 0:
        #     raise ValueError("Error: no items to return")
        
        return to_ret_dataset

                

    def normalize_data(self, dataset):
        pass

    def set_training_network(self, nn: NeuralNetwork):
        pass

    def train_model(self):
        pass

    def validate_model(self):
        pass

    def set_hyperparameters(
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
    nn = NeuralNetwork((64, 128, 128, 32), ('relu', 'relu', 'relu'))
    trainer = Trainer(
        training_network=nn,
        initial_lr=0.001,
        final_lr=0.0001,
        num_epochs=120,
        training_dataset_path='/home/troyxdp/Documents/University Work/HYP/HYP Source Code/MillionSongSubset'
    )
    # print(trainer._get_song_id("Interstellar", "Victor Ruiz", None))

    dataset_gen = trainer.get_data('/home/troyxdp/Documents/University Work/HYP/HYP Source Code/MillionSongSubset')
    i = 0
    song_arr = []
    for song in dataset_gen:
        song_arr.append(song)
        i += 1
        if i % 100 == 0:
            print(i)
            song_arr = trainer.clean_data(song_arr)
            print(len(song_arr))
            song_arr = []
            try:
                gc.collect()
            except:
                print("Error")