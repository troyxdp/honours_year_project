import numpy as np

class EmbeddingLengthException(Exception):
    pass

class Song():

    embedding_len = 32 # length of embedding vector. This is the initial value and will change

    def __init__(
        self,
        song_name: str,
        artist_name: str,
        key: int,
        mode: int,
        tempo: float,
        time_signature: int,
        danceability: float,
        energy: float,
        loudness: float,
        valence: float,
        instrumentalness: float,
        timbre_values,
        genre: str = 'electronic',
        release_year: int = 2025,
        song_id=None,
        audio_file_path:str=None,
        embedding=None,
    ):
        # Values which are necessary to set
        self.song_id = song_id
        self.song_name = song_name
        self.artists = artist_name
        self.release_year = release_year
        self.genre = genre

        # Spotify Attribuutes
        self.danceability = danceability
        self.energy = energy
        self.loudness = loudness
        self.valence = valence
        self.instrumentalness = instrumentalness
        # self.genre = genre

        # MSD attributes
        self.key = key
        self.mode = mode
        self.tempo = tempo
        self.time_signature = time_signature
        self.timbre_values = timbre_values

        # Values which are not necessary to set
        self.audio_file_path = audio_file_path

        # Values which have to be set using methods
        self.embedding = embedding

    def get_nn_input(self):
        nn_input = []

        # Add first 16 timbre values
        if len(self.timbre_values) < 16:
            raise ValueError("Error: not enough timbre values to use for embedding")
        for mfcc_value in self.timbre_values[:16]:
            nn_input.extend(mfcc_value) 
        # + LENGTH 192

        # Get key 3D vector --- convert from spherical coordinates to cartesian with theta (key angle), phi (mode angle+), and r = 1 as params
        key, mode = self.get_camelot_wheel_value()
        theta = ((key - 1) / 6) * np.pi
        phi = ((2.5 + mode) / 6) * np.pi
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(theta)
        nn_input.extend([x, y, z]) # + LENGTH 3

        # Get BPM 2D vector - pretty sure this is right but DOUBLE CHECK THIS
        if self.tempo == 0:
            raise ValueError("Error: tempo is not provided")
        
        if self.tempo > 159:
            while self.tempo > 159:
                self.tempo /= 2
        elif self.tempo < 80:
            while self.tempo < 80:
                self.tempo *= 2
        normalized_bpm = (self.tempo - 80) / 80
        x, y = np.cos(normalized_bpm * 2 * np.pi), np.sin(normalized_bpm * 2 * np.pi)
        nn_input.extend([x, y]) # + LENGTH 2

        # TODO: add genre handling - the following is temporary
        # genre_encoding = self.get_genre_encoding() 
        # nn_input.extend(genre_encoding) # + LENGTH 7 --- 125 total genres, encoding each genre as a binary number

        # Add the rest of the features (assuming they are all normalized)
        nn_input.extend([
            self.danceability,
            self.energy,
            self.loudness,
            self.valence,
            self.instrumentalness
        ]) # + LENGTH 5

        return np.array(nn_input) # TOTAL LENGTH 209 (with genre) 202 (without genre)
        
    # def get_genre_encoding(self):
    #     # TODO: implement a genre encoding
    #     return [0, 0, 0, 0, 0, 0, 1]

    def get_camelot_wheel_value(self):
        # key number: camelot number
        minor_key_numbers = {
            0: 5, # Cm
            1: 12, # Dbm
            2: 7, # Dm
            3: 2, # Ebm
            4: 9, # Em
            5: 4, # Fm
            6: 11, # F#m
            7: 6, # Gm
            8: 1, # Abm
            9: 8, # Am
            10: 3, # Bbm
            11: 10, # Bm
        }
        major_key_numbers = {
            0: 8, # C
            1: 3, # Db
            2: 10, # D
            3: 5, # Eb
            4: 12, # E
            5: 7, # F
            6: 2, # F#
            7: 9, # G
            8: 4, # Ab
            9: 11, # A
            10: 6, # Bb
            11: 1, # B
        }
        if self.mode == 0:
            return minor_key_numbers[int(self.key)], self.mode
        return major_key_numbers[int(self.key)], self.mode

    def get_embedding(self):
        return self.embedding

    def set_embedding(self, embedding):
        # Check embedding provided is valid
        # Check it is not null
        if embedding is None:
            raise ValueError("Error: 'embedding' passed into function is None")
        # Check it is a list or numpy array
        if not isinstance(embedding, np.ndarray) or isinstance(embedding, list):
            raise TypeError("Error: 'embedding' is not a numpy array or a list")
        # Check it is the correct length
        if len(embedding) != Song.embedding_len:
            raise EmbeddingLengthException("Error: length of embedding provided is wrong")

        # Set attribute value
        self.embedding = embedding

    def get_discrete_fourier_transform(self):
        if self.audio_file_path is None:
            raise AttributeError("Error: 'audio' attribute is not set")
        
    def __str__(self):
        to_ret = f'"{self.song_name}" by {self.artists}'

        to_ret += f'\nKey/Mode: {self.key}/{self.mode}'
        to_ret += f'\n{self.tempo} BPM in {self.time_signature} time'
        
        to_ret += f'\nDanceability: {self.danceability}'
        to_ret += f'\nEnergy: {self.energy}'
        to_ret += f'\nLoudness: {self.loudness}'
        to_ret += f'\nValence: {self.valence}'
        to_ret += f'\nInstrumentalness: {self.instrumentalness}'

        return to_ret
    
    def __repr__(self):
        return self.__str__()
    
    def to_string(self):
        return self.__str__()