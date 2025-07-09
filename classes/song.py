import numpy as np

class EmbeddingLengthException(Exception):
    pass

class Song():

    embedding_len = 32 # length of embedding vector. This is the initial value and will change

    def __init__(
        self,
        song_name: str,
        artist_name: str,
        release_year: int,
        genre: list,
        danceability: float,
        energy: float,
        loudness: float,
        valence: float,
        instrumentalness: float,
        key: int,
        mode: int,
        bpm: float,
        time_signature: int,
        timbre_values,
        audio_file_path:str=None,
        embedding=None,
        fourier_transform_vector=None,
    ):
        # Values which are necessary to set
        self.song_name = song_name
        self.artist_name = artist_name
        self.release_year = release_year

        # Spotify Attribuutes
        self.danceability = danceability
        self.energy = energy
        self.loudness = loudness
        self.valence = valence
        self.instrumentalness = instrumentalness
        self.genre = genre

        # MSD attributes
        self.key = key
        self.mode = mode
        self.bpm = bpm
        self.time_signature = time_signature
        self.timbre_values = timbre_values

        # Values which are not necessary to set
        self.audio_file_path = audio_file_path

        # Values which have to be set using methods
        self.embedding = embedding
        self.fourier_transform_vector = fourier_transform_vector

    def get_nn_input(self):
        nn_input = []

        # Add first 16 timbre values
        if len(self.timbre_values) < 16:
            raise ValueError("Error: not enough timbre values to use for embedding")
        for mfcc_value in self.timbre_values[:16]:
            nn_input.extend(mfcc_value) 
        # + LENGTH 192

        # Get key 3D vector - DOUBLE CHECK THIS
        key, mode = self.get_camelot_wheel_value()
        x, y = np.cos(((key - 1) / 6) * np.pi), np.sin(((key - 1) / 6) * np.pi)
        theta = ((-0.5 + mode) / 6) * np.pi
        x = x * np.cos(theta)
        y = y * np.cos(theta)
        z = np.sin(theta) 
        nn_input.extend([x, y, z]) # + LENGTH 3

        # Get BPM 2D vector - pretty sure this is right but DOUBLE CHECK THIS
        if self.bpm > 159:
            while self.bpm > 159:
                self.bpm /= 2
        elif self.bpm < 80:
            while self.bpm < 80:
                self.bpm *= 2
        normalized_bpm = (self.bpm - 80) / 80
        x, y = np.cos(normalized_bpm * 2 * np.pi), np.sin(normalized_bpm * 2 * np.pi)
        nn_input.extend([x, y]) # + LENGTH 2

        # TODO: add genre handling - the following is temporary
        genre_encoding = self.get_genre_encoding() 
        nn_input.extend(genre_encoding) # + LENGTH 7 --- 125 total genres, encoding each genre as a binary number

        # Add the rest of the features (assuming they are all normalized)
        nn_input.extend([
            self.danceability,
            self.energy,
            self.loudness,
            self.valence,
            self.instrumentalness
        ]) # + LENGTH 5

        return np.array(nn_input) # TOTAL LENGTH 209
        
    def get_genre_encoding(self):
        # TODO: implement a genre encoding
        return [0, 0, 0, 0, 0, 0, 1]

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
