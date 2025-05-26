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
        key: int,
        mode: int,
        bpm: float,
        time_signature: int,
        mfcc_values,
        audio=None
    ):
        # Values which are necessary to set
        self.song_name = song_name
        self.artist_name = artist_name
        self.release_year = release_year
        self.genre = genre
        self.danceability = danceability
        self.energy = energy
        self.loudness = loudness
        self.key = key
        self.mode = mode
        self.bpm = bpm
        self.time_signature = time_signature
        self.mfcc_values = mfcc_values

        # Values which are not necessary to set
        self.audio = audio

        # Values which have to be set using methods
        self.embedding = None
        self.fourier_transform_vector = None

    def get_embedding(self):
        return self.embedding

    def set_embedding(self, embedding):
        # Check embedding provided is valid
        # Check it is not null
        if embedding is None:
            raise ValueError("Error: 'embedding' passed into function is None")
        # Check it is a list or numpy array
        if not isinstance(embedding, numpy.ndarray) or isinstance(embedding, list):
            raise TypeError("Error: 'embedding' is not a numpy array or a list")
        # Check it is the correct length
        if len(embedding) != Song.embedding_len:
            raise EmbeddingLengthException("Error: length of embedding provided is wrong")

        # Set attribute value
        self.embedding = embedding

    def get_discrete_fourier_transform(self):
        if self.audio is None:
            raise AttributeError("Error: 'audio' attribute is not set")
