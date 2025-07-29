import os

import psycopg2
from dotenv import load_dotenv
import numpy as np

from classes.trainer import Trainer

# Load environment variables from .env file
load_dotenv()

# Get environment variables
DB = os.getenv('DB')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')

# Create connection to database
conn = psycopg2.connect(
    database=DB,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT
)

if __name__ == '__main__':
    num_tracks_to_add = 100
    cursor = conn.cursor()
    for i, track in enumerate(Trainer.get_track_data('/home/troyxdp/Documents/University Work/HYP/HYP Source Code/Back End/MillionSongSpotifyTracksDataset')):
        if i < num_tracks_to_add:
            cursor.execute(
                '''
                INSERT
                    INTO track(track_id, song_name, artist_name, release_year, danceability, energy, loudness, valence, instrumentalness, key, mode, bpm, time_signature, timbre_values, embedding, audio_file_path)
                VALUES
                    (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                ''',
                (
                    track.song_id, track.song_name, track.artists, np.random.randint(1900, 2025), float(track.danceability), float(track.energy), 
                    float(track.loudness), float(track.valence), float(track.instrumentalness), int(track.key), int(track.mode), 
                    float(track.tempo), int(track.time_signature), track.timbre_values.tolist(), np.random.rand(128).tolist(), f"{i}.mp3"
                )
            )
            conn.commit()
        else:
            break