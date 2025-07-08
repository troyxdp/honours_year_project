# Default libraries
import os
from typing import List

# External libraries
import psycopg2
from fastapi import FastAPI, HTTPException, status, File, UploadFile, Depends
from fastapi.responses import Response
import uvicorn
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import numpy as np

# Code I wrote
from classes.neural_network import NeuralNetwork
from classes.neural_network import FeedForwardLayer


# TODO: potentially add lock to insert and delete queries


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


# TODO: load neural network here
# embedder = NeuralNetwork()
# embedder.load_network(...)


# Create FastAPI app
app = FastAPI()
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_methods=["*"],
                   allow_headers=["*"])

# API GET endpoints


# API POST endpoints
class Track(BaseModel):
    track_id: str
    song_name: str
    artist_name: str
    release_year: int
    genre: str
    danceability: float
    energy: float
    loudness: float
    valence: float
    instrumentalness: float
    key: int
    mode: int
    bpm: float
    time_signature: int
    timbre_values: List[List[float]] | None = None
@app.post('/upload-track')
def upload_track(track: Track = Depends(), files: List[UploadFile] = File(...)):
    # Check number of files uploaded
    if len(files) > 1:
        raise HTTPException(status_code=400, detail="Error: cannot upload more than one track at a time")
    
    # Check if track with same name and artist has already been uploaded
    cursor = conn.cursor()
    try:
        cursor.execute(
            '''
            SELECT
                *
            FROM
                track
            WHERE
                (song_name = %s AND artist_name = %s)
                OR track_id = %s;
            ''',
            (track.song_name, track.artist_name, track.track_id)
        )
        records = cursor.fetchone()
    except Exception as e:
        raise e
    # Check if there are any records
    if not records is None and len(records) > 0:
        # If there are any records returned, song has already been added, so return an error response
        raise HTTPException(status_code=409, detail='Error: song has already been added to the database')
    
    # Validate values in track
    if not is_valid_track(track):
        raise HTTPException(status_code=400, detail="Error: invalid value/s in track provided")
    
    # Calculate timbre values
    if track.timbre_values is None:
        if not len(files) == 0:
            track.timbre_values = calculate_timbre_values(files[0])

    # TODO: add embedding process here
    embedding = np.random.rand(128) # placeholder
    
    # Insert into database
    try:
        cursor.execute(
            '''
            INSERT
                INTO track(track_id, song_name, artist_name, release_year, genre, danceability, energy, loudness, valence, instrumentalness, key, mode, bpm, time_signature, timbre_values, embedding)
            VALUES
                (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
            ''',
            (
                track.track_id, track.song_name, track.artist_name, track.release_year, track.genre, 
                track.danceability, track.energy, track.loudness, track.valence, track.instrumentalness, 
                track.key, track.mode, track.bpm, track.time_signature, track.timbre_values,
                embedding.tolist()
            )
        )
        conn.commit()
    except Exception as e:
       raise e

    # Save file to storage
    with open(f'./tracks/{track.track_id}.{os.path.splitext(files[0].filename)[1]}', 'wb+') as f:
        f.write(files[0].file.read())

    # Return 201 CREATED success response
    return Response(status_code=status.HTTP_201_CREATED)


# Functional Methods
def is_valid_track(track: Track):
    # Check song name, artist name, track ID, and genre aren't only white space
    if track.song_name.strip() == '' or track.artist_name.strip() == '' or track.track_id.strip() == '' or track.genre.strip() == '':
        return False
    
    # Check if track ID contains whitespace
    if track.track_id.__contains__(' ') or track.track_id.__contains__('\t') or track.track_id.__contains__('\n'):
        return False
    
    # Check if song attribute values provided are valid
    if track.danceability < 0 or track.danceability > 1:
        return False
    if track.energy < 0 or track.energy > 1:
        return False
    if track.valence < 0 or track.valence > 1:
        return False
    if track.instrumentalness < 0 or track.instrumentalness > 1:
        return False
    
    # Check key and mode values
    if track.key < 0 or track.key > 12 or not track.mode in (0, 1):
        return False
    
    # Check that time signature is a whole number and >= 1
    if round(track.time_signature) != track.time_signature or track.time_signature < 1:
        return False
    
    # Track data provided is valid, so return True
    return True

# TODO: implement calculating timbre values
def calculate_timbre_values(file: UploadFile):
    ...


# Run API
if __name__ == '__main__':
    uvicorn.run(
        app,
        host='0.0.0.0',
        port='8000',
        reload=False,
        workers=1,
    )