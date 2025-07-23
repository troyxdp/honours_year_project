# Default libraries
import os
from typing import List

# External libraries
import psycopg2
from fastapi import FastAPI, HTTPException, status, File, UploadFile, Depends
from fastapi.responses import Response, JSONResponse
import uvicorn
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import numpy as np

# Code I wrote
from classes.neural_network import NeuralNetwork
from classes.neural_network import FeedForwardLayer
from classes.song import Song

# TODO: add better logging

# Load environment variables from .env file
load_dotenv()

# Get environment variables
DB = os.getenv('DB')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
NEURAL_NETWORK_PATH = os.getenv('NEURAL_NETWORK_PATH')

# Create connection to database
conn = psycopg2.connect(
    database=DB,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT
)

# Load embedding neural network
embedder = NeuralNetwork.load_network(file_path=NEURAL_NETWORK_PATH)

# Create FastAPI app
app = FastAPI()
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_methods=["*"],
                   allow_headers=["*"])



# API GET endpoints
@app.get('/get-recommendation/{current_track_id}') # TODO
def get_recommendation(current_track_id):
    if current_track_id is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Error: no track ID for current track playing was provided")
    # TODO: implement rest of method where either next song on "stack" is recommended or recommendations are recalculated

@app.get('/track-search/{search_query}') # TODO
def track_search(search_query): 
    ... # TODO: implement using SOUNDEX

@app.get('/get-tracks-basic-info/start-position/{start_position}/end-position/{end_position}/sort-field/{sort_field}')
def get_tracks_basic_info(start_position, end_position, sort_field):
    # Get start/end position values as integers
    try:
        start_position = int(start_position)
        end_position = int(end_position)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Error: please provide integer start/end position values")
    
    # Check if start/end parameters provided were valid
    if start_position < 0 or end_position < 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Error: please provide positive start/end position values")
    if start_position >= end_position:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Error: please provide a start position value lower than the end position value")
    
    # Check sort field is valid
    if not sort_field in ('track_id', 'song_name', 'artist_name', 'release_year', 'key', 'mode', 'bpm', 'time_signature', 'genre'):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Error: invalid sort field provided")
    
    cursor = conn.cursor()
    try:
        cursor.execute(
            f"""
            SELECT
                track_id, song_name, artist_name, release_year, key, mode, bpm, time_signature, genre
            FROM
                track
            ORDER BY {sort_field}
            OFFSET %s
            LIMIT %s;
            """,
            (start_position, end_position - start_position)
        )
        records = cursor.fetchall()
    except Exception as e:
        print(e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error: could not retrieve tracks from database")
    
    to_ret = []
    for record in records:
        try:
            record_dict = {
                "track_id": record[0],
                "song_name": record[1],
                "artist_name": record[2],
                "release_year": int(record[3]),
                "key": int(record[4]),
                "mode": int(record[5]),
                "bpm": float(record[6]),
                "time_signature": int(record[7]),
                "genre": record[8]
            }
            to_ret.append(record_dict)
        except TypeError as te:
            print(te)

    return JSONResponse(
        {
            "songs": to_ret,
            "start_position": start_position,
            "end_position": end_position
        }
    )

@app.get('/get-detailed-track-info/{track_id}')
def get_detailed_track_info(track_id):
    # Check if song exists
    cursor = conn.cursor()
    try:
        cursor.execute(
            '''
            SELECT 
                track_id, song_name, artist_name, release_year, danceability, loudness, key, mode, bpm, time_signature, energy, valence, instrumentalness, genre
            FROM
                track
            WHERE
                track_id = %s;
            ''',
            (track_id,)
        )
        record = cursor.fetchone()
    except Exception as e:
        print(e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error: could not perform query to find track with given track ID")
    
    # Check if record is None - if so, then song does not exist in database
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Error: could not find song with given track ID")
    
    # Return JSON object with all the data
    return JSONResponse(
        {
            'track_id': record[0], 
            'song_name': record[1], 
            'artist_name': record[2], 
            'release_year': record[3], 
            'danceability': record[4], 
            'loudness': record[4], 
            'key': record[5], 
            'mode': record[6], 
            'bpm': record[7], 
            'time_signature': record[8], 
            'energy': record[9], 
            'valence': record[10], 
            'instrumentalness': record[11], 
            'genre': record[12]
        }
    )

@app.get('/get-basic-track-info/{track_id}')
def get_basic_track_info(track_id):
    # Check if song exists
    cursor = conn.cursor()
    try:
        cursor.execute(
            '''
            SELECT 
                track_id, song_name, artist_name, release_year, key, mode, bpm, time_signature, genre
            FROM
                track
            WHERE
                track_id = %s;
            ''',
            (track_id,)
        )
        record = cursor.fetchone()
    except Exception as e:
        print(e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error: could not perform query to find track with given track ID")
    
    # Check if record is None - if so, then song does not exist in database
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Error: could not find song with given track ID")
    
    # Return JSON object with all the data
    return JSONResponse(
        {
            'track_id': record[0], 
            'song_name': record[1], 
            'artist_name': record[2], 
            'release_year': record[3], 
            'key': record[4], 
            'mode': record[5], 
            'bpm': record[6], 
            'time_signature': record[7], 
            'genre': record[8]
        }
    )

@app.get('/end-set') # TODO
def end_set():
    ... # TODO: decide how to store currently playing track; how to store tracks selected for the set, and how to store already played tracks, and then clear all of these when this method is called



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
        record = cursor.fetchone()
    except Exception as e:
        print(e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error: could not execute query to search for track")
    # Check if there are any records
    if not record is None:
        # If there are any records returned, song has already been added, so return an error response
        raise HTTPException(status_code=409, detail='Error: song has already been added to the database')
    
    # Validate values in track
    if not is_valid_track(track):
        raise HTTPException(status_code=400, detail="Error: invalid value/s in track provided")
    
    # Calculate timbre values
    if track.timbre_values is None:
        if not len(files) == 0:
            track.timbre_values = calculate_timbre_values(files[0])
            # TODO: uncomment once timbre value calculation has been implemented
            # if len(track.timbre_values) < 16:
            #     raise HTTPException(
            #         status_code=status.HTTP_400_BAD_REQUEST, 
            #         detail="Error: could not generate enough timbre values using the audio file provided. Please provide another audio file"
            #     )

    # TODO: add "shifting" of timbre values according to the code in the MSongsDB repo

    # Create embedding for the track
    song = Song(
        song_name=track.song_name,
        artist_name=track.artist_name,
        release_year=track.release_year,
        genre=track.genre,
        danceability=track.danceability,
        energy=track.energy,
        loudness=track.loudness,
        valence=track.valence,
        instrumentalness=track.instrumentalness,
        key=track.key,
        mode=track.mode,
        tempo=track.bpm,
        time_signature=track.time_signature,
        timbre_values=track.timbre_values
    )
    embedding = get_embedding(song)
    
    # Save file to storage
    audio_file_path = None
    if len(files) > 0:
        audio_file_path = f'./tracks/{track.track_id}.{os.path.splitext(files[0].filename)[1]}'
        with open(audio_file_path, 'wb+') as f:
            f.write(files[0].file.read())

    # Insert into database
    try:
        cursor.execute(
            '''
            INSERT
                INTO track(track_id, song_name, artist_name, release_year, genre, danceability, energy, loudness, valence, instrumentalness, key, mode, bpm, time_signature, timbre_values, embedding, audio_file_path)
            VALUES
                (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
            ''',
            (
                track.track_id, track.song_name, track.artist_name, track.release_year, track.genre, 
                track.danceability, track.energy, track.loudness, track.valence, track.instrumentalness, 
                track.key, track.mode, track.bpm, track.time_signature, track.timbre_values,
                embedding.tolist(), audio_file_path
            )
        )
        conn.commit()
    except Exception as e:
       print(e)
       os.remove(audio_file_path) # delete track
       raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error: could not insert track into database")

    # Return 201 CREATED success response
    return Response(status_code=status.HTTP_201_CREATED)

class SetTrackIDs(BaseModel):
    track_ids: List[str]
@app.post('/select-set-tracks') # TODO
def select_set_tracks(tracks: SetTrackIDs):
    ... # TODO: decide how to store set tracks and implement this method

class SeedTrackID(BaseModel):
    track_id: str
@app.post('/select-seed-track') # TODO
def select_seed_track(track: SeedTrackID):
    track_id = track.track_id
    ... # TODO: decide how to store seed track and implement this method



# API PUT endpoints
class EditTracks(BaseModel):
    original_track: Track
    editted_track: Track
@app.put('/edit-track')
def edit_track(tracks: EditTracks):
    # Check if song is in database
    # TODO: get SELECT query to return NamedTuple so it is easier to check if energy etc. has been changed
    original_track = tracks.original_track
    editted_track = tracks.editted_track
    cursor = conn.cursor()
    try:
        cursor.execute(
            '''
            SELECT
                *
            FROM
                track
            WHERE
                track_id = %s;
            ''',
            (original_track.track_id,)
        )
        record = cursor.fetchone()
    except Exception as e:
        print(e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error: could not execute query to search for track")
    # If not found, return 404
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Error: could not find track with given ID")
    
    # TODO: implement check to see if embedding needs to be changed (i.e. if energy, danceability etc. has been updated)
    # TODO: implement recalculation of embedding
    embedding = np.random.rand(128) 
    
    # Update track with given data
    try:
        cursor.execute(
            '''
            UPDATE 
                track
            SET
                track_id = %s, 
                song_name = %s, 
                artist_name = %s, 
                release_year = %s, 
                genre = %s, 
                danceability = %s, 
                energy = %s, 
                loudness = %s, 
                valence = %s, 
                instrumentalness = %s, 
                key = %s, 
                mode = %s, 
                bpm = %s, 
                time_signature = %s, 
                embedding = %s
            WHERE 
                track_id = %s;
            ''',
            (
                editted_track.track_id, editted_track.song_name, editted_track.artist_name, editted_track.release_year, editted_track.genre,
                editted_track.danceability, editted_track.energy, editted_track.loudness, editted_track.valence, editted_track.instrumentalness,
                editted_track.key, editted_track.mode, editted_track.bpm, editted_track.time_signature, embedding.tolist(),

                original_track.track_id
            )
        )
        conn.commit()
    except Exception as e:
        print(e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error: could not update track values")
    
    # Return success response
    return Response(status_code=status.HTTP_204_NO_CONTENT)

@app.put('/add-set-tracks')
def add_set_tracks(tracks: SetTrackIDs):
    ... # TODO: decide how to store set tracks and implement this method



# API DELETE endpoints
# Delete track from database and delete audio file
@app.put('/delete-track/{track_id}')
def delete_track(track_id):
    # Check if song exists
    cursor = conn.cursor()
    try:
        cursor.execute(
            '''
            SELECT 
                track_id, audio_file_path
            FROM
                track
            WHERE
                track_id = %s;
            ''',
            (track_id,)
        )
        record = cursor.fetchone()
    except Exception as e:
        print(e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error: could not perform query to find track with given track ID")
    
    # Check if record is None - if so, then song does not exist in database
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Error: could not find song with given track ID")
    
    # If song exists, delete it
    try:
        cursor.execute(
            '''
            DELETE
            FROM
                track
            WHERE
                track_id = %s;
            ''',
            (track_id,)
        )
        conn.commit()
    except Exception as e:
        print(e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error: could not delete track from database")
    
    # Delete song file
    if not record[1] is None:
        os.remove(record[1])
        
    # Return success response
    return Response(status_code=status.HTTP_204_NO_CONTENT)
    


# Functional Methods
# Check if track data provided is valid
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

# TODO: implement calculating timbre values - calculate MFCC values and then reduce them to 12 dimensions using PCA according to ChatGPT
def calculate_timbre_values(file: UploadFile):
    ...

def get_embedding(song: Song):
    embedder.set_input(song.get_nn_input())
    embedder.forward()
    return embedder.get_output()
    


# Run API
if __name__ == '__main__':
    uvicorn.run(
        app,
        host='0.0.0.0',
        port='8000',
        reload=False,
        workers=1,
    )