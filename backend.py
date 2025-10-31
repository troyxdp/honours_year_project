# Default libraries
import os
from typing import List
import json

# External libraries
import psycopg2
from fastapi import FastAPI, HTTPException, status, File, UploadFile, Form
from fastapi.responses import Response, JSONResponse, FileResponse
import uvicorn
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import numpy as np
import librosa

# Code I wrote
from classes.recommender import Recommender
from classes.neural_network import NeuralNetwork, FeedForwardLayer
from classes.song import Song

# List of valid traversal algorithms
VALID_TRAVERSAL_ALGORITHMS = ('greedy_nearest_neighbour', 'optimal_path')



# Load environment variables from .env file
load_dotenv()

# Get environment variables
DB = os.getenv('DB')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
NEURAL_NETWORK_PATH = os.getenv('NEURAL_NETWORK_PATH')

# Load embedding neural network
embedder = NeuralNetwork.load_network(file_path=NEURAL_NETWORK_PATH)
print(NEURAL_NETWORK_PATH)

# Load recommender
recommender = Recommender()

# Create FastAPI app
app = FastAPI()
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_methods=["*"],
                   allow_headers=["*"])



# API GET endpoints
@app.get('/get-next-recommendation/{current_track_id}/') # TODO: test new Recommender
def get_next_recommendation(current_track_id, traversal_algorithm: str = "greedy_nearest_neighbour"):
    # check that a non-null value was passed to endpoint
    if current_track_id is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Error: no track ID for current track playing was provided")
    
    # check valid traversal algorithm was provided
    if not traversal_algorithm in VALID_TRAVERSAL_ALGORITHMS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Error: invalid traversal algorithm provided. Choose either 'greedy_nearest_neighbour' or 'optimal_path'")

    # check if current_track_id exists in database
    conn = get_conn()
    cursor = conn.cursor()
    try:
        cursor.execute(
            '''
            SELECT
                track_id, embedding
            FROM
                track
            WHERE
                track_id=%s;
            ''',
            (current_track_id,)
        )
        record = cursor.fetchone()
        cursor.close()
    except Exception as e:
        print(e)
        cursor.close()
        conn.close()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error: could not perform query to see if current track is in database")

    # check if a record was found
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Error: could not find current track playing in database")
    
    # check that the seed track has been provided
    if not recommender.is_current_track_info_set():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Error: no seed track has been provided")
    
    # get track ID of next recommendation (if any)
    recommended_track_id = recommender.get_next_recommendation_track_id((record[0], json.loads(record[1])), traversal_algorithm)
    if not recommended_track_id:
        return JSONResponse({
            "track_id": "",
            "song_name": "",
            "artist_name": "",
            "release_year": -1,
            "key": -1,
            "mode": -1,
            "bpm": -1,
            "time_signature": -1,
            "genre": ""
        })
        
    # get track data of recommended track
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
            (recommended_track_id,)
        )
        record = cursor.fetchone()
    except Exception as e:
        print(e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error: could not perform query to find recommended track")
    finally:
        cursor.close()
        conn.close()
    
    # Check if record is None - if so, then song does not exist in database
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Error: could not find song with given track ID")
    
    # Return JSON object with all the data
    track_id = record[0]
    song_name = record[1]
    artist_name = record[2]
    release_year = int(record[3])
    key = int(record[4])
    mode = int(record[5])
    bpm = float(record[6])
    time_signature = int(record[7])
    genre = record[8] if record[8] else ''
    return JSONResponse(
        {
            "track_id": track_id,
            "song_name": song_name,
            "artist_name": artist_name,
            "release_year": release_year,
            "key": key,
            "mode": mode,
            "bpm": bpm,
            "time_signature": time_signature,
            "genre": genre
        }
    )

@app.get('/get-unplayed-tracks/start-position/{start_position}/end-position/{end_position}/sort-field/{sort_field}/') # TODO: implement search query; test
def get_unplayed_tracks(start_position: int, end_position: int, sort_field: str, search_query: str | None = None):
    if search_query:
        print("YAY SEWCH KWEWY!!! Still got impwement tho")

    # response if there are no unplayed tracks
    if not recommender.is_unplayed_tracks():
        return JSONResponse({
            "songs": None,
            "start_position": 0,
            "end_position": 15
        })
    
    # run SQL query to get all the info about the unplayed tracks
    conn = get_conn()
    cursor = conn.cursor()
    unplayed_track_ids = tuple(recommender.get_unplayed_track_ids())
    try:
        cursor.execute(
            f'''
            SELECT 
                track_id, song_name, artist_name, release_year, key, mode, bpm, time_signature, genre
            FROM
                track
            WHERE
                track_id IN %s
            ORDER BY 
                {sort_field}
            OFFSET %s
            LIMIT %s;
            ''',
            (unplayed_track_ids, start_position, end_position - start_position)
        )
        records = cursor.fetchall()
    except Exception as e:
        print(e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error: could not perform query to get unplayed tracks")
    finally:
        cursor.close()
        conn.close()
    
    # create dicts array using retrieved data
    to_ret = []
    for record in records:
        try:
            track_id = record[0]
            song_name = record[1]
            artist_name = record[2]
            release_year = int(record[3])
            key = int(record[4])
            mode = int(record[5])
            bpm = float(record[6])
            time_signature = int(record[7])
            genre = record[8] if record[8] else ''
            record_dict = {
                "track_id": track_id,
                "song_name": song_name,
                "artist_name": artist_name,
                "release_year": release_year,
                "key": key,
                "mode": mode,
                "bpm": bpm,
                "time_signature": time_signature,
                "genre": genre
            }
            to_ret.append(record_dict)
        except TypeError as te:
            print(te)

    # return a response
    return JSONResponse(
        {
            "songs": to_ret,
            "start_position": start_position,
            "end_position": end_position
        }
    )

@app.get('/get-tracks-basic-info/start-position/{start_position}/end-position/{end_position}/sort-field/{sort_field}') # TODO: implement search query
def get_tracks_basic_info(start_position: int, end_position: int, sort_field: str, search_query: str | None = None):
    if search_query:
        print("YAY SEWCH KWEWY!!! Still got impwement tho")
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
    
    conn = get_conn()
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
    finally:
        cursor.close()
        conn.close()
    
    to_ret = []
    for record in records:
        try:
            track_id = record[0]
            song_name = record[1]
            artist_name = record[2]
            release_year = int(record[3])
            key = int(record[4])
            mode = int(record[5])
            bpm = float(record[6])
            time_signature = int(record[7])
            genre = record[8] if record[8] else ''
            record_dict = {
                "track_id": track_id,
                "song_name": song_name,
                "artist_name": artist_name,
                "release_year": release_year,
                "key": key,
                "mode": mode,
                "bpm": bpm,
                "time_signature": time_signature,
                "genre": genre
            }
            to_ret.append(record_dict)
        except Exception as e:
            print(e)

    return JSONResponse(
        {
            "songs": to_ret,
            "start_position": start_position,
            "end_position": end_position
        }
    )

@app.get('/get-detailed-track-info/{track_id}') # TODO: test
def get_detailed_track_info(track_id):
    # Check if song exists
    conn = get_conn()
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
    finally:
        cursor.close()
        conn.close()
    
    # Check if record is None - if so, then song does not exist in database
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Error: could not find song with given track ID")
    
    # Return JSON object with all the data
    track_id = record[0]
    song_name = record[1]
    artist_name = record[2]
    release_year = int(record[3])
    danceability = float(record[4])
    loudness = float(record[5])
    key = int(record[6])
    mode = int(record[7])
    bpm = float(record[8])
    time_signature = int(record[9])
    energy = float(record[10])
    valence = float(record[11])
    instrumentalness = float(record[12])
    genre = record[13] if record[13] else ''
    return JSONResponse(
        {
            'track_id': track_id, 
            'song_name': song_name, 
            'artist_name': artist_name, 
            'release_year': release_year, 
            'danceability': danceability, 
            'loudness': loudness, 
            'key': key, 
            'mode': mode, 
            'bpm': bpm, 
            'time_signature': time_signature, 
            'energy': energy, 
            'valence': valence, 
            'instrumentalness': instrumentalness, 
            'genre': genre
        }
    )

@app.get('/get-unselected-tracks-basic-info/start-position/{start_position}/end-position/{end_position}/sort-field/{sort_field}')
def get_unselected_tracks(start_position: int, end_position: int, sort_field: str, search_query: str | None = None):
    if search_query:
        print("YAY SEWCH KWEWY!!! Still got impwement tho")
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
    
    # List of selected tracks
    selected_track_ids = recommender.get_selected_track_ids()
    selected_track_ids.append(recommender.get_seed_track_id())
    
    conn = get_conn()
    cursor = conn.cursor()
    try:
        cursor.execute(
            f"""
            SELECT
                track_id, song_name, artist_name, release_year, key, mode, bpm, time_signature, genre
            FROM
                track
            WHERE
                track_id NOT IN %s
            ORDER BY {sort_field}
            OFFSET %s
            LIMIT %s;
            """,
            (tuple(selected_track_ids), start_position, end_position - start_position)
        )
        records = cursor.fetchall()
    except Exception as e:
        print(e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error: could not retrieve tracks from database")
    finally:
        cursor.close()
        conn.close()
    
    to_ret = []
    for record in records:
        try:
            track_id = record[0]
            song_name = record[1]
            artist_name = record[2]
            release_year = int(record[3])
            key = int(record[4])
            mode = int(record[5])
            bpm = float(record[6])
            time_signature = int(record[7])
            genre = record[8] if record[8] else ''
            record_dict = {
                "track_id": track_id,
                "song_name": song_name,
                "artist_name": artist_name,
                "release_year": release_year,
                "key": key,
                "mode": mode,
                "bpm": bpm,
                "time_signature": time_signature,
                "genre": genre
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

@app.get('/get-basic-track-info/{track_id}')
def get_basic_track_info(track_id):
    # Check if song exists
    conn = get_conn()
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
    finally:
        cursor.close()
        conn.close()
    
    # Check if record is None - if so, then song does not exist in database
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Error: could not find song with given track ID")
    
    # Return JSON object with all the data
    track_id = record[0]
    song_name = record[1]
    artist_name = record[2]
    release_year = int(record[3])
    key = int(record[4])
    mode = int(record[5])
    bpm = float(record[6])
    time_signature = int(record[7])
    genre = record[8] if record[8] else ''
    return JSONResponse(
        {
            "track_id": track_id,
            "song_name": song_name,
            "artist_name": artist_name,
            "release_year": release_year,
            "key": key,
            "mode": mode,
            "bpm": bpm,
            "time_signature": time_signature,
            "genre": genre
        }
    )

@app.get('/get-audio-file/{track_id}')
def get_audio_file(track_id: str):
    # Check if song exists
    conn = get_conn()
    cursor = conn.cursor()
    try:
        cursor.execute(
            '''
            SELECT 
                audio_file_path
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
    finally:
        cursor.close()
        conn.close()

    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Error: could not find song with given track ID")
    
    # If file exists, return it
    audio_file_path = record[0]
    return FileResponse(audio_file_path)


@app.get('/end-set')
def end_set():
    recommender.reset_recommender()
    return Response(status_code=status.HTTP_204_NO_CONTENT)



# API POST endpoints
@app.post('/upload-track') # TODO: fix
def upload_track(
        track_id: str = Form(...), 
        song_name: str = Form(...),
        artist_name: str = Form(...),
        release_year: int = Form(...),
        genre: str = Form(...),
        danceability: float = Form(...),
        energy: float = Form(...),
        loudness: float = Form(...),
        valence: float = Form(...),
        instrumentalness: float = Form(...),
        key: int = Form(...),
        mode: int = Form(...),
        bpm: float = Form(...),
        time_signature: int = Form(...),
        files: List[UploadFile] = File(...)
    ):
    # Check number of files uploaded
    if len(files) > 1:
        raise HTTPException(status_code=400, detail="Error: cannot upload more than one track at a time")
    if len(files) < 1:
        raise HTTPException(status_code=400, detail="Error: no file provided for analysis")
    
    # Validate values in track
    track_id = track_id.upper()
    if not is_valid_track(track_id, song_name, artist_name, release_year, genre, danceability, energy, loudness, valence, instrumentalness, key, mode, bpm, time_signature):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Error: invalid value/s in track provided")
    
    # Check if track with same name and artist has already been uploaded
    conn = get_conn()
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
            (song_name, artist_name, track_id)
        )
        record = cursor.fetchone()
        cursor.close()
    except Exception as e:
        print(e)
        cursor.close()
        conn.close()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error: could not execute query to search for track")
    
    # Check if there are any records
    if not record is None:
        # If there are any records returned, song has already been added, so return an error response
        conn.close()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail='Error: song has already been added to the database')
    
    # Save file to storage --- not doing for now
    audio_file_path = f'./tracks/{track_id}{os.path.splitext(files[0].filename)[1]}'
    with open(audio_file_path, 'wb+') as f:
        f.write(files[0].file.read())
    
    # Calculate timbre values
    try:
        chroma_values = calculate_chroma_values(audio_file_path)
    except Exception as e:
        print(e)
        os.remove(audio_file_path)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Error: could not analyze audio file provided")

    # Create embedding for the track
    try:
        song = Song(
            song_id=track_id,
            song_name=song_name,
            artist_name=artist_name,
            release_year=release_year,
            genre=genre,
            danceability=danceability,
            energy=energy,
            loudness=loudness,
            valence=valence,
            instrumentalness=instrumentalness,
            key=key,
            mode=mode,
            tempo=bpm,
            time_signature=time_signature,
            chroma_values=chroma_values
        )
        embedding = get_embedding(song)
    except Exception as e:
        print(e)
        os.remove(audio_file_path)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Error: could not get embedding for track")
    
    # Insert into database
    cursor = conn.cursor()
    try:
        cursor.execute(
            '''
            INSERT
                INTO track(track_id, song_name, artist_name, release_year, genre, danceability, energy, loudness, valence, instrumentalness, key, mode, bpm, time_signature, pitch_values, embedding, audio_file_path)
            VALUES
                (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
            ''',
            (
                track_id, song_name, artist_name, release_year, genre, 
                danceability, energy, loudness, valence, instrumentalness, 
                key, mode, bpm, time_signature, chroma_values.tolist(), 
                embedding.tolist(), audio_file_path
            )
        )
        conn.commit()
    except Exception as e:
       print(e)
       conn.rollback()
       os.remove(audio_file_path)
       raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error: could not insert track into database")
    finally:
        cursor.close()
        conn.close()

    # Return 201 CREATED success response
    return Response(status_code=status.HTTP_201_CREATED)

class SetTrackIDs(BaseModel):
    track_ids: List[str]
@app.post('/select-set-tracks')
def select_set_tracks(tracks: SetTrackIDs):
    # check that the seed track has been provided
    if not recommender.is_current_track_info_set():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Error: no seed track has been provided")
    
    # check that the parameters provided are valid
    track_ids = tracks.track_ids
    if len(track_ids) == 0:
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Error: no track IDs were provided")
    if len(track_ids) == 1:
        if track_ids[0] == recommender.get_current_track_id(): # check if provided track is the seed track
            return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Error: only seed track was provided for set")
            
    # check that at least one of the non-seed track IDs provided are in the database
    conn = get_conn()
    cursor = conn.cursor()
    try:
        cursor.execute(
            '''
            SELECT
                track_id, embedding
            FROM
                track
            WHERE
                track_id IN %s;
            ''', (tuple(track_ids),))
        records = cursor.fetchall()
    except Exception as e:
       print(e)
       raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error: could not search database for track IDs")
    finally:
        cursor.close()
        conn.close()
    
    # check if any records were returned
    if len(records) == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Error: no tracks were found in the database with the provided track IDs")
    
    # check if only seed track was provided or found
    if len(records) == 1:
        if records[0][0] == recommender.get_current_track_id():
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Error: only duplicate of seed track found in provided values")
            
    # if one or more non-seed track values were found, add them to the unplayed_tracks and recommended_tracks lists
    selected_tracks_info = [(record[0], json.loads(record[1])) for record in records]
    try:
        recommender.init_selected_track_ids(selected_tracks_info)
        recommender.init_recommendations()
    except Exception as e:
        print(e)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Error: cannot select tracks for a set before setting seed track")
    
    # return success response
    return Response(status_code=status.HTTP_202_ACCEPTED)

class SeedTrackID(BaseModel):
    track_id: str
@app.post('/select-seed-track')
def select_seed_track(track: SeedTrackID):
    if track is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Error: no data provided")
    if track.track_id is None or track.track_id.strip() == '':
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Error: null or empty value provided for seed track ID")
    
    # check that seed track is in database
    conn = get_conn()
    cursor = conn.cursor()
    try:
        cursor.execute(
            '''
            SELECT
                track_id, embedding
            FROM 
                track
            WHERE
                track_id = %s;
            ''', (track.track_id,)
        )
        record = cursor.fetchone()
    except Exception as e:
        print(e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error: could not search database to find provided seed track ID")
    finally:
        cursor.close()
        conn.close()

    # Check if any records were returned, i.e. if seed track is in database
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Error: could not find track with given ID in database")
    
    # Set seed track value
    try:
        recommender.set_seed_track_info((record[0], json.loads(record[1])))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.__str__())
    # return success response
    return Response(status_code=status.HTTP_202_ACCEPTED)



# API PUT endpoints
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
class EditTracks(BaseModel):
    original_track_id: str
    editted_track: Track
@app.put('/edit-track') # TODO: test
def edit_track(tracks: EditTracks):
    # Store parameters in variables
    original_track_id = tracks.original_track_id
    editted_track = tracks.editted_track

    # Check original_track is in DB and get timbre values while at it
    conn = get_conn()
    cursor = conn.cursor()
    try:
        cursor.execute(
            '''
            SELECT
                pitch_values
            FROM
                track
            WHERE
                track_id = %s;
            ''',
            (original_track_id,)
        )
        record = cursor.fetchone()
        cursor.close()
    except Exception as e:
        print(e)
        cursor.close()
        conn.close()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error: could not execute query to search for track")
    
    # If not found, return 404
    if record is None:
        conn.close()
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Error: could not find track with given ID")
    
    # Check if editted track is valid
    is_valid = is_valid_track(
        editted_track.track_id, 
        editted_track.song_name, 
        editted_track.artist_name, 
        editted_track.release_year, 
        editted_track.genre, 
        editted_track.danceability, 
        editted_track.energy, 
        editted_track.loudness, 
        editted_track.valence, 
        editted_track.instrumentalness, 
        editted_track.key, 
        editted_track.mode, 
        editted_track.bpm, 
        editted_track.time_signature
    )
    if not is_valid:
        conn.close()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Error: invalid values provided for editted track")
    
    chroma_values = np.array(record[0])
    
    song = Song(
        song_id=editted_track.track_id,
        song_name=editted_track.song_name,
        artist_name=editted_track.artist_name,
        release_year=editted_track.release_year,
        genre=editted_track.genre,
        danceability=editted_track.danceability,
        energy=editted_track.energy,
        loudness=editted_track.loudness,
        valence=editted_track.valence,
        instrumentalness=editted_track.instrumentalness,
        key=editted_track.key,
        mode=editted_track.mode,
        tempo=editted_track.bpm,
        time_signature=editted_track.time_signature,
        chroma_values=chroma_values
    )
    embedding = get_embedding(song)
    
    # Update track with given data
    cursor = conn.cursor()
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

                original_track_id
            )
        )
        conn.commit()
    except Exception as e:
        print(e)
        conn.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error: could not update track values")
    finally:
        cursor.close()
        conn.close()
    
    # Return success response
    return Response(status_code=status.HTTP_204_NO_CONTENT)

@app.put('/add-set-tracks')
def add_set_tracks(tracks: SetTrackIDs):
    # check that the seed track has been provided
    if not recommender.is_current_track_info_set():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Error: no seed track has been provided")
    
    # check that the parameters provided are valid
    track_ids = tracks.track_ids
    if len(track_ids) == 0:
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Error: no track IDs were provided")
    if len(track_ids) == 1:
        if track_ids[0] == recommender.get_current_track_id(): # check if provided track is the seed track
            return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Error: only seed track was provided for set")
            
    # check that at least one of the non-seed track IDs provided are in the database
    conn = get_conn()
    cursor = conn.cursor()
    try:
        cursor.execute(
            '''
            SELECT
                track_id, embedding
            FROM
                track
            WHERE
                track_id IN %s;
            ''', (tuple(track_ids),))
        records = cursor.fetchall()
    except Exception as e:
        print(e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error: could not search database for track IDs")
    finally:
        cursor.close()
        conn.close()
    
    # check if any records were returned
    if len(records) == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Error: no tracks were found in the database with the provided track IDs")
    
    # if one or more non-seed track values were found, add them to the unplayed_tracks and recommended_tracks lists
    selected_tracks_info = [(record[0], json.loads(record[1])) for record in records]
    try:
        recommender.add_to_selected_track_ids(selected_tracks_info)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Error: cannot select tracks for a set before setting seed track")
    
    # return success response
    return Response(status_code=status.HTTP_202_ACCEPTED)


# API DELETE endpoints
# Delete track from database and delete audio file
@app.delete('/delete-track/{track_id}') # TODO: test
def delete_track(track_id):
    # Check if song exists
    conn = get_conn()
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
        cursor.close()
    except Exception as e:
        print(e)
        cursor.close()
        conn.close()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error: could not perform query to find track with given track ID")
    
    # Check if record is None - if so, then song does not exist in database
    if record is None:
        conn.close()
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Error: could not find song with given track ID")
    
    # If song exists, delete it
    cursor = conn.cursor()
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
        conn.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error: could not delete track from database")
    finally:
        cursor.close()
        conn.close()
    
    # Delete song file ---- commented out for now. Need to implement functionality. May even not allow saving of uploaded mp3 files
    if not record[1] is None:
        os.remove(record[1])
        
    # Return success response
    return Response(status_code=status.HTTP_204_NO_CONTENT)
    


# Functional Methods
# Check if track data provided is valid
def is_valid_track(track_id, song_name, artist_name, release_year, genre, danceability, energy, loudness, valence, instrumentalness, key, mode, bpm, time_signature) -> bool:
    # Check song name, artist name, track ID, and genre aren't only white space
    if song_name.strip() == '' or artist_name.strip() == '' or track_id.strip() == '' or genre.strip() == '':
        return False
    
    # Check if track ID contains whitespace
    if track_id.__contains__(' ') or track_id.__contains__('\t') or track_id.__contains__('\n'):
        return False
    
    # Check if song attribute values provided are valid
    if danceability < 0 or danceability > 1:
        return False
    if energy < 0 or energy > 1:
        return False
    if valence < 0 or valence > 1:
        return False
    if instrumentalness < 0 or instrumentalness > 1:
        return False
    
    # Check key and mode values
    if key < 0 or key > 12 or not mode in (0, 1):
        return False
    
    # Check that time signature is a whole number and >= 1
    if round(time_signature) != time_signature or time_signature < 1:
        return False
    
    # Track data provided is valid, so return True
    return True

# calculate chroma values
def calculate_chroma_values(file_path: str, num_values: int = 16) -> np.ndarray: # TODO: implement
    # Load file
    y, sr = librosa.load(file_path)

    # Get Constant Q Chroma values
    chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr, norm=1, n_chroma=12) # shape is (12, n_frames)

    # Get note onsets
    onsets = librosa.onset.onset_detect(y=y, sr=sr)

    # Get chroma values at each note onset and return them
    chroma_at_onsets = np.zeros((16, 12))
    for i in range(num_values):
        chroma_at_onsets[i] = chroma_cq[:, onsets[i]]
    return chroma_at_onsets
    


def get_embedding(song: Song) -> np.ndarray: 
    embedder.set_input(song.get_nn_input())
    embedder.feed_forward()
    return embedder.get_output()

def get_conn():
    return psycopg2.connect(
        database=DB,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )

# Run API
if __name__ == '__main__':
    uvicorn.run(
        app,
        host='0.0.0.0',
        port='8000',
        reload=False,
        workers=1,
    )