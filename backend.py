# Default libraries
import os

# External libraries
import psycopg2
from fastapi import FastAPI
import uvicorn
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Code I wrote
from classes.neural_network import NeuralNetwork
from classes.neural_network import FeedForwardLayer


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
@app.get('/')
def home():
    return "Hello"

# API POST endpoints
class Track(BaseModel):
    track_id: str
    song_name: str
    artist_name: str
    release_year: int | None = None
    genre: list | None = None
    danceability: float | None = None
    energy: float | None = None
    loudness: float | None = None
    valence: float | None = None
    instrumentalness: float | None = None
    key: int | None = None
    mode: int | None = None
    bpm: float | None = None
    time_signature: int | None = None
    mfcc_values: List

@app.post('/upload-song')
def upload_song(song: Track):
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