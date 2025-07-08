import requests

# TEST API IS RUNNING
# response = requests.get("http://localhost:8000/")
# print("Status Code:", response.status_code)
# print("Response Body:", response.text)

# TEST TRACK UPLOAD
data = {
    'track_id': "abc123def456",
    'song_name': "Aura",
    'artist_name': "Avoure",
    'release_year': 2020,
    'genre': "electronic",
    'danceability': 0.88,
    'energy': 0.78,
    'loudness': -3.678,
    'valence': 0.541,
    'instrumentalness': 1.0,
    'key': 1,
    'mode': 0,
    'bpm': 123,
    'time_signature': 4,
}
response = requests.post("http://localhost:8000/upload-track", params=data, files=[('files', open('test_files/Avoure - Aura.mp3', 'rb'))])
print("Status Code:", response.status_code)
print("Response Body:", response.text)