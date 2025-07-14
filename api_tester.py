import requests

# TEST TRACK UPLOAD
data = {
    'track_id': "def456ghi789",
    'song_name': "Two Zulu Men in Ibiza",
    'artist_name': "Enoo Napa, DJ Merlon",
    'release_year': 2022,
    'genre': "electronic",
    'danceability': 0.78,
    'energy': 0.85,
    'loudness': -2.436,
    'valence': 0.567,
    'instrumentalness': 1.0,
    'key': 2,
    'mode': 1,
    'bpm': 120,
    'time_signature': 4,
}
response = requests.post("http://localhost:8000/upload-track", params=data, files=[('files', open('test_files/DJ Merlon & Enoo Napa - Two Zulu Men In Ibiza (MIDH Premiere).mp3', 'rb'))])
print("Status Code:", response.status_code)
print("Response Body:", response.text)

# TEST GET TRACKS INFO
start_position = 0
end_position = 50
sort_field = 'song_name'
url = f"http://localhost:8000/get-tracks-basic-info/start-position/{start_position}/end-position/{end_position}/sort-field/{sort_field}"
response = requests.get(url)
print("Status Code:", response.status_code)
print("Response Body:", response.text)

# TEST GET TRACK INFO
track_id = 'def456ghi789'
url = f"http://localhost:8000/get-basic-track-info/{track_id}"
response = requests.get(url)
print("Status Code:", response.status_code)
print("Response Body:", response.text)