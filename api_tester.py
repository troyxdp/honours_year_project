import requests

# TEST TRACK UPLOAD
# data = {
#     'track_id': "abc123def456",
#     'song_name': "Aura",
#     'artist_name': "Avoure",
#     'release_year': 2020,
#     'genre': "electronic",
#     'danceability': 0.88,
#     'energy': 0.78,
#     'loudness': -3.678,
#     'valence': 0.541,
#     'instrumentalness': 1.0,
#     'key': 1,
#     'mode': 0,
#     'bpm': 123,
#     'time_signature': 4,
# }
# response = requests.post("http://localhost:8000/upload-track", params=data, files=[('files', open('test_files/Avoure - Aura.mp3', 'rb'))])
# print("Status Code:", response.status_code)
# print("Response Body:", response.text)

# TEST GET TRACK INFO
start_position = 0
end_position = 50
sort_field = 'song_name'
url = f"http://localhost:8000/get-tracks-basic-info/start-position/{start_position}/end-position/{end_position}/sort-field/{sort_field}"
response = requests.get(url)
print("Status Code:", response.status_code)
print("Response Body:", response.text)