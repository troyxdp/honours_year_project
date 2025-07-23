import tables

# Code below is adapted from https://github.com/tbertinmahieux/MSongsDB/blob/master/PythonSrc/hdf5_descriptors.py

MAXSTRLEN = 1024

class SongMetaData(tables.IsDescription):
    """
    Class to hold the metadata of one song
    """
    # Identifying features
    song_id = tables.StringCol(32)
    title = tables.StringCol(MAXSTRLEN)
    artist_name = tables.StringCol(MAXSTRLEN)
    # genre = tables.StringCol(MAXSTRLEN)

class SongAnalysis(tables.IsDescription):
    """
    Class to hold the analysis of one song
    """
    # Spotify features
    danceability = tables.Float64Col()
    energy = tables.Float64Col()
    instrumentalness = tables.Float64Col()
    loudness = tables.Float64Col()
    valence = tables.Float64Col()

    # High-level descriptors
    key = tables.IntCol()
    mode = tables.IntCol()
    tempo = tables.Float64Col()
    time_signature = tables.IntCol()

    # Identifier
    track_id = tables.StringCol(32)

    # Timbre segments
    idx_segments_timbre = tables.IntCol()
    
class SongMusicBrainz(tables.IsDescription):
    """
    Class to hold information coming from
    MusicBrainz for one song
    """
    # Release year
    year = tables.IntCol()