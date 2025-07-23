import os

import tables
import numpy as np

import classes.hdf5_utils.hdf5_descriptors as DESC
from classes.song import Song

# The code below in this file is adapted from https://github.com/tbertinmahieux/MSongsDB/blob/master/PythonSrc/hdf5_utils.py#L391

def create_song_file(h5filename: str, title='H5 Song File', force=False, complevel=1):
    """
    Create a new HDF5 file for a new song.
    If force=False, refuse to overwrite an existing file
    Raise a ValueError if it's the case.
    Other optional param is the H5 file.
    Setups the groups, each containing a table 'songs' with one row:
    - metadata
    - analysis
    DETAIL
    - we set the compression level to 1 by default, it uses the ZLIB library
      to disable compression, set it to 0
    """
    # Check if file exists
    if not force:
        if os.path.exists(h5filename):
            raise ValueError('file exists, can not create HDF5 song file')
        
    # Create the H5 file
    h5 = tables.open_file(h5filename, mode='w', title='H5 Song File')

    # Set filter level
    h5.filters = tables.Filters(complevel=complevel, complib='zlib')

    # setup the groups and tables
    # Metadata group
    group = h5.create_group("/", 'metadata', 'metadata about the song')
    table = h5.create_table(group, 'songs', DESC.SongMetaData, 'table of metadata for one song')
    r = table.row
    r.append() # filled with default values 0 or '' (depending on type)
    table.flush()

    # Analysis group
    group = h5.create_group("/", 'analysis', 'Echo Nest analysis of the song')
    table = h5.create_table(group, 'songs', DESC.SongAnalysis, 'table of Echo Nest analysis for one song')
    r = table.row
    r.append() # filled with default values 0 or '' (depending on type)
    table.flush()

    # MusicBrainz group
    group = h5.create_group("/", 'musicbrainz', 'data about the song coming from MusicBrainz')
    table = h5.create_table(group, 'songs', DESC.SongMusicBrainz, 'table of data coming from MusicBrainz')
    r = table.row
    r.append() # filled with default values 0 or '' (depending on type)
    table.flush()

    # create arrays
    group = h5.root.analysis
    h5.create_earray(group, 'segments_timbre', tables.Float64Atom(shape=()), (0, 12), 'array of timbre of segments (MFCC-like)')

    # close it, done
    h5.close()


def fill_hdf5_from_track(h5, track: Song):
    """
    Fill an open hdf5 using all the content in a track object
    from the Echo Nest python API
    """
    # get the metadata table, fill it
    metadata = h5.root.metadata.songs

    metadata.cols.song_id[0] = track.song_id
    metadata.cols.artist_name[0] = track.artists
    metadata.cols.title[0] = track.song_name

    metadata.flush()

    # get the analysis table, fill it
    analysis = h5.root.analysis.songs

    analysis.cols.track_id[0] = track.song_id

    analysis.cols.key[0] = track.key
    analysis.cols.mode[0] = track.mode
    analysis.cols.tempo[0] = track.tempo
    analysis.cols.time_signature[0] = track.time_signature

    analysis.cols.danceability[0] = track.danceability
    analysis.cols.energy[0] = track.energy
    analysis.cols.instrumentalness[0] = track.instrumentalness
    analysis.cols.loudness[0] = track.loudness
    analysis.cols.valence[0] = track.valence

    analysis.flush()

    # analysis arrays (segments)
    group = h5.root.analysis

    group.segments_timbre.append(np.array(track.timbre_values))
    analysis.cols.idx_segments_timbre[0] = 0

    analysis.flush()

def open_h5_file_read(h5filename):
    """
    Open an existing H5 in read mode.
    """
    return tables.open_file(h5filename, mode='r')

def open_h5_file_append(h5filename):
    """
    Open an existing H5 in append mode.
    """
    return tables.open_file(h5filename, mode='a')