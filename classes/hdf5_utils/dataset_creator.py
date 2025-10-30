import os
import shutil

import tables
import numpy as np

import classes.hdf5_utils.hdf5_utils as HDF5
from classes.song import Song

# The code below in this file is adapted from Bertin-Mahieux, T. (2010) (https://github.com/tbertinmahieux/MSongsDB/blob/master/PythonSrc/hdf5_utils.py#L391)

def path_from_trackid(trackid):
    """
    Returns the typical path, with the letters[2-3-4]
    of the trackid (starting at 0), hence a song with
    trackid: TRABC1839DQL4H... will have path:
    A/B/C/TRABC1839DQL4H....h5
    """
    p = os.path.join(trackid[2],trackid[3])
    p = os.path.join(p,trackid[4])
    p = os.path.join(p,trackid+'.h5')
    return p

def create_track_file(maindir, track: Song):
    """
    Main function to create an HDF5 song file.
    You got to have the track, song and artist already.
    Returns True if song was created, False otherwise.
    We also check whether the path exists.
    INPUT
       maindir      - main directory of the Million Song Dataset
       track        - Song object
    RETURN
       True if a track file was created, False otherwise
    """
    # Check if path provided exists
    hdf5_path = os.path.join(maindir, path_from_trackid(track.song_id))
    if os.path.exists(hdf5_path):
        return False # file already exists, no stress

    # create file and fill it
    try:
        # try create file and directories
        if not os.path.isdir(os.path.split(hdf5_path)[0]):
            os.makedirs(os.path.split(hdf5_path)[0])

        # check / delete file if exist
        if os.path.isfile(hdf5_path):
            os.remove(hdf5_path)

        # create file
        HDF5.create_song_file(hdf5_path)
        h5 = HDF5.open_h5_file_append(hdf5_path)
        HDF5.fill_hdf5_from_track(h5, track)
        h5.close()
    except Exception as e:
        # close hdf5
        try:
            h5.close()
        except NameError:
            pass
        except ValueError:
            pass

        # delete path
        try:
            os.remove( hdf5_path )
        except IOError:
            pass

        # print and wait
        print('ERROR creating track:', track.song_id)
        print(e)
        return False
    # Return success
    return True

