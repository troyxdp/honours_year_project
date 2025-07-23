import os
import shutil

import tables
import numpy as np

import classes.hdf5_utils.hdf5_utils as HDF5
from classes.song import Song

# The code below in this file is adapted from https://github.com/tbertinmahieux/MSongsDB/blob/master/PythonSrc/hdf5_utils.py#L391

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
    If you pass an open connection to the musicbrainz database, we also use it.
    Returns True if song was created, False otherwise.
    False can mean another thread is already doing that song.
    We also check whether the path exists.
    INPUT
       maindir      - main directory of the Million Song Dataset
       trackid      - Echo Nest track id of the track object
       track        - pyechonest track object
       song         - pyechonest song object
       artist       - pyechonest artist object
       mbconnect    - open musicbrainz pg connection
    RETURN
       True if a track file was created, False otherwise
    """
    # Check if path provided exists
    hdf5_path = os.path.join(maindir, path_from_trackid(track.song_id))
    if os.path.exists(hdf5_path):
        return False # file already exists, no stress

    # create file and fill it
    try_cnt = 0
    try:
        while True: # try until we make it work!
            try:
                # we try one more time
                try_cnt += 1
                if not os.path.isdir(os.path.split(hdf5_path)[0]):
                    os.makedirs(os.path.split(hdf5_path)[0])

                # check / delete tmp file if exist
                if os.path.isfile(hdf5_path):
                    os.remove(hdf5_path)

                # create tmp file
                HDF5.create_song_file(hdf5_path)
                h5 = HDF5.open_h5_file_append(hdf5_path)
                HDF5.fill_hdf5_from_track(h5, track)
                h5.close()
            # we dont panic, delete file, wait and retry
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
                if try_cnt >= 20:
                    print(f'Giving up after {try_cnt} tries')
                    return False
            # move tmp file to real file and rename it --- similar to mv command
            break
    except IOError as e:
        print('GOT Error', e)
        raise
    except OSError as e:
        print('GOT Error', e)
        raise
    # IF WE GET HERE WE'RE GOOD
    return True

