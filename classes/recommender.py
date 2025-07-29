import os
import random

from classes.neural_network import NeuralNetwork

class Recommender():
    
    def __init__(
        self,
    ):
        # set values
        self._selected_track_ids = [] # list of track IDs for the songs to be played in the set
        self._is_selected_track_ids_set = False
        self._current_track_id = '' # track ID of the current song that is playing
        self._is_current_track_id_set = False
        self._played_tracks = [] # list of tracks that have been played in the set
        self._recommendations = [] # planned ordered list of tracks to play in the set. They are also all of the unplanned tracks


    # SETTER METHODS
    def init_selected_track_ids(self, selected_track_ids: list):
        # check if seed track is set, which is necessary for setting selected_track_ids
        if not self._is_current_track_id_set:
            raise Exception("Error: seed track is not set")

        if len(selected_track_ids) == 0:
            raise ValueError("Error: no selected tracks have been provided")
        if self._current_track_id and len(selected_track_ids) == 1 and selected_track_ids[0] == self._current_track_id:
            raise ValueError("Error: only seed track was provided as selected track")

        # remove seed track from selected_track_ids
        if self._current_track_id:
            for i in range(len(selected_track_ids)):
                if selected_track_ids[i] == self._current_track_id:
                    selected_track_ids.pop(i)
                    break
        
        # set values
        self._selected_track_ids = selected_track_ids
        self._is_selected_track_ids_set = True

    def add_to_selected_track_ids(self, track_ids_to_add: list):
        # add tracks
        curr_num_tracks = len(self._selected_track_ids)
        for track_id in track_ids_to_add:
            if (not track_id in self._selected_track_ids) and (track_id != self._current_track_id): # check if track was already added or is being played
                self._selected_track_ids.append(track_id)
                self._recommendations.append(track_id)
        
        # check if any tracks were actually added and adjust recommendations accordingly
        if curr_num_tracks != len(self._selected_track_ids):
            self._regenerate_recommendations(self._current_track_id)

    def set_seed_track_id(self, seed_track_id: str):
        # check if selected tracks are set - cannot be set before seed track is set
        if self._is_selected_track_ids_set:
            raise Exception("Error: cannot set selected tracks before setting seed track")
        
        # validate provided seed_track_id and set value
        if not seed_track_id.strip():
            raise ValueError("Error: no value provided for seed track ID")
        
        # set values
        self._current_track_id = seed_track_id
        self._is_current_track_id_set = True

    def reset_recommender(self):
        # reset all values to defaults seen in constructor
        self._selected_track_ids = []
        self._is_selected_track_ids_set = False
        self._current_track_id = ''
        self._is_current_track_id_set = False
        self._played_tracks = []
        self._recommendations = []


    # GETTER METHODS
    def get_selected_track_ids(self):
        return self._selected_track_ids

    def is_selected_track_ids_set(self):
        return self._is_selected_track_ids_set
    
    def get_current_track_id(self):
        return self._current_track_id

    def is_current_track_id_set(self):
        return self._is_current_track_id_set
    
    def get_played_tracks(self):
        return self._played_tracks
    
    def get_unplayed_tracks(self):
        return self._recommendations

    def is_unplayed_tracks(self):
        return len(self._recommendations) > 0

    # get the currently recommended track which is at the front of self.recommendations
    def get_current_recommendation(self):
        if self._recommendations:
            return self._recommendations[0]
        return None
    
    # get the next track recommendation given which track is currently playing
    def get_next_recommendation(self, curr_track_id):
        # update currently playing track data and list of played tracks
        self._current_track_id = curr_track_id 
        self._played_tracks.append(curr_track_id)

        # return next recommendation
        if curr_track_id == self._recommendations[0]: # user played recommended track
            if len(self._recommendations) > 1: # check that there is something left to recommend
                # remove track from recommendations
                self._recommendations.pop(0)

                # return next recommended track, which is the first track in the recommendations list
                return self._recommendations[0]
        elif len(self._recommendations) > 1: # user did not play recommended track
            # remove track from recommendations
            for i in range(len(self._recommendations)):
                if self._recommendations[i] == curr_track_id:
                    self._recommendations.pop(i)

            # regenerate recommendations and return recommended track
            self._regenerate_recommendations(curr_track_id)
            return self._recommendations[0]
        
        # nothing to recommend, so return None
        return None
    
    
    # FUNCTIONALITY METHODS
    # this method is called to generate new recommendations 
    def _regenerate_recommendations(self, curr_track_id: str): # TODO
        # TODO: implement TSP algorithm here - for now it is just a random shuffle
        random.shuffle(self._recommendations)

    # method to initialize recommendations
    def init_recommendations(self): # TODO
        # TODO: implement TSP algorithm here - for now it is just a random shuffle
        return random.shuffle(self._selected_track_ids.copy())
    