import os
import random

from classes.neural_network import NeuralNetwork

class Recommender():
    
    def __init__(
        self,
    ):
        # set values
        self._selected_tracks_info = [] # list of track IDs for the songs to be played in the set
        self._is_selected_tracks_info_set = False
        self._current_track_info = None # track ID of the current song that is playing
        self._is_current_track_info_set = False
        self._seed_track_info = None
        self._played_tracks = [] # list of tracks that have been played in the set
        self._recommendations = [] # planned ordered list of tracks to play in the set. They are also all of the unplanned tracks


    # SETTER METHODS
    def init_selected_track_ids(self, selected_tracks_info: list):
        # check if seed track is set, which is necessary for setting selected_track_ids
        if not self._is_current_track_info_set:
            raise Exception("Error: seed track is not set")

        if len(selected_tracks_info) == 0:
            raise ValueError("Error: no selected tracks have been provided")
        if self._current_track_info and len(selected_tracks_info) == 1 and selected_tracks_info[0][0] == self._current_track_info[0]:
            raise ValueError("Error: only seed track was provided as selected track")

        # remove seed track from selected_track_ids
        if self._current_track_info:
            for i in range(len(selected_tracks_info)):
                if selected_tracks_info[i][0] == self._current_track_info[0]:
                    selected_tracks_info.pop(i)
                    break
        
        # set values
        self._selected_tracks_info = selected_tracks_info
        self._is_selected_tracks_info_set = True

    def add_to_selected_track_ids(self, tracks_info_to_add: list):
        # add tracks
        curr_num_tracks = len(self._selected_tracks_info)
        for track_info in tracks_info_to_add:
            # check if track from track_info is in selected_tracks_info list
            is_in_selected_tracks = False
            for track in self._selected_tracks_info:
                if track[0] == track_info[0]:
                    is_in_selected_tracks = True
                    break
            if not is_in_selected_tracks and (track_info[0] != self._current_track_info): # check if track was already added or is being played
                self._selected_tracks_info.append(track_info)
                self._recommendations.append(track_info)
        
        # check if any tracks were actually added and adjust recommendations accordingly
        if curr_num_tracks != len(self._selected_tracks_info):
            self._regenerate_recommendations(self._current_track_info)

    def set_seed_track_info(self, seed_track_info: str):
        # check if selected tracks are set - cannot be set before seed track is set
        if self._is_selected_tracks_info_set:
            raise Exception("Error: cannot set selected tracks before setting seed track")
        
        # validate provided seed_track_id and set value
        if not seed_track_info[0].strip():
            raise ValueError("Error: no value provided for seed track ID")
        
        # set values
        self._current_track_info = seed_track_info
        self._seed_track_info = seed_track_info
        self._is_current_track_info_set = True

    def reset_recommender(self):
        # reset all values to defaults seen in constructor
        self._selected_tracks_info = []
        self._is_selected_tracks_info_set = False
        self._current_track_info = None
        self._is_current_track_info_set = False
        self._seed_track_info = None
        self._played_tracks = []
        self._recommendations = []


    # GETTER METHODS
    def get_selected_track_ids(self):
        return [selected_track[0] for selected_track in self._selected_tracks_info]

    def is_selected_tracks_info_set(self):
        return self._is_selected_tracks_info_set
    
    def get_current_track_id(self):
        return self._current_track_info[0]

    def is_current_track_info_set(self):
        return self._is_current_track_info_set
    
    def get_seed_track_id(self):
        return self._seed_track_info[0]

    def get_played_tracks(self):
        return [played_track[0] for played_track in self._played_tracks]
    
    def get_unplayed_track_ids(self):
        return [unplayed_track[0] for unplayed_track in self._recommendations]

    def is_unplayed_tracks(self):
        return len(self._recommendations) > 0

    # get the currently recommended track which is at the front of self.recommendations
    def get_current_recommendation_track_id(self):
        if self._recommendations:
            return self._recommendations[0][0]
        return None
    
    # get the next track recommendation given which track is currently playing
    def get_next_recommendation_track_id(self, curr_track_info):
        # update currently playing track data and list of played tracks
        self._current_track_info = curr_track_info

        # TODO: restructure this logic - sure there is a better way to do it
        # return next recommendation
        if len(self._recommendations) > 1: # user played recommended track
            if curr_track_info[0] == self._recommendations[0][0]: # check that there is something left to recommend
                # remove track from recommendations
                self._recommendations.pop(0)

                # add to array of played tracks
                self._played_tracks.append(curr_track_info)
            else: # did not play the recommended track
                # remove track from recommendations
                is_in_recommendations = False # boolean flag to see if curr_track_id is in recommendations list, i.e. is being played currently
                for i in range(len(self._recommendations)):
                    if self._recommendations[i][0] == curr_track_info[0]:
                        is_in_recommendations = True
                        self._recommendations.pop(i)
                        break
                
                if is_in_recommendations:    
                    # regenerate recommendations
                    self._regenerate_recommendations(curr_track_info) # this is also called to make the first recommendation
                    
                    # add to array of played tracks
                    self._played_tracks.append(curr_track_info)

        # return recommended track
        return self._recommendations[0][0] # if there is nothing else left to recommend or if track not updated, make same recommendation
    
    # FUNCTIONALITY METHODS
    # this method is called to generate new recommendations 
    def _regenerate_recommendations(self, curr_track_id: str): # TODO
        # TODO: implement TSP algorithm here - for now it is just a random shuffle
        random.shuffle(self._recommendations)

    # method to initialize recommendations
    def init_recommendations(self): # TODO
        # TODO: implement TSP algorithm here - for now it is just a random shuffle
        self._recommendations = self._selected_tracks_info.copy()
    