import itertools
import random

import numpy as np

HELD_KARP_CUTOFF = 17

class Recommender():
    
    def __init__(
        self,
        traversal_algorithm:str="greedy_nearest_neighbour",
    ):
        # set values
        self._selected_tracks_info = [] # list of track IDs for the songs to be played in the set
        self._is_selected_tracks_info_set = False
        self._current_track_info = None # track ID of the current song that is playing
        self._is_current_track_info_set = False
        self._seed_track_info = None
        self._played_tracks = [] # list of tracks that have been played in the set
        self._recommendations = [] # planned ordered list of tracks to play in the set. They are also all of the unplanned tracks
        self._traversal_algorithm = traversal_algorithm


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
        self._traversal_algorithm = 'greedy_nearest_neighbour'


    # GETTER METHODS
    def get_track(self, track_id):
        # Go through played tracks
        for track in self._played_tracks:
            if track[0] == track_id:
                return track
        # Go through unplayed tracks (recommendations)
        for track in self._recommendations:
            if track[0] == track_id:
                return track
        # Check seed track
        if track_id == self._seed_track_info[0]:
            return self._seed_track_info
        # Not found - raise exception
        raise Exception("Error: could not find track with given ID")

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
    
    def get_recommendation_path(self):
        path = [self._current_track_info]
        path.extend(self._recommendations)
        return path

    # get the currently recommended track which is at the front of self.recommendations
    def get_current_recommendation_track_id(self):
        if self._recommendations:
            return self._recommendations[0][0]
        return None
    
    # get the next track recommendation given which track is currently playing
    def get_next_recommendation_track_id(self, curr_track_info: tuple, traversal_algorithm: str = None):
        # update currently playing track data and list of played tracks
        self._current_track_info = curr_track_info

        # return next recommendation
        if len(self._recommendations) > 0: # check that there are still recommendations
            if curr_track_info[0] == self._recommendations[0][0]: # song that was played was recommended track
                # remove track from recommendations
                self._recommendations.pop(0)

                # add to array of played tracks
                self._played_tracks.append(curr_track_info)

                # regenerate recommendations if traversal algorithm changed
                if not traversal_algorithm is None and traversal_algorithm != self._traversal_algorithm:
                    # update traversal algorithm
                    self._traversal_algorithm = traversal_algorithm

                    # regenerate recommendations
                    self._regenerate_recommendations(curr_track_info)

            else: # did not play the recommended track
                # remove track from recommendations
                is_in_recommendations = False # boolean flag to see if curr_track_id is in recommendations list, i.e. is being played currently
                for i in range(len(self._recommendations)):
                    if self._recommendations[i][0] == curr_track_info[0]:
                        is_in_recommendations = True
                        self._recommendations.pop(i)
                        break
                
                # check if traversal algorithm has changed or new track was played that was not recommended track 
                if is_in_recommendations or (not traversal_algorithm is None and traversal_algorithm != self._traversal_algorithm):
                    # update traversal algorithm
                    self._traversal_algorithm = traversal_algorithm

                    # regenerate recommendations
                    self._regenerate_recommendations(curr_track_info) # this is also called to make the first recommendation
                    
                    # updated played tracks if first condition of above if loop was true
                    if is_in_recommendations:
                        # add to array of played tracks
                        self._played_tracks.append(curr_track_info)
        else:
            return None
        
        # return recommended track
        return self._recommendations[0][0] if self._recommendations else None # if there is nothing else left to recommend or if track not updated, make same recommendation
    

    # FUNCTIONALITY METHODS
    # this method is called to generate new recommendations 
    def _regenerate_recommendations(self, curr_track_info: tuple): # TODO
        # get data for generating recommendations
        unplayed_tracks = self._recommendations.copy()
        
        # check if current track data was found
        if curr_track_info is None:
            raise ValueError("Error: could not find current track data")
        
        # generate recommendations
        if self._traversal_algorithm == 'greedy_nearest_neighbour' or len(self._selected_tracks_info) > HELD_KARP_CUTOFF:
            # Get greedy nearest neighbour path
            self._recommendations = self._get_greedy_nearest_neighbour_path(curr_track_info, unplayed_tracks.copy())
        elif self._traversal_algorithm == 'optimal_path' or len(self._selected_tracks_info) > HELD_KARP_CUTOFF:
            # Get optimal path
            # TODO: test this algorithm on self._selected_tracks_info length = 1
            self._recommendations = self._get_optimal_path(curr_track_info, unplayed_tracks.copy())
        else:
            raise Exception("Error: traversal algorithm provided is invalid. Choose either 'greedy_nearest_neighbour' or 'optimal_path', or leave blank, which defaults to 'greedy_nearest_neighbour'")

    # method to initialize recommendations
    def init_recommendations(self): 
        # generate recommendations
        # generate recommendations using greedy nearest neighbour if that is selected traversal algorithm or number
        if self._traversal_algorithm == 'greedy_nearest_neighbour' or len(self._selected_tracks_info) > HELD_KARP_CUTOFF:
            # Get greedy nearest neighbour path
            self._recommendations = self._get_greedy_nearest_neighbour_path(self._seed_track_info, self._selected_tracks_info.copy())
        elif self._traversal_algorithm == 'optimal_path':
            # Get optimal path
            self._recommendations = self._get_optimal_path(self._seed_track_info, self._selected_tracks_info.copy())
        elif self._traversal_algorithm == 'random':
            # Get random path (used for testing)
            self._recommendations = self._get_random_path(self._seed_track_info, self._selected_tracks_info.copy())


    # greedy nearest neighbour search
    def _get_greedy_nearest_neighbour_path(self, curr_track: tuple, tracks: list) -> list:
        # path being generated
        recommendation_path = []
        
        # find next track
        nearest_neighbour_index = 0
        nearest_neighbour_distance = self._get_euclidean_distance(curr_track[1], tracks[0][1])
        for i, track in enumerate(tracks[1:]):
            dist = self._get_euclidean_distance(curr_track[1], track[1])
            if dist < nearest_neighbour_distance:
                nearest_neighbour_index = i + 1
                nearest_neighbour_distance = dist

        # add next track to recommendation path and remove from unplayed tracks
        recommendation_path.append(tracks[nearest_neighbour_index])
        tracks.pop(nearest_neighbour_index)

        # generate the rest of the path
        last_added_track = recommendation_path[0]
        while len(tracks) > 0:
            # get nearest neighbour
            nearest_neighbour_index = 0
            nearest_neighbour_distance = self._get_euclidean_distance(last_added_track[1], tracks[0][1])
            for i, track in enumerate(tracks[1:]):
                dist = self._get_euclidean_distance(last_added_track[1], track[1])
                if dist < nearest_neighbour_distance:
                    nearest_neighbour_index = i + 1
                    nearest_neighbour_distance = dist
            
            # update last added track; add to recommendation path; remove from tracks
            last_added_track = tracks[nearest_neighbour_index]
            recommendation_path.append(tracks[nearest_neighbour_index])
            tracks.pop(nearest_neighbour_index)

        return recommendation_path

    def _get_euclidean_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        sigma = 0
        for xi, yi in zip(x, y):
            sigma += (xi - yi) ** 2
        return np.sqrt(sigma)
    
    # get all of the subsets of given length for given set. Values are sorted in increasing order in each subset
    def _get_subsets(self, set, length):
        set_list = list(set)
        return itertools.combinations(set_list, length)

    # get distances between embeddings of all the tracks provided
    def _get_dists(self, curr_track: tuple, tracks: list[tuple]):
        # Combine inputs into single list
        all_tracks = [curr_track]
        all_tracks.extend(tracks)

        # Get distances
        dists = np.zeros((len(all_tracks), len(all_tracks)))
        for i in range(len(dists)):
            for j in range(i + 1, len(dists)):
                dists[i][j] = self._get_euclidean_distance(all_tracks[i][1], all_tracks[j][1])
                dists[j][i] = dists[i][j]

        # Return distances
        return dists
    
    # calculate distances between tracks and get the optimal path using the Held-Karp algorithm
    def _get_optimal_path(self, curr_track: tuple, tracks: list):
        # get distances and calculate optimal path
        dists = self._get_dists(curr_track, tracks.copy())
        optimal_path, _ = self._held_karp(dists)

        # translate optimal path into recommendations list
        recommendations = []
        for i in optimal_path[1:]:
            recommendations.append(tracks[i-1])

        # return recommendations list
        return recommendations

    # Made with assistance from https://en.wikipedia.org/wiki/Held%E2%80%93Karp_algorithm
    # Made with assistance from https://stackoverflow.com/questions/69902373/can-you-help-explain-this-held-karp-tsp-pseudocode
    def _held_karp(self, dists: np.ndarray):
        # Initialize values
        cities = [_ for _ in range(len(dists))]
        g = {} # keeps track of the G values for each "subpath" 
        parent = {} # keeps track of second-last city visited in each "subpath" 
        e = cities[0] # number of initial city

        # Initialize g with 1 step subpaths
        for k in range(1, len(dists)):
            g[((k,), k)] = dists[0][k]
            parent[((k,), k)] = 0

        # Add subpaths of increasing length s - this is the "dynamic programming loop"
        for s in range(2, len(dists)):
            # Get the subsets of cities excluding the starting city of length s
            subsets = self._get_subsets(cities[1:], s)
            for S in subsets:
                # Go through each end destination k
                for k in S:
                    # Find the path of minimum cost (or rather set of vertices in the path) that ends at k with second last vertex m
                    S_minus_k = [i for i in S if i != k]
                    S_minus_k = tuple(S_minus_k)
                    min_cost = np.inf
                    arg_min_m = -1
                    for m in S:
                        if m == k:
                            continue
                        g_val = g[(S_minus_k, m)]
                        cost_val = dists[m][k]
                        cost = g_val + cost_val
                        if cost < min_cost:
                            min_cost = cost
                            arg_min_m = m
                    g[(tuple(S), k)] = min_cost
                    parent[(tuple(S), k)] = arg_min_m

        # Find optimal cost
        optimal_cost = np.inf
        arg_min_k = -1
        for k in cities[1:]:
            cost = g[(tuple(cities[1:]), k)] + dists[k][e]
            if cost < optimal_cost:
                optimal_cost = cost
                arg_min_k = k

        # Find optimal path by backtracking through parents
        # Made with assistance from https://www.youtube.com/watch?v=-JjA4BLQyqE
        path = []
        subset = tuple(cities[1:])
        k = arg_min_k
        while len(subset) != 0:
            path.append(k)
            m = parent[(subset, k)]
            subset = tuple([city for city in subset if city != k])
            k = m
        path.append(0)
        path = list(reversed(path))

        return path, optimal_cost

    # Generate a random path
    def _get_random_path(self, curr_track: tuple, tracks: list[tuple]):
        # Randomly shuffle the tracks
        random.shuffle(tracks)
        return tracks


if __name__ == '__main__':
    curr_track = ('a', np.array([1, 1]))
    tracks = [
        ('f', np.array([-1, 1])),
        ('c', np.array([1, -1])),
        ('b', np.array([2, 0])),
        ('e', np.array([-2, 0])),
        ('d', np.array([-1, -1])),
    ]

    # test initializing recommendations
    recommender = Recommender()
    recommender.set_seed_track_info(curr_track)
    recommender.init_selected_track_ids(tracks.copy())
    recommender.init_recommendations()
    print("Recommendations path after initialization:")
    print(recommender._recommendations)
    print()

    # test getting next recommendation when previous recommendation taken
    next_rec_id = recommender.get_next_recommendation_track_id(('b', np.array([2, 0])))
    print("Next recommendation given user played b:")
    print(next_rec_id)
    print("Recommendation path:")
    print(recommender._recommendations)
    print()

    # test getting next recommendation when previous recommendation NOT taken
    next_rec_id = recommender.get_next_recommendation_track_id(tracks[0])
    print("Next recommendation given user played f:")
    print(next_rec_id)
    print("Recommendation path:")
    print(recommender._recommendations)