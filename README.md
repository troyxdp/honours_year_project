# Algorhythm Back End

This is all the code for my Honour's Year Project done at the University of Johannesburg in 2025. My mentor was Mr J Orfao. 

The system is a DJ recommendation system that recommends to a DJ what track to play next during their set given what song they are currently playing as well as the songs they have selected to play during the set. The system works by using an autoencoder's encoder section to generate a latent space embedding for each track using low-level features (chroma values) and high-level features (key/mode, BPM, danceability, energy, etc). This latent space is then traversed to find the shortest path between each of the embeddings for the tracks using either the Greedy Nearest Neighbour algorithm or the Held-Karp algorithm. However, the Held-Karp algorithm cannot be used for sets of songs selected by the DJ that are larger than 18 tracks (including the "seed track"/first track selected to play during the set) as using it for larger sets than this is not computationally feasible, especially for real-time usage. 

The project made use of a custom training dataset that was created by combining the Spotify Tracks Dataset with the Million Song Dataset, with the low-level features coming from the MSD and most of the high-level features from the Spotify Tracks Dataset. The test dataset was constructed using the Echo Nest Taste Profile Dataset, which is a dataset consisting of the plays data for a subset of songs from the MSD, and combining it with the custom training dataset. The test dataset consists of text files for each track that specify all other tracks with at least one common listener in the `song_links_dataset` folder; CSV files containing the user ID and play counts for each user that listened to a song in the `songs_with_links_play_data` folder, and the HDF5 files like those in the custom training dataset containing the song features for each of the songs in the test dataset. The code for the creation of these datasets is in the `hdf5_utils` folder for the training dataset and the `utils` folder for the test dataset.

The back end makes use of FastAPI and a PostgreSQL database for storing song data. The neural network used to generate embeddings that is used by the API is a handwritten neural network that uses Numpy as the basis. The code for the back end API is in the `backend.py` file, and the neural network code is in the `neural_network.py` file. The code used for training is in the `trainer.py` file, and training is done by executing the `train.py` file. There is also code for a PyTorch implementation of the network used by the API, and it is in `autoencoder.py`. The code used for training this PyTorch network is in `pytorch_trainer.py`, and training is done by executing the `train_pytorch_model.py` file. 

The code for extracting the encoder section of handwritten neural networks is also in the `train.py` file. Running the `train.py` file will first prompt you asking if you would like to train a model, and after training/saying you do not want to train the model, it prompts you asking if you would like to extract the encoder section. You will then have to enter the path to the weights file (which is a Pickle file) and the path to save the encoder to, and then it will extract the layers in the first half of the network.

To set up the project, run the following commands:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The second line will not work on Windows, and I cannot for the life of me remember what the Windows equivalent is and I can't be bothered to check. Google yourself and find out.

The weights files are also not included in this repo as they are too large, and neither are the datasets. The datasets may be available for download depending on what my mentor wants me to do.



# REFERENCES FOR THE CODE

The date of each of these is set to 30 October 2025 as I do not remember when I first accessed any of these sites
1. Adeli, E. et al. (2025) **Neural-networks-3: Parameter updates (SGD)**, **CS231n: Deep Learning for Computer Vision**. Available at: https://cs231n.github.io/neural-networks-3/#sgd (Accessed: 30 October 2025).
2. Aivean (2021) **Can you help explain this Held–Karp TSP pseudocode**, **Stack Overflow**, 1 July. Available at: https://stackoverflow.com/questions/69902373/can-you-help-explain-this-held-karp-tsp-pseudocode (Accessed: 30 October 2025).
3. Anand, H. (2021) **L1 & L2 regularization — Adding penalties to the loss function**, **Medium**, 16 December. Available at: https://neuralthreads.medium.com/l1-l2-regularization-adding-penalties-to-the-loss-function-b5c330d30b3f (Accessed: 30 October 2025).
4. Bertin-Mahieux, T. (2011) **Million Song Dataset (MSongsDB)**, **GitHub Repository**. Available at: https://github.com/tbertinmahieux/MSongsDB/ (Accessed: 30 October 2025).
5. Kashyap, P. (2024) **Mastering Weight Initialization in Neural Networks: A Beginner’s Guide**, **Medium**, 2 November. Available at: https://medium.com/@piyushkashyap045/mastering-weight-initialization-in-neural-networks-a-beginners-guide-6066403140e9 (Accessed: 30 October 2025).
6. Nielsen, M.A. (2015) **Neural Networks and Deep Learning**. **Determination Press**. Available at: http://neuralnetworksanddeeplearning.com/chap2.html (Accessed: 30 October 2025).
7. Tushar, R. (2015) **Traveling Salesman Problem Dynamic Programming Held–Karp** [video], **YouTube**, 7 December. Available at: https://www.youtube.com/watch?v=-JjA4BLQyqE (Accessed: 30 October 2025).
8. Wikipedia contributors (2025) **Held–Karp algorithm**, **Wikipedia: The Free Encyclopedia**. Available at: https://en.wikipedia.org/wiki/Held%E2%80%93Karp_algorithm (Accessed: 30 October 2025).



# LINKS THAT WERE USED 

This is basically just the links from the references to make it easier for you to view them
- https://cs231n.github.io/neural-networks-3/#sgd
- https://stackoverflow.com/questions/69902373/can-you-help-explain-this-held-karp-tsp-pseudocode
- https://neuralthreads.medium.com/l1-l2-regularization-adding-penalties-to-the-loss-function-b5c330d30b3f
- https://github.com/tbertinmahieux/MSongsDB/
- https://medium.com/@piyushkashyap045/mastering-weight-initialization-in-neural-networks-a-beginners-guide-6066403140e9
- http://neuralnetworksanddeeplearning.com/chap2.html
- https://www.youtube.com/watch?v=-JjA4BLQyqE
- https://en.wikipedia.org/wiki/Held%E2%80%93Karp_algorithm