# HYP Backend
Welcome to the back end code for my HYP! This contains all of my handwritten neural network code; training code, and back end code for my API.

To download all of the necessary libraries, run the following command:
```
pip install -r requirements.txt
```

To train a model, run the following in the terminal:
```
python train.py
```
To adjust the network architecture or hyperparameters you will have to do so in the code for the file. You will also need to make sure that you have the properly formatted dataset.

To run the backend, run the following in the terminal:
```
python backend.py
```
You will need to have a properly set up PostgreSQL database with PG Vector installed. You can work out how to do so by looking inside the `add_data_to_database.py` file. 

To run a demo of the system, you can run the `demo.ipynb` notebook. To do so requires having Jupyter Notebook installed. I had it set up on VS Code and ran the notebook from there. 
