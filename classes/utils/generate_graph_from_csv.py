import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Plot data path and plot labels
    file_path = '/home/troyxdp/Documents/University Work/HYP/HYP Source Code/Back End/pytorch_networks/experiment_1/train_stats.csv'
    xlabel = 'Training Loss'
    ylabel = 'Epoch Number'
    plot_title = 'Training Loss Per Epoch'

    # Get data
    x_values = []
    y_values = []
    data = pd.read_csv(file_path)
    for row in data.itertuples():
        x_values.append(row.epoch_number - 1)
        y_values.append(row.loss)
    
    # Plot data
    plt.plot(x_values, y_values)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(plot_title)
    plt.show()