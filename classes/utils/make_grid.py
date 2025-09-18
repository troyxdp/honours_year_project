import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as im

# Load images
im0_0 = im.imread('/home/troyxdp/Documents/University Work/HYP/HYP Source Code/Back End/PyTorch Statistics/Graphs/Experiment 1/Test A - Precision/pytorch_experiment_1_A_precision_scores.png')
im0_1 = im.imread('/home/troyxdp/Documents/University Work/HYP/HYP Source Code/Back End/PyTorch Statistics/Graphs/Experiment 1/Test B - Precision/pytorch_experiment_1_B_precision_scores.png')
im0_2 = im.imread('/home/troyxdp/Documents/University Work/HYP/HYP Source Code/Back End/PyTorch Statistics/Graphs/Experiment 1/Test C - Precision/pytorch_experiment_1_C_precision_scores.png')
im1_0 = im.imread('/home/troyxdp/Documents/University Work/HYP/HYP Source Code/Back End/PyTorch Statistics/Graphs/Experiment 2/Test A - Precision/pytorch_experiment_2_A_precision_scores.png')
im1_1 = im.imread('/home/troyxdp/Documents/University Work/HYP/HYP Source Code/Back End/PyTorch Statistics/Graphs/Experiment 2/Test B - Precision/pytorch_experiment_2_B_precision_scores.png')
im1_2 = im.imread('/home/troyxdp/Documents/University Work/HYP/HYP Source Code/Back End/PyTorch Statistics/Graphs/Experiment 2/Test C - Precision/pytorch_experiment_2_C_precision_scores.png')
im2_0 = im.imread('/home/troyxdp/Documents/University Work/HYP/HYP Source Code/Back End/PyTorch Statistics/Graphs/Experiment 3/Test A - Precision/pytorch_experiment_3_A_precision_scores.png')
im2_1 = im.imread('/home/troyxdp/Documents/University Work/HYP/HYP Source Code/Back End/PyTorch Statistics/Graphs/Experiment 3/Test B - Precision/pytorch_experiment_3_B_precision_scores.png')
im2_2 = im.imread('/home/troyxdp/Documents/University Work/HYP/HYP Source Code/Back End/PyTorch Statistics/Graphs/Experiment 3/Test C - Precision/pytorch_experiment_3_C_precision_scores.png')

f, axarr = plt.subplots(3, 3, figsize=(15, 15), constrained_layout=True)
axarr[0, 0].set_ylabel("Experiment 1", fontsize=14, labelpad=10)
axarr[0, 0].imshow(im0_0)
axarr[0, 1].imshow(im0_1)
axarr[0, 2].imshow(im0_2)
axarr[1, 0].set_ylabel("Experiment 2", fontsize=14, labelpad=10)
axarr[1, 0].imshow(im1_0)
axarr[1, 1].imshow(im1_1)
axarr[1, 2].imshow(im1_2)
axarr[2, 0].set_ylabel("Experiment 3", fontsize=14, labelpad=10)
axarr[2, 0].imshow(im2_0)
axarr[2, 1].imshow(im2_1)
axarr[2, 2].imshow(im2_2)

axarr[0, 0].set_title("Test A", fontsize=14, pad=10)
axarr[0, 1].set_title("Test B", fontsize=14, pad=10)
axarr[0, 2].set_title("Test C", fontsize=14, pad=10)

for i in range(3):
    for j in range(3):
        axarr[i, j].set_xticks([])
        axarr[i, j].set_yticks([])
        for spine in axarr[i, j].spines.values():
            spine.set_visible(False)

plt.show()