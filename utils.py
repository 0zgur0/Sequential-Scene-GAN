import matplotlib.pyplot as plt
plt.switch_backend('agg')
import itertools
import numpy as np 
import torch

def show_result(num_epoch,test_images, show = False, save = False, path = 'result.png'):
    size_figure_grid = int(np.sqrt(test_images.size()[0]))
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(size_figure_grid*size_figure_grid):
        i = k // size_figure_grid
        j = k % size_figure_grid
        ax[i, j].cla()
        if test_images.size()[1] == 1:         
            ax[i, j].imshow((test_images[k].cpu().data.numpy().transpose(1, 2, 0).squeeze() + 1) / 2, cmap='gray')
        else:
            ax[i, j].imshow((test_images[k].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
            
        

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()
        

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def mse_loss(input, target):
    return torch.sum((input - target)**2) / input.data.nelement()