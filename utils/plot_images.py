import matplotlib.pyplot as plt


def plot_images(images, grid_size=(10, 10)):
    fig, axes = plt.subplots(
        nrows=grid_size[0], ncols=grid_size[1], figsize=(20, 20))
    axes = axes.flatten()

    for ax, img in zip(axes, images):
        ax.imshow(img)
        ax.axis("off")

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
