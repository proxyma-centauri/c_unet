from typing import List
import torchio as tio
import matplotlib.pyplot as plt
from typing import List


def plot_middle_slice(subject,
                      nb_of_classes: int,
                      cmap: str,
                      save_name: str,
                      classes_names: List[str] = None,
                      with_labels: bool = True):
    """
    Plots some slices of the image, labels and predictions, in coronal and sagital orientation.

    Args:
        - subject (torchio.Subject): Subject whose slices to plot
        - nb_of_classes (int): number of classes in the segmentation
        - cmap (str): Color map to use in plots
        - save_name (str): name of the file to save the plots to
        - classes_names (list): names of the classes of the segmentation
        - with_labels (bool): whether or not the subject has a field "label"
    """
    nb_rows = 2 + nb_of_classes

    # 2 figs for coronal and sagittal
    if with_labels:
        # Image, all labels, all predictions, then 2 more columns per class
        fig_sag, axarr_sag = plt.subplots(nb_rows, 2, figsize=(10, 20))
        fig_coro, axarr_coro = plt.subplots(nb_rows, 2, figsize=(10, 25))
    else:
        # No predictions printed in this case
        fig_sag, axarr_sag = plt.subplots(nb_rows, 1, figsize=(10, 20))
        fig_coro, axarr_coro = plt.subplots(nb_rows, 1, figsize=(10, 20))

    # Taking the middle slice
    slice_nb_sag = subject['image'][tio.DATA].shape[1] // 2
    slice_nb_coro = subject['image'][tio.DATA].shape[2] // 2

    # SAGITTAL # [:, :, slice_nb_sag, :] [:, slice_nb_sag, :]
    image_sag = subject['image'][tio.DATA][:, slice_nb_sag, :, :].squeeze()
    prediction_sag = subject['prediction'][tio.DATA].argmax(
        dim=0)[slice_nb_sag, :, :]

    to_print = {}

    ## Adding all images to print
    for index in range(nb_of_classes):
        prediction_sag_i = subject['prediction'][tio.DATA][index,
                                                           slice_nb_sag, :, :]
        to_print[index] = [prediction_sag_i]

        if with_labels:
            label_sag_i = subject['label'][tio.DATA][index, slice_nb_sag, :, :]
            to_print[index].append(label_sag_i)

    to_print[nb_of_classes] = [prediction_sag]
    if with_labels:
        label_sag = subject['label'][tio.DATA].argmax(
            dim=0)[slice_nb_sag, :, :]
        to_print[nb_of_classes].append(label_sag)

    ## Displaying images
    if with_labels:
        for key, items in to_print.items():
            axarr_sag[key, 0].imshow(items[0], cmap=cmap)
            axarr_sag[key, 1].imshow(items[1], cmap=cmap)

        axarr_sag[nb_rows - 1, 0].imshow(image_sag, cmap="gray")
        axarr_sag[nb_rows - 1, 1].imshow(image_sag, cmap="gray")

    else:
        for key, items in to_print.items():
            axarr_sag[key].imshow(items[0], cmap=cmap)

        axarr_sag[nb_rows - 1].imshow(image_sag, cmap="gray")

    # CORONAL
    image_coro = subject['image'][tio.DATA][:, :, slice_nb_coro, :].squeeze()
    prediction_coro = subject['prediction'][tio.DATA].argmax(
        dim=0)[:, slice_nb_coro, :]

    to_print = {}

    for index in range(nb_of_classes):
        prediction_coro_i = subject['prediction'][tio.DATA][index, :,
                                                            slice_nb_coro, :]
        to_print[index] = [prediction_coro_i]

        if with_labels:
            label_coro_i = subject['label'][tio.DATA][index, :,
                                                      slice_nb_coro, :]
            to_print[index].append(label_coro_i)

    to_print[nb_of_classes] = [prediction_coro]
    if with_labels:
        label_coro = subject['label'][tio.DATA].argmax(dim=0)[:,
                                                              slice_nb_coro, :]
        to_print[nb_of_classes].append(label_coro)

    # Displaying
    if with_labels:
        for key, items in to_print.items():
            axarr_coro[key, 0].imshow(items[0], cmap=cmap)
            axarr_coro[key, 1].imshow(items[1], cmap=cmap)

        axarr_coro[nb_rows - 1, 0].imshow(image_coro, cmap="gray")
        axarr_coro[nb_rows - 1, 1].imshow(image_coro, cmap="gray")

    else:
        for key, items in to_print.items():
            axarr_coro[key].imshow(items[0], cmap=cmap)

        axarr_coro[nb_rows - 1].imshow(image_coro, cmap="gray")

    # FORMATTING
    rows = classes_names + ['All classes', 'Image']

    if with_labels:
        cols = ['Pred', 'Label']
        for ax, col in zip(axarr_sag[0], cols):
            ax.set_title(col)
        for ax, col in zip(axarr_coro[0], cols):
            ax.set_title(col)

        for ax, row in zip(axarr_sag[:, 0], rows):
            ax.set_ylabel(row, rotation=90, size='large')
        for ax, row in zip(axarr_coro[:, 0], rows):
            ax.set_ylabel(row, rotation=90, size='large')
    else:
        for ax, row in zip(axarr_sag, rows):
            ax.set_ylabel(row, rotation=90, size='large')
        for ax, row in zip(axarr_coro, rows):
            ax.set_ylabel(row, rotation=90, size='large')

    fig_sag.tight_layout()
    fig_coro.tight_layout()

    fig_sag.savefig(f"{save_name}-SAG.png")
    fig_coro.savefig(f"{save_name}-CORO.png")

    plt.close(fig_sag)
    plt.close(fig_coro)
