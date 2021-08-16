import torchio as tio
import matplotlib.pyplot as plt


def plot_middle_slice(subject,
                      nb_of_classes,
                      cmap,
                      save_name,
                      classes_names=None,
                      with_labels=True):
    """
    Plots some slices of the image, labels and predictions
    """
    if with_labels:
        # Image, all labels, all predictions, then 2 more columns per class
        nb_columns = 3 + 2 * nb_of_classes
    else:
        # No predictions printed in this case
        nb_columns = 2 + nb_of_classes

    # 2 lines for coronal and sagittal
    fig_sag, axarr_sag = plt.subplots(1, nb_columns, figsize=(20, 10))
    fig_coro, axarr_coro = plt.subplots(1, nb_columns, figsize=(20, 10))

    # Taking the middle slice
    slice_nb_sag = subject['image'][tio.DATA].shape[1] // 2
    slice_nb_coro = subject['image'][tio.DATA].shape[2] // 2

    # SAGITTAL # [:, :, slice_nb_sag, :] [:, slice_nb_sag, :]
    image_sag = subject['image'][tio.DATA][:, slice_nb_sag, :, :].squeeze()
    prediction_sag = subject['prediction'][tio.DATA].argmax(
        dim=0)[slice_nb_sag, :, :]
    to_print = [prediction_sag]

    if with_labels:
        label_sag = subject['label'][tio.DATA].argmax(
            dim=0)[slice_nb_sag, :, :]
        to_print.append(label_sag)

    for index in range(nb_of_classes):
        prediction_sag_i = subject['prediction'][tio.DATA][index,
                                                           slice_nb_sag, :, :]
        to_print.append(prediction_sag_i)

        if with_labels:
            label_sag_i = subject['label'][tio.DATA][index, slice_nb_sag, :, :]
            to_print.append(label_sag_i)

    axarr_sag[0, 0].imshow(image_sag, cmap="gray")

    for index in range(0, nb_columns - 1):
        axarr_sag[0, index + 1].imshow(to_print[index], cmap=cmap)

    # CORONAL
    image_coro = subject['image'][tio.DATA][:, :, slice_nb_coro, :].squeeze()
    prediction_coro = subject['prediction'][tio.DATA].argmax(
        dim=0)[:, slice_nb_coro, :]
    to_print = [prediction_coro]
    if with_labels:
        label_coro = subject['label'][tio.DATA].argmax(dim=0)[:,
                                                              slice_nb_coro, :]
        to_print.append(label_coro)

    for index in range(nb_of_classes):
        prediction_coro_i = subject['prediction'][tio.DATA][index, :,
                                                            slice_nb_coro, :]
        to_print.append(prediction_coro_i)

        if with_labels:
            label_coro_i = subject['label'][tio.DATA][index, :,
                                                      slice_nb_coro, :]
            to_print.append(label_coro_i)

    axarr_coro[0, 0].imshow(image_coro, cmap="gray")
    for index in range(0, nb_columns - 1):
        axarr_coro[0, index + 1].imshow(to_print[index], cmap=cmap)

    # Formatting
    if classes_names:
        cols_names = classes_names
    else:
        cols_names = range(nb_of_classes)

    label_cols = ['Label {}'.format(row) for row in cols_names]
    pred_cols = ['Pred {}'.format(row) for row in cols_names]
    all_cols = list(zip(pred_cols, label_cols))

    if with_labels:
        cols = ['Image', 'Pred', 'Label'
                ] + [elt for sublist in all_cols for elt in sublist]

    else:
        cols = ['Image', 'Label'] + label_cols

    for ax, col in zip(axarr_sag[0], cols):
        ax.set_title(col)
    for ax, col in zip(axarr_coro[0], cols):
        ax.set_title(col)

    fig_sag.tight_layout()
    fig_coro.tight_layout()

    fig_sag.savefig(f"{save_name}-SAG.png")
    fig_coro.savefig(f"{save_name}-CORO.png")

    plt.close(fig_sag)
    plt.close(fig_coro)
