import torchio as tio
import matplotlib.pyplot as plt


def plot_middle_slice(subject, cmap, save_name):
    fig, axarr = plt.subplots(2, 11, figsize=(15, 8))

    # Taking the middle slice
    slice_nb = subject['image'][tio.DATA].shape()[1]

    # Sagital
    image_sag = subject['image'][tio.DATA][:, slice_nb, :, :].squeeze()
    label_sag = subject['label'][tio.DATA].argmax(dim=0)[slice_nb, :, :]
    prediction_sag = subject['prediction'][tio.DATA].argmax(
        dim=0)[slice_nb, :, :]

    to_print = [label_sag, prediction_sag]

    for index in range(4):
        label_sag_i = subject['label'][tio.DATA][index, slice_nb, :, :]
        prediction_sag_i = subject['prediction'][tio.DATA][index,
                                                           slice_nb, :, :]
        to_print.append(label_sag_i)
        to_print.append(prediction_sag_i)

    axarr[0, 0].imshow(image_sag, cmap="gray")

    for index in range(0, 10, 2):
        axarr[0, index + 1].imshow(to_print[index], cmap=cmap)
        axarr[0, index + 2].imshow(to_print[index + 1], cmap=cmap)

    # Coronal
    image_coro = subject['image'][tio.DATA][:, :, slice_nb, :].squeeze()
    label_coro = subject['label'][tio.DATA].argmax(dim=0)[:, slice_nb, :]
    prediction_coro = subject['prediction'][tio.DATA].argmax(
        dim=0)[:, slice_nb, :]

    to_print = [label_coro, prediction_coro]

    for index in range(4):
        label_coro_i = subject['label'][tio.DATA][index, :, slice_nb, :]
        prediction_coro_i = subject['prediction'][tio.DATA][index, :,
                                                            slice_nb, :]
        to_print.append(label_coro_i)
        to_print.append(prediction_coro_i)

    axarr[1, 0].imshow(image_coro, cmap="gray")
    for index in range(0, 10, 2):
        axarr[1, index + 1].imshow(to_print[index], cmap=cmap)
        axarr[1, index + 2].imshow(to_print[index + 1], cmap=cmap)

    # Formatting
    cols = [
        '{}'.format(row) for row in [
            'Image', 'Label', 'Pred', 'Label 0', 'Pred 0', 'Label 1', 'Pred 1',
            'Label 2', 'Pred 2', 'Label 3', 'Pred 3'
        ]
    ]

    for ax, col in zip(axarr[0], cols):
        ax.set_title(col)

    fig.tight_layout()
    plt.savefig(save_name)