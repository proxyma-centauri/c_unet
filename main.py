import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
import pytorch_lightning as pl

from pytorch_lightning import loggers as pl_loggers
from decouple import config

from c_unet.training.datamodule import DataModule
from c_unet.architectures.unet import Unet
from c_unet.training.tverskyLosses import FocalTversky_loss
from c_unet.training.lightningUnet import LightningUnet
from c_unet.utils.plots.plot import plot_middle_slice

# TODO : subject name for figs
# TODO : add default values for config
# TODO : save weights


def main(args):
    # DATA
    data = DataModule("PATH_TO_DATA",
                      subset_name="SUBSET_NAME",
                      batch_size="BATCH_SIZE",
                      num_workers="NUM_WORKERS")

    data.prepare_data()
    data.setup()

    print('Training:  ', len(data.train_set))
    print('Validation: ', len(data.val_set))
    print('Test:      ', len(data.test_set))

    # MODEL
    if args.get("GROUP") is not None:
        model = Unet(args.get("GROUP"),
                     args.get("GROUP_DIM"),
                     args.get("IN_CHANNELS"),
                     args.get("OUT_CHANNELS"),
                     final_activation=args.get("FINAL_ACTIVATION"),
                     nonlinearity=args.get("NONLIN"),
                     divider=args.get("DIVIDER"),
                     model_depth=args.get("MODEL_DEPTH"),
                     dropout=args.get("DROPOUT"))
    else:
        model = Unet(None,
                     1,
                     args.get("IN_CHANNELS"),
                     args.get("OUT_CHANNELS"),
                     final_activation=args.get("FINAL_ACTIVATION"),
                     nonlinearity=args.get("NONLIN"),
                     model_depth=args.get("MODEL_DEPTH"),
                     dropout=args.get("DROPOUT"))

    # LIGHTNING
    loss = FocalTversky_loss({"apply_nonlin": None})

    tb_logger = pl_loggers.TensorBoardLogger(args.get("LOGS_DIR"),
                                             name=args.get("LOG_NAME"),
                                             default_hp_metric=False)
    callbacks = []

    if args.get("EARLY_STOPPING") is not None:
        early_stopping = pl.callbacks.early_stopping.EarlyStopping(
            monitor='val_loss')
        callbacks.append(early_stopping)

    lightning_model = LightningUnet(
        loss,
        torch.optim.AdamW,
        model,
        learning_rate=args.get("LEARNING_RATE"),
        gradients_histograms=args.get("HISTOGRAMS"))

    trainer = pl.Trainer(gpus=args.get("GPUS"),
                         precision=args.get("PRECISION"),
                         log_gpu_memory=True,
                         max_epochs=args.get("MAX_EPOCHS"),
                         log_every_n_steps=args.get("LOG_STEPS"),
                         logger=tb_logger,
                         callbacks=callbacks,
                         benchmark=True,
                         gradient_clip_val=0.5)

    # Training
    start = datetime.now()
    print('Training started at', start)
    # torch.autograd.set_detect_anomaly(True)
    trainer.fit(model=lightning_model.cuda(), datamodule=data)
    print('Training duration:', datetime.now() - start)

    # PREDICTIONS
    def make_predictions_over_dataloader(batch,
                                         list_of_predictions,
                                         dataloader_type="train"):
        inputs = batch['image'][tio.DATA].to(lightning_model.device)

        if args.get("GROUP"):
            inputs = inputs.unsqueeze(1)
        predictions = lightning_model.unet(inputs).cpu()

        batch_subjects = tio.utils.get_subjects_from_batch(batch)
        tio.utils.add_images_from_batch(batch_subjects, predictions,
                                        tio.LabelMap)
        list_of_predictions[dataloader_type].append(batch_subjects)

    list_of_predictions = {"train": [], "val": [], "test": []}

    with torch.no_grad():
        for batch in data.train_dataloader():
            make_predictions_over_dataloader(batch,
                                             list_of_predictions,
                                             dataloader_type="train")
            for subject in batch:
                plot_middle_slice(subject, args.get("CMAP"),
                                  f"train-{subject.name}")

        for batch in data.val_dataloader():
            make_predictions_over_dataloader(batch,
                                             list_of_predictions,
                                             dataloader_type="val")

            for subject in batch:
                plot_middle_slice(subject, args.get("CMAP"),
                                  f"val-{subject.name}")

        for batch in data.test_dataloader():
            make_predictions_over_dataloader(batch,
                                             list_of_predictions,
                                             dataloader_type="test")

            for subject in batch:
                plot_middle_slice(subject, args.get("CMAP"),
                                  f"test-{subject.name}")


if __name__ == "__main__":
    args = {}

    args["PATH_TO_DATA"] = config("PATH_TO_DATA")
    args["SUBSET_NAME"] = config("SUBSET_NAME")
    args["BATCH_SIZE"] = config("BATCH_SIZE")
    args["NUM_WORKERS"] = config("NUM_WORKERS")

    args["GROUP"] = config("GROUP")
    args["GROUP_DIM"] = config("GROUP_DIM")
    args["IN_CHANNELS"] = 1
    args["OUT_CHANNELS"] = config("OUT_CHANNELS")
    args["FINAL_ACTIVATION"] = config("FINAL_ACTIVATION")
    args["NONLIN"] = config("NONLIN")
    args["DIVIDER"] = config("DIVIDER")
    args["MODEL_DEPTH"] = config("MODEL_DEPTH")
    args["DROPOUT"] = config("DROPOUT")

    args["LOGS_DIR"] = config("LOGS_DIR")
    args["LOG_NAME"] = config("LOG_NAME")

    args["EARLY_STOPPING"] = config("EARLY_STOPPING")

    args["LEARNING_RATE"] = config("LEARNING_RATE")
    args["HISTOGRAMS"] = config("HISTOGRAMS")

    args["GPUS"] = config("GPUS")
    args["PRECISION"] = config("PRECISION")

    args["MAX_EPOCHS"] = config("MAX_EPOCHS")
    args["LOG_STEPS"] = config("LOG_STEPS")

    args["GRADIENT_CLIP"] = config("GRADIENT_CLIP")
    args["CMAP"] = config("CMAP")

    main(args)
