from datetime import datetime
import os
import torch
import torchio as tio
import pytorch_lightning as pl
import numpy as np
import nibabel as nib
import pymia.evaluation.evaluator as eval_
import pymia.evaluation.metric as metric
import pymia.evaluation.writer as writer

from pytorch_lightning import loggers as pl_loggers
from pathlib import Path
from decouple import config

from c_unet.training.datamodule import DataModule
from c_unet.architectures.unet import Unet
from c_unet.training.tverskyLosses import FocalTversky_loss
from c_unet.training.lightningUnet import LightningUnet
from c_unet.utils.plots.plot import plot_middle_slice
from c_unet.utils.logging.logging import configure_and_return_logger


def main(logger, args):
    # CONFIG
    logger.info(f"CONFIGURATION \n\n {args}")
    print("\nYou are running with the following configuration:\n")
    print(args)
    print("\n --- \n")

    # DATA
    data = DataModule(args.get("PATH_TO_DATA"),
                      subset_name=args.get("SUBSET_NAME"),
                      batch_size=args.get("BATCH_SIZE"),
                      num_workers=args.get("NUM_WORKERS"),
                      test_has_labels=args.get("TEST_HAS_LABELS"),
                      seed=args.get("SEED"))

    data.prepare_data()
    data.setup()
    logger.info("Data set up\n")

    print('Training:  ', len(data.train_set))
    print('Validation: ', len(data.val_set))
    print('Test:      ', len(data.test_set))
    print("\n --- \n")

    # MODEL
    if args.get("GROUP") is not None:
        model = Unet(args.get("GROUP"),
                     args.get("GROUP_DIM"),
                     args.get("IN_CHANNELS"),
                     args.get("OUT_CHANNELS"),
                     final_activation=args.get("FINAL_ACTIVATION"),
                     nonlinearity=args.get("NONLIN"),
                     normalization=args.get("NORMALIZATION"),
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
                     normalization=args.get("NORMALIZATION"),
                     divider=args.get("DIVIDER"),
                     model_depth=args.get("MODEL_DEPTH"),
                     dropout=args.get("DROPOUT"))

    # LIGHTNING
    loss = FocalTversky_loss({"apply_nonlin": None})

    log_name = f"{args.get('LOG_NAME')}-{args.get('MODEL_DEPTH')}-{args.get('LEARNING_RATE')}-{args.get('GRADIENT_CLIP')}"
    tb_logger = pl_loggers.TensorBoardLogger(args.get("LOGS_DIR"),
                                             name=log_name,
                                             default_hp_metric=False)

    callbacks = [pl.callbacks.ModelCheckpoint(monitor='val_loss')]

    if args.get("EARLY_STOPPING") is not None and args.get("EARLY_STOPPING"):
        early_stopping = pl.callbacks.early_stopping.EarlyStopping(
            monitor='val_loss')
        callbacks.append(early_stopping)

    # LOAD FROM CHECKPOINTS
    if args.get("LOAD_FROM_CHECKPOINTS"):
        path_checkpoint = os.path.abspath(args.get("CHECKPOINTS_PATH"))
        lightning_model = LightningUnet.load_from_checkpoint(path_checkpoint)
        logger.info("Logged from CHECKPOINTS\n")
    else:
        lightning_model = LightningUnet(
            loss,
            torch.optim.AdamW,
            model,
            learning_rate=args.get("LEARNING_RATE"),
            gradients_histograms=args.get("HISTOGRAMS"))
        logger.info("Created new model\n")

    # SUMMARY OF MODEL
    print(lightning_model.summarize())

    # TRAINING
    if args.get("SHOULD_TRAIN"):
        trainer = pl.Trainer(
            gpus=args.get("GPUS"),
            precision=args.get("PRECISION"),
            log_gpu_memory=True,
            max_epochs=args.get("MAX_EPOCHS"),
            log_every_n_steps=args.get("LOG_STEPS"),
            logger=tb_logger,
            callbacks=callbacks,
            benchmark=True,
            gradient_clip_val=args.get("GRADIENT_CLIP"),
            gradient_clip_algorithm='value',
            stochastic_weight_avg=True,
            progress_bar_refresh_rate=2,
        )

        start = datetime.now()
        print('\nTraining started at', start)
        logger.info(f"Training started at {start}")
        trainer.fit(model=lightning_model.cuda(), datamodule=data)
        print('\nTraining duration:', datetime.now() - start)
        logger.info(f"Training duration: {datetime.now() - start}")

    else:
        print("\nTraining skipped")

    # MEASURES
    metrics = [
        metric.DiceCoefficient(),
        metric.HausdorffDistance(percentile=100),
        metric.VolumeSimilarity(),
    ]
    labels = {i: name for i, name in enumerate(args.get("CLASSES_NAME"))}

    evaluator = eval_.SegmentationEvaluator(metrics, labels)

    # PREDICTIONS
    lightning_model.eval()

    def make_predictions_over_subject_set(subject,
                                          list_of_predictions,
                                          dataloader_type="train"):
        input = subject['image'][tio.DATA].to(lightning_model.device)
        filename = subject['label']['filename']

        # Make sure there is a channel and a group dimension when needed
        input = input.unsqueeze(0)
        if args.get("GROUP"):
            input = input.unsqueeze(1)

        prediction_for_subject = lightning_model.unet(input)
        subject.add_image(
            tio.LabelMap(tensor=prediction_for_subject[0, :, :, :, :]),
            'prediction')

        list_of_predictions[dataloader_type].append((subject, filename))

    list_of_predictions = {"train": [], "val": [], "test": []}
    datasets = {
        #"train": data.train_set,
        "test": data.test_set,
        #"val": data.val_set
    }

    with torch.no_grad():
        for type_loader, subjects_dataset in datasets.items():
            print(f" --- PREDICTING {type_loader} --- ")
            for subject in subjects_dataset:
                make_predictions_over_subject_set(subject,
                                                  list_of_predictions,
                                                  dataloader_type=type_loader)

    logger.info("Finished PREDICTING\n")
    # EVALUATING
    Path(f"results/{log_name}").mkdir(parents=True, exist_ok=True)

    for type_predictions, list_of_subjects in list_of_predictions.items():
        print(f" --- EVALUATING {type_predictions} --- ")
        logger.info(f" --- EVALUATING {type_predictions} --- ")

        # Making sure that we only try to evaluate on test when there are test labels
        should_evaluate_and_plot_normaly = (type_predictions != "test") or (
            args.get("TEST_HAS_LABELS"))

        for (subject, filename) in list_of_subjects:
            # for subject, filename in zip(subjects_data, filenames):
            subject_id = f"{type_predictions}-{filename}"

            # SAVING THE SEGMENTATION
            folder_name = 'labelsTs' if type_predictions == "test" else "labelsTr"

            header = nib.load(
                f'{args.get("PATH_TO_DATA")}/{folder_name}/{filename}').header

            inverted_subject = subject.apply_inverse_transform()
            prediction_to_save = inverted_subject['prediction'][
                tio.DATA].argmax(dim=0)
            print(prediction_to_save.shape)
            print(prediction_to_save.unique())

            affine = inverted_subject['image'][tio.AFFINE]

            saved_prediction = nib.Nifti1Image(prediction_to_save.numpy(),
                                               affine=affine,
                                               header=header)
            nib.save(saved_prediction, f"results/{log_name}/{subject_id}")

            # EVALUATION
            if should_evaluate_and_plot_normaly:
                sub_label = subject['label'][tio.DATA].argmax(dim=0).numpy()
                sub_prediction = subject['prediction'][tio.DATA].argmax(
                    dim=0).numpy()

                evaluator.evaluate(sub_prediction, sub_label, subject_id)

            # EXAMPLE SLICE PLOTTING
            plot_middle_slice(subject,
                              nb_of_classes=len(args.get("CLASSES_NAME")),
                              cmap=args.get("CMAP"),
                              save_name=f"results/{log_name}/{subject_id}",
                              classes_names=args.get("CLASSES_NAME"),
                              with_labels=should_evaluate_and_plot_normaly)

    logger.info("Finished EVALUATING\n")
    # SAVING METRICS
    functions = {'MEAN': np.mean, 'STD': np.std}
    writer.ConsoleStatisticsWriter(functions=functions).write(
        evaluator.results)

    writer.CSVWriter(f"results/{log_name}/metrics_report.csv").write(
        evaluator.results)
    writer.CSVStatisticsWriter(
        f"results/{log_name}/metrics_report_summary.csv",
        functions=functions).write(evaluator.results)
    logger.info("Saved metrics to files")


if __name__ == "__main__":
    # LOGGER
    logger = configure_and_return_logger(
        'c_unet/utils/logging/loggingConfig.yml')

    # ARGS
    args = {}

    args["LOAD_FROM_CHECKPOINTS"] = config("LOAD_FROM_CHECKPOINTS",
                                           default=False,
                                           cast=bool)
    args["CHECKPOINTS_PATH"] = config("CHECKPOINTS_PATH",
                                      default=None,
                                      cast=str)
    args["SHOULD_TRAIN"] = config("SHOULD_TRAIN", default=True, cast=bool)

    args["CLASSES_NAME"] = config("CLASSES_NAME").split(", ")

    args["PATH_TO_DATA"] = config("PATH_TO_DATA")
    args["SUBSET_NAME"] = config("SUBSET_NAME")
    args["BATCH_SIZE"] = config("BATCH_SIZE", cast=int)
    args["NUM_WORKERS"] = config("NUM_WORKERS", cast=int)
    args["TEST_HAS_LABELS"] = config("TEST_HAS_LABELS",
                                     default=False,
                                     cast=bool)
    args["SEED"] = config("SEED", default=1, cast=int)

    args["GROUP"] = config("GROUP", default=None)
    args["GROUP_DIM"] = config("GROUP_DIM", cast=int)
    args["IN_CHANNELS"] = 1
    args["OUT_CHANNELS"] = config("OUT_CHANNELS", cast=int)
    args["FINAL_ACTIVATION"] = config("FINAL_ACTIVATION", default="softmax")
    args["NONLIN"] = config("NONLIN", default="leaky-relu")
    args["NORMALIZATION"] = config("NORMALIZATION", default="bn")
    args["DIVIDER"] = config("DIVIDER", cast=int)
    args["MODEL_DEPTH"] = config("MODEL_DEPTH", cast=int)
    args["DROPOUT"] = config("DROPOUT", cast=float)

    args["LOGS_DIR"] = config("LOGS_DIR")
    args["LOG_NAME"] = config("LOG_NAME")

    args["EARLY_STOPPING"] = config("EARLY_STOPPING", default=False, cast=bool)

    args["LEARNING_RATE"] = config("LEARNING_RATE", default=1e-3, cast=float)
    args["HISTOGRAMS"] = config("HISTOGRAMS", default=False, cast=bool)

    args["GPUS"] = [config("GPUS", default=1, cast=int)]
    args["PRECISION"] = config("PRECISION", default=16, cast=int)

    args["MAX_EPOCHS"] = config("MAX_EPOCHS", default=30, cast=int)
    args["LOG_STEPS"] = config("LOG_STEPS", default=5, cast=int)

    args["GRADIENT_CLIP"] = config("GRADIENT_CLIP", default=0.5, cast=float)
    args["CMAP"] = config("CMAP", default="Oranges")

    main(logger, args)
