import argparse
import numpy as np
import pandas as pd
import keras
import spacy
from pathlib import Path
from typing import Tuple
from dataclasses import dataclass, asdict
import wandb
# from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

keras.utils.set_random_seed(42)

@dataclass
class WandbConfig:
    batch_size = 50
    patience = 10
    epochs = 300
    optimizer = "rmsprop"
    learning_rate = .001
    spacy_model = "en_core_web_lg"
    convo_range = [2,3,1]
    convo_filters = 56
    convo_activation = "sigmoid"
    dense_units = 112
    dense_activation = "sigmoid"

config = WandbConfig()

TRAIN = Path() / "train.csv"
DEV = Path() / "dev.csv"
TEST = Path() / "test_in.csv"
MODEL = Path() / "checkmeout.keras"

def getspacy():
    try:
        return spacy.load(config.spacy_model)
    except OSError:
        spacy.cli.download(config.spacy_model)
        return spacy.load(config.spacy_model)

NLP = getspacy()

emotions = ['admiration', 'amusement', 'anger', 'annoyance', 'approval',
            'confusion', 'curiosity', 'disapproval', 'gratitude', 'joy',
            'love', 'optimism', 'sadness', 'surprise']

def labels_to_labels(series: pd.Series) -> np.ndarray:
    """
    convert strings to vector of binary labels
    """
    return (
        series
        .str.get_dummies(sep = ' ')
        .reindex(columns = emotions)
        .values
    )

def launch_into_space(series: pd.Series) -> np.ndarray:
    """
    convert text to vectors, using spacy
    """
    variable_length_examples = (
        series
        .map(NLP)
        .apply(
            lambda x: np.array([
                    w.vector for sent in x.sents for w in sent
                    if not w.is_punct
                    and not w.is_oov
                    and not w.is_stop
                ])
        )
    )
    # Motivated by a need for batches to be uniform
    m = max(len(e) for e in variable_length_examples)
    uniform_length = keras.utils.pad_sequences(
        sequences = variable_length_examples,
        maxlen = m,
        dtype = 'float32', padding = "post"
    )
    # print(uniform_length.shape)
    return uniform_length


def preprocess(df, inflate = False):
    """
    Process both columns of input files
    """
    labels = labels_to_labels(df.loc[:,'labels'])
    embeddings = launch_into_space(df.loc[:,'text'])
    # trying to mitigate overabudance of unlabeled texts by creating data
    if inflate:
        tinyrand = lambda: np.random.uniform(-1,1) * 1e-12
        embeddings = np.concat([
            embeddings,
            embeddings[labels.sum(axis = 1) > 0] + tinyrand(),
            embeddings[labels.sum(axis = 1) > 0] + tinyrand(),
        ], axis = 0)
        labels = np.concat([
            labels,
            labels[labels.sum(axis = 1) > 0],
            labels[labels.sum(axis = 1) > 0],
        ], axis = 0)
    return embeddings, labels


def create_model(
    config = config,
    n_outputs = len(emotions)
 ) -> Tuple[keras.Model, dict]:
    """
    Build that model
    """
    inputs = keras.Input(shape = (None, 300))

    convo = [
        keras.layers.Conv1D(
            filters = config.convo_filters,
            kernel_size = i,
            strides = 1,
            activation = config.convo_activation
        )(inputs)
        for i in range(1, 2, 1)
    ]
    convo = [
        keras.layers.GlobalMaxPooling1D()(w) for w in convo
    ]
    outputs = keras.layers.Concatenate()(convo)

    outputs = keras.layers.Dense(
        units = config.dense_units,
        activation = config.dense_activation
    )(outputs)

    outputs = keras.layers.Dense(
        units = n_outputs,
        activation = keras.activations.sigmoid
    )(outputs)
    model = keras.Model(
        inputs, outputs
    )
    optimizer = keras.optimizers.get(config.optimizer)
    optimizer.learning_rate = config.learning_rate
    model.compile(
        optimizer = optimizer,
        loss = keras.losses.BinaryFocalCrossentropy(),
    )
    fit = {
        'callbacks' : [
            keras.callbacks.EarlyStopping(
                monitor = "val_loss",
                patience = config.patience,
                restore_best_weights = True, # how did I miss this before
            ),
        ],
        'batch_size' : config.batch_size,
        'epochs' : config.epochs,
    }
    return model, fit


def train(train_path = TRAIN, dev_path = DEV, model_path = MODEL, config = config):
    train_df = pd.read_csv(train_path, keep_default_na=False)
    dev_df = pd.read_csv(dev_path, keep_default_na=False)

    train_in, train_out = preprocess(train_df, inflate = True)
    dev_in, dev_out = preprocess(dev_df)

    model, fit = create_model(config)
    model.fit(
        x = train_in,
        y = train_out,
        validation_data = (dev_in, dev_out),
        **fit
    )

    keras.models.save_model(model, str(model_path))


def predict(input_path, model_path = MODEL):
    df = pd.read_csv(input_path, keep_default_na=False)
    data_in = launch_into_space(df['text'])
    if 'labels' in df.columns:
        data_out = labels_to_labels(df['labels'])

    model = keras.models.load_model(str(model_path))
    pred = model.predict(data_in)

    df['labels'] = pd.Series([
        " ".join(
            emotions[i]
            for i, l in enumerate(labels)
            if l > .5
        )
        for labels in pred
    ])
    try:
        print(
            "F1 SCore:",
            keras.metrics.F1Score(average = "micro")(pred > .5, data_out).numpy(),
            sep = "\n", end = "\n" * 2
        )
    except NameError:
        ...

    df.to_csv("submission.csv", index=False)
    df.to_csv("submission.zip", index=False, compression=dict(
        method='zip', archive_name=f'submission.csv'))

if __name__ == "__main__":
    ...
    # parse the command line arguments
    # parser = argparse.ArgumentParser()
    # subparsers = parser.add_subparsers(dest="command")
    # train_parser = subparsers.add_parser("train")
    # train_parser.add_argument("train_path")
    # train_parser.add_argument("dev_path")
    # train_parser.add_argument("model_path")
    # predict_parser = subparsers.add_parser("predict")
    # predict_parser.add_argument("model_path")
    # predict_parser.add_argument("input_path")
    # args = parser.parse_args()
    # kwargs = vars(args)
    # globals()[kwargs.pop('command')](**kwargs)
