from nn import *
train_df = pd.read_csv(TRAIN, keep_default_na=False)
dev_df = pd.read_csv(DEV, keep_default_na=False)
wandb.login()

sweep_configuration = {
    "method" : "random",
    "metric" : {
        "goal" : "maximize",
        "name" : "f1_score"
    },
    "parameters" : {
        "batch_size" : {
            "values" : [50, 100]
        },
        "patience" : {
            "values" : [10]
        },
        "epochs" : {
            "values" : [300]
        },
        "optimizer" : {
            "values" : [
                "adam",
                "sgd",
                "adagrad",
                "rmsprop"
            ],
        },
        "learning_rate" : {
            "values" : [ 1e-2, 1e-3, 1e-4 ]
        },
        "spacy_model" : {
            "values" : [
                "en_core_web_lg",
            ]
        },
        "convo_range" : {
            "values" : [
                [1, 2, 1],
                [1, 3, 1],
                [2, 3, 1],
                [2, 4, 1],
            ]
        },
        "convo_filters" : {
            "values" : [ 14, 28, 56, 112]
        },
        "convo_activation" : {
            "values" : [
                "relu",
                "tanh",
                "sigmoid",
            ]
        },
        "dense_units" : {
            "values" : [ 14, 28, 56, 112]
        },
        "dense_activation" : {
            "values" : [
                "relu",
                "tanh",
                "sigmoid",
            ]
        },
    },
}

def objective(config):
    score = keras.metrics.F1Score(average = 'micro')(config.pred, config.y)
    return score

def main():
    with wandb.init() as run:

        config = wandb.config
        train_in, train_out = preprocess(train_df, inflate = True)
        dev_in, dev_out = preprocess(dev_df)
        model, fit = create_model(config = config)
        model.fit(
            x = train_in,
            y = train_out,
            validation_data = (dev_in, dev_out),
            **fit
        )
        pred = model.predict(dev_in)
        f1 = keras.metrics.F1Score(average = "micro")((pred > .5), dev_out).numpy()
        run.log({"f1_score" : f1})

sweep_id = wandb.sweep(sweep = sweep_configuration, project = "nn")

wandb.agent(sweep_id, function = main, count = 1000)
