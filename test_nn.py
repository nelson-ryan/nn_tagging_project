import pytest
from nn import *
train_df = pd.read_csv(TRAIN, keep_default_na = False)
dev_df = pd.read_csv(DEV, keep_default_na = False)
test_df = pd.read_csv(TEST, keep_default_na = False)

@pytest.fixture(autouse=True)
def set_seed():
    keras.utils.set_random_seed(42)

def test_labels():
    labels = labels_to_labels(train_df['labels'])
    assert labels.shape[0] == len(labels)
    assert labels.shape[1] == len(emotions)

def test_preprocess():
    embeddings, _ = preprocess(train_df)
    assert len(embeddings) == 2022
    assert all(x.shape[1] == 300 for x in embeddings)

def test_create_model():
    model, _ = create_model()

def test_train():
    train_in, train_out = preprocess(train_df)
    dev_in, dev_out = preprocess(dev_df)
    model, fit = create_model()
    model.fit(x = train_in, y = train_out, **fit)


def test_predict():

    train_in, train_out = preprocess(train_df, inflate = True)
    dev_in, dev_out = preprocess(dev_df)
    model, fit = create_model(config)
    model.fit(
        x = train_in,
        y = train_out,
        validation_data = (dev_in, dev_out),
        **fit
    )
    pred = model.predict(dev_in)

    metric = keras.metrics.F1Score(average = "micro")
    print(metric(pred, dev_out).numpy())
    print(keras.metrics.F1Score(average = "micro")(pred > .5, dev_out).numpy())
    pd.DataFrame(keras.metrics.F1Score()(pred > .5, dev_out), index = emotions)
    for threshold in range(1, 8, 1):
        t = threshold / 10
        print(t, keras.metrics.F1Score(average = "micro")(pred > t, dev_out).numpy())

        # tp = (p * dev_out).sum(axis = 0)
        # fp = (p * (1 - dev_out)).sum(axis = 0)
        # fn = (dev_out * (1 - p)).sum(axis = 0)
        # precision = tp / (tp + fp)
        # recall = tp / (tp + fn)
        # f1 = 2 * precision * recall / (precision + recall)
        # print(pd.DataFrame({
        #     'precision' : precision,
        #     'recall' : recall,
        #     'f1' : f1,
        # }, index = emotions)
        # )


def test_test():
    m = max(len(x) for x in embeddings)
    embeddings = np.stack([
        np.pad(x,m) for x in embeddings
    ])

    train_in = np.zeros((
        train_df.shape[0],
        m
    ))
    train_in[:] = train_df.loc[:,'embeddings']
