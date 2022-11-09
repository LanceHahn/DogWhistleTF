import os
import numpy as np
from scipy.spatial import distance
from tensorflow import keras
from keras import layers
from PartyClassifier import fetchData
from preTrainedSample import organizeData, loadEmbedding


def retrainEmbeddings(dim, folderName, batchSize, windowSz) -> np.array:
    """Return a retrained embedding layer based on GloVe embeddings of given dimension

    Args:
        dim (int): Dimension of GloVe vectors in given folder
        samples (numpy array): Data samples to train with
        labels (numpy array): Data labels to train with
        folderName (str): Path to folder from CWD
    Returns:
        TensorFlow Embedding Layer: Embedding space with retrained embeddings
    """

    whole_labels, whole_samples = fetchData(folderName)
    train_samples, train_labels, val_samples, val_labels, vectorizer = organizeData(whole_labels, whole_samples)
    classCount = len(set(whole_labels))

    standard_embeddings = loadEmbedding(vectorizer, os.path.join(os.getcwd(), "glove.6B", f"glove.6B.{dim}d.txt"))

    voc = vectorizer.get_vocabulary()
    embedding_layer = layers.Embedding(
        len(voc) + 2,
        len(standard_embeddings[0]),
        embeddings_initializer=keras.initializers.Constant(standard_embeddings),
        trainable = True
    )

    int_input = keras.Input(shape=(None,), dtype="int64")
    embedded_int = embedding_layer(int_input)

    x = layers.Conv1D(batchSize, windowSz, activation="relu")(embedded_int)
    x = layers.MaxPooling1D(windowSz)(x)
    x = layers.Conv1D(batchSize, windowSz, activation="relu")(x)
    x = layers.MaxPooling1D(windowSz)(x)
    x = layers.Conv1D(batchSize, windowSz, activation="relu")(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(batchSize, activation="relu")(x)
    x = layers.Dropout(.5)(x)
    final = layers.Dense(classCount, activation="softmax")(x)

    main_model = keras.Model(int_input, final)
    
    main_model.summary()

    int_train_samples = vectorizer(np.array([[s] for s in train_samples])).numpy()
    int_val_samples = vectorizer(np.array([[s] for s in val_samples])).numpy()

    int_train_labels = np.array(train_labels)
    int_val_labels = np.array(val_labels)

    main_model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="rmsprop",
        metrics=["acc"]
    )

    main_model.fit(int_train_samples, int_train_labels, batch_size=128,
    epochs=20, validation_data=(int_val_samples, int_val_labels))

    string_layer = keras.Input(shape=(1,), dtype="string")
    x = vectorizer(string_layer)
    preds = main_model(x)
    main_model.summary()
    final_model = keras.Model(string_layer, preds)
    final_model.summary()
    return embedding_layer.embeddings


def compareEmbeddings(untrained, trained):
    """Function comparing word embedding similarities from two embedding spaces using GenSim

    Args:
        untrained (Numpy matrix): Standard embedding space
        trained (Numpy matrix): Retrained/customized embedding space
    """
    if len(untrained) != len(trained):
        raise Exception("Matrix dimensions must be identical")

    untr_dict={}
    tr_dict = {}
    with open(os.path.join(os.getcwd(), "glove.6B", "glove.6B.50d.txt")) as f:
        for line, untr_vect, tr_vect in zip(f, untrained, trained):
            word = line.split(maxsplit=1)[0]
        untr_dict[word] = untr_vect
        tr_dict[word] = tr_vect


if __name__ == "__main__":
    organizeData(fetchData("trump_obama_corpus"))
    new_embed = retrainEmbeddings(50, "trump_obama_corpus", 128, 5)
    old_embed = loadEmbedding(organizeData(fetchData("trump_obama_corpus"))[4], os.path.join(os.getcwd(), "glove.6B", "glove.6b.50d.txt"))
    compareEmbeddings(old_embed, new_embed)
    