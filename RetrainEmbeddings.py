import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
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
    epochs=10, validation_data=(int_val_samples, int_val_labels))

    string_layer = keras.Input(shape=(1,), dtype="string")
    x = vectorizer(string_layer)
    preds = main_model(x)
    main_model.summary()
    final_model = keras.Model(string_layer, preds)
    final_model.summary()
    return embedding_layer, vectorizer


def compareEmbeddings(untrained_mat, trained_layer, vectorizer, n_words):
    """Function comparing word embedding similarities from two embedding spaces using GenSim

    Args:
        untrained (Numpy matrix): Standard embedding space
        trained (Numpy matrix): Retrained/customized embedding space as Keras Embedding layer
        vectorier (TensorFlow TextVectorization object): Vectorizer used to create trained embeddings
        n_words (int): Number of most similar words to compare
    """
    untrained_layer = layers.Embedding(
        len(vectorizer.get_vocabulary()) + 2,
        len(untrained_mat[0]),
        embeddings_initializer=keras.initializers.Constant(untrained_mat),
        trainable=False
    )

    trained_mat = trained_layer.weights[0].numpy()
    vocab = np.array(vectorizer.get_vocabulary())


    test = input("Input word to be compared(enter quit to stop being prompted): ")
    while(test.lower() != "quit"):
        print(f"Using standard embeddings, the {n_words} most similar words for {test} are:")
        print(vocab[np.argpartition(distance.cdist([untrained_layer(vectorizer(test))[0]], untrained_layer.weights[0].numpy()), n_words+1)[0][1:n_words]])
        print(f"Using customized embeddings, the {n_words} most similar words for {test} are:")
        print(vocab[np.argpartition(distance.cdist([trained_layer(vectorizer(test))[0]], trained_layer.weights[0].numpy()), n_words+1)[0][1:n_words]])
        test = input("Input word to be compared(enter quit to stop being prompted): ")
        

if __name__ == "__main__":
    new_embed, vectorizer = retrainEmbeddings(50, "trump_obama_corpus", 128, 5)
    old_embed = loadEmbedding(vectorizer, os.path.join(os.getcwd(), "glove.6B", "glove.6b.50d.txt"))
    compareEmbeddings(old_embed, new_embed, vectorizer, 5)
    