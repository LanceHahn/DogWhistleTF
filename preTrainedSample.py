"""
Originally inspired by
https://keras.io/examples/nlp/pretrained_word_embeddings/
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import TextVectorization

import os
import pathlib

HEADER = False

def findSampleMax(data_dir, dirnames):
    maxClassSample = 0
    classCounts = {}
    for dirname in sorted(dirnames):
        fnames = os.listdir(data_dir / dirname)
        classCounts[dirname] = len(fnames)
        if maxClassSample == 0:
            maxClassSample = len(fnames)
        elif maxClassSample > len(fnames):
            maxClassSample = len(fnames)

    return maxClassSample

def acquireData(dataDir):
    """
    generate a list of labels and sample data by processing the path provided
    :param dataDir: the path to a directory with directories of documents where
    the subdirectory is treated as the label and the documents are sample data
    :return: list of labels, list of data
    """
    print(f"Reading in training and testing data from {dataDir}")
    dirnames = os.listdir(dataDir)
    dirnames = sorted([x for x in dirnames if '.DS_Store' != x])
    print("We've created an access point to some newsgroup data. \nSummary Info:")
    print("Number of directories:", len(dirnames))
    print("Directory names:", dirnames)

    sampleDir = dirnames[0]
    fnames = os.listdir(dataDir / sampleDir)
    print(f"As an example, here are the number of files in one directory named `{sampleDir}`:", len(fnames))
    print(f"Some example filenames from '{sampleDir}':", fnames[:5])

    sampleFile = dataDir / sampleDir / fnames[0]
    print(f"\nA sample file ({sampleFile}) contents:")
    print(open(dataDir / sampleDir / fnames[0]).read())
    print("Acquiring data as a list of document strings and a list of labels associated with the strings.")
    print("The class name is the source discussion board and the label is the index")
    samples = []
    labels = []
    class_names = []
    class_index = 0
    maxClassSample = findSampleMax(data_dir, dirnames)
    for dirname in sorted(dirnames):
        class_names.append(dirname)
        dirpath = data_dir / dirname
        fnames = os.listdir(dirpath)
        print(f"Processing directory {dirname}, {len(fnames)} files will be associated "
              f"with the label {dirname}")
        for IX, fname in enumerate(fnames):
            if IX >= maxClassSample:
                IX = IX -1
                break
            fpath = dirpath / fname
            f = open(fpath, encoding="latin-1")
            content = f.read()
            lines = content.split("\n")
            if HEADER:
                lines = lines[10:]  # skip header info in message
            content = "\n".join(lines).lower()  # recombine msg as single string
            samples.append(content)  # list of training strings/documents
            labels.append(class_index)   # label index associated with text
        print(f"{dirname} class:  {len(fnames)} samples available and {IX + 1} used.")
        class_index += 1
    print(f"Classes ({len(class_names)}):", class_names)
    print("Number of samples:", len(samples))
    return labels, samples


def organizeData(labels, samples, batchSize=128, validSplit=0.2, mxVocab=20000, mxSentence=200):
    """
    Break the data into a training set and a validation set.  Also create
    a vectorizer that can be used to convert a string into a list of indexes which
    can be used to train and use the model

    :param labels: the list of labels associated with the data samples
    :param samples: the list of the data samples
    :param batchSize: how many data samples are processed at one time
    :param validSplit: the proportion of data to be used for validation
    :param mxVocab: the size limit of the vocabulary
    :param mxSentence: the number of tokens in a sentence
    :return:
    """
    # Shuffle the data
    seed = 1337
    rng = np.random.RandomState(seed)
    rng.shuffle(samples)
    rng = np.random.RandomState(seed)
    rng.shuffle(labels)

    # Extract a training & validation split
    num_validation_samples = int(validSplit * len(samples))
    print(f"{validSplit} of dataset of total {len(samples)} is separated into "
          f"{len(samples)-num_validation_samples} training samples and"
          f" {num_validation_samples} validation samples.")
    train_samples = samples[:-num_validation_samples]
    val_samples = samples[-num_validation_samples:]
    train_labels = labels[:-num_validation_samples]
    val_labels = labels[-num_validation_samples:]
    print(f"{len(train_samples)} training samples")
    print(f"{len(val_labels)} validation samples")
    print(f"Vector representing a sentence is at most {mxSentence} tokens long.")
    print(f"Vocabulary is the top {mxVocab} words.")
    print(f"The training sample will be processed in batches containing {batchSize} values.")

    print(f"Training vectors are {mxSentence} in length, representing single sentences.")
    vectorizer = TextVectorization(max_tokens=mxVocab, output_sequence_length=mxSentence)
    text_ds = tf.data.Dataset.from_tensor_slices(train_samples).batch(batchSize)
    vectorizer.adapt(text_ds)

    print(f"The top 5 words in vectorization: {vectorizer.get_vocabulary()[:5]}")
    sampleText = "the cat sat on the mat"
    output = vectorizer([[sampleText]])
    print(f"The sentence '{sampleText}' converts into a vector of {len(output[0])} token indexes that is padded with 0s of:"
          f"{output.numpy()[0, :6]}")
    return train_samples, train_labels, val_samples, val_labels, vectorizer


def loadEmbedding(vectorizer, embedFile):

    voc = vectorizer.get_vocabulary()
    word_index = dict(zip(voc, range(len(voc))))

    print(f"Begin acquiring embedding vectpors from {embedFile}")
    #C:\Users\drlwh\OneDrive\Documents\GitHub\DogWhistleTF\glove.6B
    embeddings_index = {}
    with open(embedFile, encoding="utf8") as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs
    print(f"Found {len(embeddings_index)} word vectors. Each vector is {len(embeddings_index[word])} long.")

    num_tokens = len(voc) + 2
    hits = 0
    misses = 0
    # Prepare embedding matrix
    embedding_matrix = np.zeros((num_tokens, len(embeddings_index['the'])))
    missed = set()
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            # This includes the representation for "padding" and "OOV"
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
            missed.add(word)
    print("Converted %d words (%d misses)" % (hits, misses))
    print("'misses' are words that didn't make it into our embedding space.")
    print(f"Full list of misses: {missed}")
    return embedding_matrix


def designModel(embedMatrix, classCount, batchSize, vectorizer):
    # Create the keras embedding layer and initialize it using our acquired embedding vectors
    #from tensorflow.keras.layers import Embedding
    from keras.layers import Embedding
    voc = vectorizer.get_vocabulary()
    num_tokens = len(voc) + 2
    embedding_layer = Embedding(
        num_tokens,
        len(embedMatrix[0]),
        embeddings_initializer=keras.initializers.Constant(embedMatrix),
        trainable=False,
    )

    # A simple 1D convnet with global max pooling and a classifier at the end.
    windowSz = 5
    dropProb = 0.5
    print(f"A 3-layer Conv1D with {batchSize} input vectors having window size of {windowSz} ")
    print("Global Max pooling looks across the input vector and takes the single largest value")
    print(f"A Dropout layer with dropout probability of {dropProb} is near the end of the model.")


    int_sequences_input = keras.Input(shape=(None,), dtype="int64")
    embedded_sequences = embedding_layer(int_sequences_input)
    x = layers.Conv1D(batchSize, windowSz, activation="relu")(embedded_sequences)
    x = layers.MaxPooling1D(windowSz)(x)
    x = layers.Conv1D(batchSize, windowSz, activation="relu")(x)
    x = layers.MaxPooling1D(windowSz)(x)
    x = layers.Conv1D(batchSize, windowSz, activation="relu")(x)
    # x = layers.MaxPooling1D(windowSz)(x)
    # x = layers.Conv1D(batchSize, windowSz, activation="relu")(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(batchSize, activation="relu")(x)
    x = layers.Dropout(dropProb)(x)
    preds = layers.Dense(classCount, activation="softmax")(x)
    model = keras.Model(int_sequences_input, preds)
    print("A summary description of our model.")
    model.summary()
    return model


def trainModel(trainData, trainLabels, validData, validLabels, vectorizer, model):

    x_train = vectorizer(np.array([[s] for s in trainData])).numpy()
    x_val = vectorizer(np.array([[s] for s in validData])).numpy()

    y_train = np.array(trainLabels)
    y_val = np.array(validLabels)

    model.compile(
        loss="sparse_categorical_crossentropy", optimizer="rmsprop",
        metrics=["acc"]
    )
    model.fit(x_train, y_train, batch_size=128, epochs=20,
              validation_data=(x_val, y_val))
    string_input = keras.Input(shape=(1,), dtype="string")
    x = vectorizer(string_input)
    preds = model(x)
    end_to_end_model = keras.Model(string_input, preds)

    return end_to_end_model


def useModel(model, classLabels):

    testText = "this message is about computer graphics and 3D modeling".lower()
    probabilities = model.predict([[testText]])
    print(f"'{testText}' -> {classLabels[np.argmax(probabilities[0])]}")
    print("Enter QUIT to stop being prompted for a phrase.")
    while testText != 'quit':
        testText = input("Phrase to categorize: ").lower()
        probabilities = model.predict([[testText]])
        bestProbIX = np.argmax(probabilities[0])
        bestClass = classLabels[bestProbIX]
        print(f"'{testText}' is most associated with class '{bestClass}' with weight "
              f"{probabilities[0][bestProbIX]}")
        # [print(f"{classLabels[ix]}: {probabilities[0][ix]} ")
        #  for ix in range(len(classLabels)) if ix != bestProbIX]
        print(f"{','.join(f'{classLabels[ix]}: {probabilities[0][ix]}' for ix in range(len(classLabels)) if ix != bestProbIX)}")


    return


if __name__ == '__main__':
    embedding_dim = 300
    batchSize = 128
    dataSource = ['local', "news"][0]
    if dataSource == "news":
        # news groups
        data_path = keras.utils.get_file(
            "news20.tar.gz",
            "http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.tar.gz",
            untar=True,
        )
        data_dir = pathlib.Path(data_path).parent / "20_newsgroup"
    else:
        # local data
        data_dir = pathlib.Path("data")

    dirnames = os.listdir(data_dir)
    dirnames = sorted([x for x in dirnames if '.DS_Store' != x])
    labels, data = acquireData(data_dir)

    classLabels = dirnames.copy()
    classCount = len(list(set(labels)))

    samplesTrain, labelsTrain, samplesValidate, labelsValidate, vectorizer = \
        organizeData(labels, data, batchSize=128, validSplit=0.2,
                     mxVocab=20000, mxSentence=200)
    embedFileName = os.path.join(os.getcwd(), "glove.6B", f"glove.6B.{embedding_dim}d.txt")
    print(f"Loading vector space with {embedding_dim} dimensions.")
    embedSpace = loadEmbedding(vectorizer, embedFileName)
    modelArch = designModel(embedSpace, len(classLabels), batchSize, vectorizer)
    modelTrained = trainModel(samplesTrain, labelsTrain, samplesValidate, labelsValidate, vectorizer, modelArch)
    useModel(modelTrained, classLabels)
