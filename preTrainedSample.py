"""
Originally inspired by
https://keras.io/examples/nlp/pretrained_word_embeddings/
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import TextVectorization
from keras.layers import Embedding
from keras.models import load_model
from statistics import mean, median
import os
import pathlib
from datetime import datetime as dt
import json
HEADER = False
dataFocus = ('document', 'sentence', 'line')[2]

def findSampleMax(data_dir, dirnames):
    maxClassSample = 0
    for dirname in sorted(dirnames):
        fnames = os.listdir(data_dir / dirname)
        if dataFocus == 'document':
            metric = len(fnames)
        elif dataFocus == 'line':
            lineCount = 0
            for fName in fnames:
                f = open(os.path.join(os.getcwd(), data_dir, dirname, fName), encoding="latin-1")
                content = f.read()
                lines = content.replace('.', '\n').split("\n")
                lineCount += len(lines)
            metric = lineCount
        else:
            print(f"findSampleMax(): ERROR - Unintelligible or unimplemented dataFocus {dataFocus}")
            exit()
        if maxClassSample == 0:
            maxClassSample = metric
        elif maxClassSample > metric:
            maxClassSample = metric
        print(f"findSampleMax with {dataFocus} focus: {dirname}: {metric}")
    return maxClassSample

def acquireData(dataDir):
    """
    generate a list of labels and sample data by processing the path provided
    :param dataDir: the path to a directory with directories of documents where
    the subdirectory is treated as the label and the documents are sample data
    :return: list of labels, list of data
    """
    DBName = 'NorthAmer273.txt'
    print(f"Reading in training and testing data from {dataDir}")
    dirnames = os.listdir(dataDir)
    dirnames = sorted([x for x in dirnames if '.DS_Store' != x])
    print("Summary Info:")
    print(f"{len(dirnames)} Directory names: {dirnames}")

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
    for dirname in sorted(dirnames)[:5]:
        class_names.append(dirname)
        dirpath = data_dir / dirname
        fnames = os.listdir(dirpath)
        print(f"Processing directory {dirname}, {len(fnames)} files will be associated "
              f"with the label {dirname}")
        print(f"Data will be processed at the *{dataFocus}* level of analysis")
        sampleCount = 0
        for IX, fname in enumerate(fnames):
            if DBName == fname:
                print(f"here {DBName}")
            #if IX >= maxClassSample:
            if sampleCount >= 100:
                IX = IX -1
                break
            fpath = dirpath / fname
            f = open(fpath, encoding="latin-1")
            content = f.read()
            lines = content.replace('.', '\n').split("\n")
            if HEADER:
                lines = lines[10:]  # skip header info in message
            if dataFocus == 'document':
                content = "\n".join(lines).lower()  # recombine msg as single string
                samples.append(content)  # list of training strings/documents
                labels.append(class_index)   # label index associated with text
                sampleCount += 1
            elif dataFocus == 'line':
                for content in lines:
                    if sampleCount < maxClassSample:
                        samples.append(content)  # list of training strings/documents
                        labels.append(class_index)   # label index associated with text
                        sampleCount += 1
            else:
                print(f"acquireData(): ERROR - Unintelligible or unimplemented dataFocus {dataFocus}")
                exit()
        print(f"{dirname} class from {len(fnames)} files:  {sampleCount} samples used and {IX + 1} files used.")
        class_index += 1
    print(f"Classes ({len(class_names)}):", class_names)
    print("Number of samples:", len(samples))
    return labels, samples


def organizeData(labels, samples, batchSize=128, validSplit=0.2, mxVocab=20000,
                 mxSentence=200):
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

    trainWC = [len(samp.split(' ')) for samp in train_samples]
    tooLong = sum(1 if x > mxSentence else 0 for x in trainWC)
    print(f"{len(train_samples)} training samples word count range "
          f"{min(trainWC)}:{max(trainWC)}, ave: {mean(trainWC)}, med:{median(trainWC)}")
    print(f"{tooLong} ({tooLong/len(trainWC)}) of the training phrases were "
          f"truncated by using threshold of {mxSentence}.")
    valWC = [len(samp.split(' ')) for samp in val_samples]
    tooLong = sum(1 if x > mxSentence else 0 for x in valWC)
    print(f"{len(valWC)} validation samples word count range "
          f"{min(valWC)}:{max(valWC)}, ave: {mean(valWC)}, med:{median(valWC)}")
    print(f"{tooLong} ({tooLong/len(valWC)}) of the validation phrases were "
          f"truncated by using threshold of {mxSentence}.")
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
    print(f"The sentence '{sampleText}' converts into a vector of {len(output[0])} "
          f"token indexes that is padded with 0s of:"
          f"{output.numpy()[0, :6]}")
    return train_samples, train_labels, val_samples, val_labels, vectorizer


def loadEmbedding(vectorizer, embedFile):

    voc = vectorizer.get_vocabulary()
    word_index = dict(zip(voc, range(len(voc))))

    print(f"Begin acquiring embedding vectors from {embedFile}")
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
#    print(f"'radar': {voc.index('radar')} 'anomaly': {voc.index('anomaly')}")
    print("Converted %d words (%d misses)" % (hits, misses))
    print("'misses' are words that didn't make it into our embedding space.")
    print(f"Full list of misses: {missed}")
    return embedding_matrix


def designModel(embedMatrix, classCount, batchSize, vectorizer):
    # Create the keras embedding layer and initialize it using our acquired embedding vectors
    #from tensorflow.keras.layers import Embedding

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
    print(f"A 4-layer Conv1D with {batchSize} input vectors having window size of {windowSz} ")
    print("Global Max pooling looks across the input vector and takes the single largest value")
    print(f"A Dropout layer with dropout probability of {dropProb} is near the end of the model.")

    int_sequences_input = keras.Input(shape=(None,), dtype="int64")
    embedded_sequences = embedding_layer(int_sequences_input)
    x = layers.Conv1D(batchSize, windowSz, activation="relu")(embedded_sequences)
    x = layers.MaxPooling1D(windowSz)(x)
    x = layers.Conv1D(batchSize, windowSz, activation="relu")(x)
    x = layers.MaxPooling1D(windowSz)(x)
    x = layers.Conv1D(batchSize, windowSz, activation="relu")(x)
    x = layers.MaxPooling1D(windowSz)(x)
    x = layers.Conv1D(batchSize, windowSz, activation="relu")(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(batchSize, activation="relu")(x)
    x = layers.Dropout(dropProb)(x)
    preds = layers.Dense(classCount, activation="softmax")(x)
    model = keras.Model(int_sequences_input, preds)
    print("A summary description of our model.")
    model.summary()
    return model


def trainModel(trainData, trainLabels, validData, validLabels, vectorizer,
               model, batchSize=128, epochs=20):

    x_train = vectorizer(np.array([[s] for s in trainData])).numpy()
    x_val = vectorizer(np.array([[s] for s in validData])).numpy()

    y_train = np.array(trainLabels)
    y_val = np.array(validLabels)

    model.compile(
        loss="sparse_categorical_crossentropy", optimizer="rmsprop",
        metrics=["acc"]
    )
    model.fit(x_train, y_train, batch_size=batchSize, epochs=epochs,
              validation_data=(x_val, y_val))
    string_input = keras.Input(shape=(1,), dtype="string")
    x = vectorizer(string_input)
    preds = model(x)
    end_to_end_model = keras.Model(string_input, preds)

    return end_to_end_model

# ADD DETECTION OF WORDS THAT AREN'T IN THE VOCABULARY
def testModel(testFileName, modelTrained, classLabels, vectorizer):
    """
    run the model on the contents of a test file that contains
    label, probe text pairs.
    :param testFileName: name of file with label, probe text
    :param modelTrained: model
    :param classLabels: class labels
    :return:  list of dict describing each result
    """
    contents = open(testFileName).readlines()
    results = []
    voc = vectorizer.get_vocabulary(include_special_tokens=False)
    for con in contents:
        label, text = con.rstrip().split(',', 1)
        misses = [word for word in text.split(' ') if word not in voc]
        if misses:
            print(f"Invalid tokens {', '.join(misses)} given. The prompt: \n{text}\nwill be skipped in testing.")
            continue
        probabilities = modelTrained.predict([[text]])
        result = {
            'probe': text,
            'expected label': label,
            'predicted label': classLabels[np.argmax(probabilities[0])],
            'probabilities':
                {classLabels[ix]: probabilities[0][ix]
                 for ix in range(len(classLabels))
                 },
            'correct': 1 if label == classLabels[np.argmax(probabilities[0])] else 0
        }
        results.append(result)
    if results:
        return results

    else:
        return ['All prompts contained unknown tokens']

def showResults(results):
    """
    display a list of results
    :param results: list of dicts with each dict describing the model product
    from a text probe
    :return:
    """
    try:
        results[1]
    except IndexError:
        print('Attempting to present blank results, exiting showResults')
        return
    probKeys = results[0]['probabilities'].keys()

    print(f"expected label,predicted label,{[' prob, '.join(probKeys)]} prob,probe")
    summary = dict()
    for res in results:
        print(f"{res['expected label']},{res['predicted label']},"
              f"{','.join([str(res['probabilities'][k]) for k in probKeys])},{res['probe']}")
        if res['expected label'] in summary.keys():
            summary[res['expected label']].append(res['correct'])
        else:
            summary[res['expected label']] = [res['correct']]
    print('label,correct,incorrect,total')
    for k in summary.keys():
        print(f"{k},{sum(summary[k])},{len(summary[k])-sum(summary[k])},"
              f"{len(summary[k])}")
    return


def useModel(model, classLabels):
    """
    User interaction with the text model
    :param model:  trained model
    :param classLabels: list of class labels
    :return:
    """

    testText = "this message is about computer graphics and 3D modeling".lower()
    probabilities = model.predict([[testText]])
    print(f"'{testText}' -> {classLabels[np.argmax(probabilities[0])]}")
    testText = "radar anomaly".lower()
    probabilities = model.predict([[testText]])
    print(f"'{testText}' -> {classLabels[np.argmax(probabilities[0])]}")
    testText = "welfare should be replaced by religious charities".lower()
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
        print(f"{','.join(f'{classLabels[ix]}: {probabilities[0][ix]}' for ix in range(len(classLabels)) if ix != bestProbIX)}")
    return

def saveModel(model, params, fName):
    """
    save the parameters of the model to a JSON and and a copy of the model
    :param model: trained model
    :param params: training parameters
    :param fName: name to write model to
    :return:
    """
    print(f"Saving parameters in {modelFileName}.json")
    print(f"Saving model in {modelFileName}.xx")
    with open(f'{modelFileName}.json', 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=4)
    model.save(fName)
    return


def loadModel(fName):
    """
    load a pre-trained model
    NOT TESTED YET
    :param fName:
    :return:
    """
    model = load_model(fName)
    return model

if __name__ == '__main__':
    startTime = dt.now()
    embedding_dim = 50
    batchSize = 128
    epochs = 40
    mxSentence = 750  # good: 750 800 900 1000  # bad: 740 720 700 600, 500, 100
    mxVocab = 20000
    dataSource = ['local', "news"][1]
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
        organizeData(labels, data, batchSize=batchSize, validSplit=0.2,
                     mxVocab=mxVocab, mxSentence=mxSentence)
    embedFileName = os.path.join(os.getcwd(), "glove.6B", f"glove.6B.{embedding_dim}d.txt")
    print(f"Loading vector space with {embedding_dim} dimensions.")
    embedSpace = loadEmbedding(vectorizer, embedFileName)
    modelArch = designModel(embedSpace, len(classLabels), batchSize, vectorizer)
    modelTrained = trainModel(samplesTrain, labelsTrain, samplesValidate,
                              labelsValidate, vectorizer, modelArch,
                              batchSize=batchSize, epochs=epochs)
    testFileName = r"C:\Users\desmo\OneDrive\Desktop\GitHub\DogWhistleTF\testProbes.txt"
    results = testModel(testFileName, modelTrained, classLabels, vectorizer)
    showResults(results)
    modelTime = dt.now().isoformat()[:19].replace(':', '_')
    modelFileName = f"model_{modelTime}"
    endTime = dt.now()
    # add test probe results
    params = {
        'modelFileName': modelFileName,
        'embedding_dim': embedding_dim,
        'batchSize': batchSize,
        'mxSentence': mxSentence,
        'mxVocab': mxVocab,
        'epochs': epochs,
        'dataSource': dataSource,
        'classCount': classCount,
        'classes': list(set(labels)),
        'embedFileName': embedFileName,
        'testFileName': testFileName,
        'startTime': str(startTime),
        'endTime': str(endTime),
        'duration (s)': (endTime-startTime).total_seconds()
    }
    saveModel(modelTrained, params, modelFileName)

    useModel(modelTrained, classLabels)

    newModel = loadModel(modelFileName)
    retestResults = testModel(testFileName, newModel, classLabels, vectorizer)
    showResults(results)
