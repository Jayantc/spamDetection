import os
import io
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
import joblib


def readFiles(path):
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(root, filename)

            inBody = False
            lines = []
            f = io.open(path, 'r', encoding='latin1')
            for line in f:
                if inBody:
                    lines.append(line)
                elif line == '\n':
                    inBody = True
            f.close()
            message = '\n'.join(lines)
            yield path, message


def dataFrameFromDirectory(path, classification):
    rows = []
    index = []
    for filename, message in readFiles(path):
        rows.append({'message': message, 'class': classification})
        index.append(filename)

    return DataFrame(rows, index=index)

data = DataFrame({'message': [], 'class': []})

data = data.append(dataFrameFromDirectory('spam', 'spam'))
data = data.append(dataFrameFromDirectory('ham', 'ham'))


vectorizer = CountVectorizer()
vectorizer.fit_transform(data['message'].values)


model= joblib.load('SpamDetectModel')
examples = [input('Paste your email data: ')]
example_counts = vectorizer.transform(examples)
predictions = model.predict(example_counts)
print(predictions)