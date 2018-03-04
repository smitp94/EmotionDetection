import csv
from nltk import TweetTokenizer

def pre_process():
    data = []
    emotions = []
    word_dict = {}
    sentence = []

    with open('../data/text_emotion.csv') as csvDataFile:
        csv_reader = csv.reader(csvDataFile)
        for row in csv_reader:
            emotions.append(row[1])
            data.append(row[3])


    tknzr = TweetTokenizer()
    for d in data:
        tokens = tknzr.tokenize(d)
        sentence.append(tokens)

        # print(tokens)

    for s in sentence:
        for i in s:
            if i.lower() in word_dict:
                word_dict[i.lower()] += 1
            else:
                word_dict[i.lower()] = 1

    return [word_dict,sentence,emotions]


pre_process()
