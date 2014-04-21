import nltk.classify.util
import csv
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import movie_reviews
import random
 
def word_feats(words):
    return dict([(word, True) for word in words])

class PipeDialect(csv.Dialect):
    delimiter = "|"
    quotechar = None
    escapechar = None
    doublequote = None
    lineterminator = "\r\n"
    quoting = csv.QUOTE_NONE
    skipinitialspace = False

fneg = "data.neg.txt"
fpos = "data.pos.txt"
f = "data.txt"
fileNeg = open(fneg, "rb")
filePos = open(fpos, "rb")
file = open(f, "rb")


reader = csv.reader(file, PipeDialect())
readerNeg = csv.reader(fileNeg, PipeDialect())
readerPos = csv.reader(filePos, PipeDialect())

sentencesMostNeg = []
sentencesNeg = []
sentencesNeutral = []
sentencesPos = []
sentencesMostPos = []

wordsMostNeg = []
wordsNeg = []
wordsNeutral = []
wordsPos = []
wordsMostPos = []

for row in reader:
    value = float(row[1])
    if value >= 0 and value < 0.2:
        sentencesMostNeg.append(row[2].lower())
    elif value >= 0.2 and value < 0.4:
        sentencesNeg.append(row[2].lower())
    elif value >= 0.4 and value < 0.6:
        sentencesNeutral.append(row[2].lower())
    elif value >= 0.6 and value < 0.8:
        sentencesPos.append(row[2].lower())
    else:
        sentencesMostPos.append(row[2].lower())


tokenizer = RegexpTokenizer(r'\w+')

for i in range(0, len(sentencesMostNeg)-1):
    wordsMostNeg.append(tokenizer.tokenize(sentencesMostNeg[i]))

for i in range(0, len(sentencesNeg)-1):
    wordsNeg.append(tokenizer.tokenize(sentencesNeg[i]))

for i in range(0, len(sentencesNeutral)-1):
    wordsNeutral.append(tokenizer.tokenize(sentencesNeutral[i]))

for i in range(0, len(sentencesPos)-1):
    wordsPos.append(tokenizer.tokenize(sentencesPos[i]))

for i in range(0, len(sentencesMostPos)-1):
    wordsMostPos.append(tokenizer.tokenize(sentencesMostPos[i]))

mostnegfeats = [(word_feats(wordsMostNeg[i]), 'very bad') for i in range(0, len(wordsMostNeg)-1)]
negfeats = [(word_feats(wordsNeg[i]), 'bad') for i in range(0, len(wordsNeg)-1)]
neutralfeats = [(word_feats(wordsNeutral[i]), 'neutral') for i in range(0, len(wordsNeutral)-1)]
posfeats = [(word_feats(wordsPos[i]), 'good') for i in range(0, len(wordsPos)-1)]
mostposfeats = [(word_feats(wordsMostPos[i]), 'very good') for i in range(0, len(wordsMostPos)-1)]

random.shuffle(mostnegfeats)
random.shuffle(mostposfeats)
random.shuffle(negfeats)
random.shuffle(posfeats)
random.shuffle(neutralfeats)

mostnegcutoff = len(mostnegfeats)*3/4
negcutoff = len(negfeats)*3/4
neutralcutoff = len(neutralfeats)*3/4
poscutoff = len(posfeats)*3/4
mostposcutoff = len(mostposfeats)*3/4
 
trainfeats = mostnegfeats[:mostnegcutoff] + negfeats[:negcutoff] + neutralfeats[:neutralcutoff] + posfeats[:poscutoff] + mostposfeats[:mostposcutoff]
testfeats = mostnegfeats[mostnegcutoff:] + negfeats[negcutoff:] + neutralfeats[neutralcutoff:] + posfeats[poscutoff:] + mostposfeats[mostposcutoff:]
print 'train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats))
 
classifier = NaiveBayesClassifier.train(trainfeats)
print 'accuracy:', nltk.classify.util.accuracy(classifier, testfeats)

classifier.show_most_informative_features()



file.close()
filePos.close()
fileNeg.close()
