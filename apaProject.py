import nltk.classify.util
import csv
from nltk.classify import NaiveBayesClassifier
import collections
import nltk.metrics
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

sentencesNeg = []
sentencesPos = []
wordsNeg = []
wordsPos = []

for row in readerNeg:
   sentencesNeg.append(row[2].lower())
for row in readerPos:
   sentencesPos.append(row[2].lower())


tokenizer = RegexpTokenizer(r'\w+')

for i in range(0, len(sentencesNeg)-1):
    wordsNeg.append(tokenizer.tokenize(sentencesNeg[i]))

for i in range(0, len(sentencesPos)-1):
    wordsPos.append(tokenizer.tokenize(sentencesPos[i]))

negfeats = [(word_feats(wordsNeg[i]), 'neg') for i in range(0, len(wordsNeg)-1)]
posfeats = [(word_feats(wordsPos[i]), 'pos') for i in range(0, len(wordsPos)-1)]

random.shuffle(negfeats)
random.shuffle(posfeats)

negcutoff = len(negfeats)*3/4
poscutoff = len(posfeats)*3/4
 
trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]

print 'train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats))
 
classifier = NaiveBayesClassifier.train(trainfeats)
refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)

for i, (feats, label) in enumerate(testfeats):
    refsets[label].add(i)
    observed = classifier.classify(feats)
    testsets[observed].add(i)
 
print 'accuracy:', nltk.classify.util.accuracy(classifier, testfeats)
print 'pos precision:', nltk.metrics.precision(refsets['pos'], testsets['pos'])
print 'pos recall:', nltk.metrics.recall(refsets['pos'], testsets['pos'])
print 'neg precision:', nltk.metrics.precision(refsets['neg'], testsets['neg'])
print 'neg recall:', nltk.metrics.recall(refsets['neg'], testsets['neg'])

classifier.show_most_informative_features()



file.close()
filePos.close()
fileNeg.close()
