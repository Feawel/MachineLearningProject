import csv
import collections, itertools
import nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews, stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import random
from nltk.stem.snowball import EnglishStemmer

stopset = set(stopwords.words('english'))
lmtzr = EnglishStemmer(True)

#Different frequencies/scoring function between unigram and bigram
def bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=150):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return dict([(ngram, True) for ngram in itertools.chain(words, bigrams)])

#Construction du dictionnaire en tenant compte des stopwords
def stopword_filtered_word_feats(words):
    return dict([(word, True) for word in words if word not in stopset])

#Construction du dictionnaire en tenant compte des stopwords
def stemming_word_feats(words):
    return dict([(lmtzr.stem(word), True) for word in words])

# Construction du dictionnaire avec variable bool indiquant presence du mot
def word_feats(words):
    return dict([(word, True) for word in words])

#Classe permettant l'extraction des phrases du fichier
class PipeDialect(csv.Dialect):
    delimiter = "|"
    quotechar = None
    escapechar = None
    doublequote = None
    lineterminator = "\r\n"
    quoting = csv.QUOTE_NONE
    skipinitialspace = False

#Classifieur binaire de base
def evaluate_classifier(featx):
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

    words = wordsNeg + wordsPos
    print len(set([y for x in words for y in x]))

    negfeats = [(featx(wordsNeg[i]), 'neg') for i in range(0, len(wordsNeg)-1)]
    posfeats = [(featx(wordsPos[i]), 'pos') for i in range(0, len(wordsPos)-1)]

    print len(set([lmtzr.stem(y) for x in words for y in x]))

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

print 'evaluating single word features'
evaluate_classifier(word_feats)

print 'evaluating single word features with no stopword'
evaluate_classifier(stopword_filtered_word_feats)

print 'evaluating single word features with no stopword and stemming'
evaluate_classifier(stemming_word_feats)
