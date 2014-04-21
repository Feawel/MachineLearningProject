import csv
import collections, itertools
import nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews, stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.tokenize import RegexpTokenizer
import random
from nltk.stem.snowball import EnglishStemmer

lmtzr = EnglishStemmer(True)

# Construction du dictionnaire avec variable bool indiquant presence du mot
def word_feats(words):
    return dict([(lmtzr.stem(word), True) for word in words])

def best_word_feats(words):
    return dict([(lmtzr.stem(word), True) for word in words if word in bestwords])

#On recupere uniquement les 200 bigrams les plus pertinents
def bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=5000):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    d = dict([(bigram, True) for bigram in bigrams])
    d.update(word_feats(words))
    return d

#On recupere uniquement les 200 bigrams les plus pertinents
def best_bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=5000):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    d = dict([(bigram, True) for bigram in bigrams])
    d.update(best_word_feats(words))
    return d

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
    totalfeats = negfeats + posfeats

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
wordsFullPos = []
wordsFullNeg = []

for row in readerNeg:
   sentencesNeg.append(row[2])
for row in readerPos:
   sentencesPos.append(row[2])

tokenizer = RegexpTokenizer(r'\w+')
for i in range(0, len(sentencesNeg)-1):
    wordsNeg.append(tokenizer.tokenize(sentencesNeg[i]))

for i in range(0, len(sentencesPos)-1):
    wordsPos.append(tokenizer.tokenize(sentencesPos[i]))

#Frequence des mots

word_fd = FreqDist()
label_word_fd = ConditionalFreqDist()

for i in range(0, len(wordsPos)-1):
    for j in range(0, len(wordsPos[i])-1):
        wordsFullPos.append(wordsPos[i][j])
for i in range(0, len(wordsNeg)-1):
    for j in range(0, len(wordsNeg[i])-1):
        wordsFullNeg.append(wordsNeg[i][j])

for word in wordsFullPos:
    word_fd.inc(word.lower())
    label_word_fd['pos'].inc(word.lower())

for word in wordsFullNeg:
    word_fd.inc(word.lower())
    label_word_fd['neg'].inc(word.lower())

# n_ii = label_word_fd[label][word]
# n_ix = word_fd[word]
# n_xi = label_word_fd[label].N()
# n_xx = label_word_fd.N()

#Nombre d'occurence des mots

pos_word_count = label_word_fd['pos'].N()
neg_word_count = label_word_fd['neg'].N()
total_word_count = pos_word_count + neg_word_count


word_scores = {}

#Utilisation de bigrams

for word, freq in word_fd.iteritems():
    pos_score = BigramAssocMeasures.chi_sq(label_word_fd['pos'][word],
    (freq, pos_word_count), total_word_count)
    neg_score = BigramAssocMeasures.chi_sq(label_word_fd['neg'][word],
    (freq, neg_word_count), total_word_count)
    word_scores[word] = pos_score + neg_score

#On utilise uniquement les 5 000 mots les plus informatifs
best = sorted(word_scores.iteritems(), key=lambda (w,s): s, reverse=True)[:5000]
bestwords = set([w for w, s in best])


print 'evaluating best word features'
evaluate_classifier(best_word_feats)




print 'bigram word features'
evaluate_classifier(bigram_word_feats)

print 'evaluating best words + bigram word features'
evaluate_classifier(best_bigram_word_feats)