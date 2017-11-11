from gensim import corpora, models, similarities 
import os
from six import iteritems

# Generate dataset and save

# Corpus Streaming
class MyCorpus(object):
    def __iter__(self):
        with open("C:\Users\avour\Documents\GitHub\topic-indexing-lsa-gensim\my_corpus.txt", "r") as file:
            for line in file:
                # Assume there's one document per line, tokens separated by whitespace
                yield dictionary.doc2bow(line.lower().split())

corpus = MyCorpus()

# Dictionary
# collect statistics about all tokens

class MyDictionary(object):
    def __iter__(object):
        with open('C:\\Users\avour\\Documents\\GitHub\\topic-indexing-lsa-gensim\\my_corpus.txt', 'r') as file:
            dictionary = corpora.Dictionary(line.lower().split() for line in file)

            # remove stop words and words that appear only once
            stop_ids = [dictionary.token2id[stopword] for stopword in stoplist
                       if stopword in dictionary.token2id]
            once_ids = [tokenid for tokenid, docfreq in iteritems(dicitionary.dfs) if docfreq==1]
            dictionary.filter_tokens(stop_ids+once_ids)
            dictionary.compactify()
            yield dictionary

dictionary = MyDictionary()

# Save corpus and dictionary
corpora.MmCorpus.serialize('something.mm', corpus)
dictionary.save('something.dict')

# Corpus of documents represented as a stream of vectors
if(os.path.exists('something.dict')):
    corpus = corpora.MmCorpus('/something.mm')
    dictionary = corpora.Dictionary.load('/something.dict')
    print('Used saved dataset')
else:
    print('Please generate data set')


# Step 1 -- initialize a model
tfidf = models.TfidfModel(corpus)

# Step 2 -- use models to transform vectors
corpus_tfidf = tfifd[corpus]
for doc in corpus_tfidf:
    print(doc)

# Step 3 -- Perform transformation

lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2) #Initialize an LSI transformation
corpus_lsi = lsi[corpus_tfidf] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
    
lsi.print_topics(2)

for doc in corpus_lsi:
    print(doc)
