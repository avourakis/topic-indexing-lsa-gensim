from gensim import corpora, models, similarities 
import os
from six import iteritems

directory_path = r"c:\users\avour\Documents\GitHub\topic-indexing-lsa-gensim\corpus"
corpus_name = r"corpus_1\corpus.txt"
corpus_path = os.path.join(directory_path, corpus_name)
print(corpus_path)

stop_list = set('for a of the and to is has they be are as from their in'.split())

# Generate Dictionary
with open(corpus_path, 'r') as file:
    dictionary = corpora.Dictionary(line.lower().split() for line in file)

    # remove stop words and words that appear only once
    stop_ids = [dictionary.token2id[stopword] for stopword in stop_list
        if stopword in dictionary.token2id]
    once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq==1]
    dictionary.filter_tokens(stop_ids+once_ids)
    #dictionary.filter_tokens(stop_ids)
    dictionary.compactify()

    
# Corpus Streaming
class MyCorpus(object):
    def __iter__(self):
        with open(corpus_path, 'r') as file:
            for line in file:
                # Assume there's one document per line, tokens separated by whitespace
                yield dictionary.doc2bow(line.lower().split())

corpus = MyCorpus()


# Save corpus and dictionary
corpora.MmCorpus.serialize('something.mm', corpus)
dictionary.save('something.dict')


# Corpus of documents represented as a stream of vectors

if(os.path.exists('something.dict')):
    corpus = corpora.MmCorpus('something.mm')
    dictionary = corpora.Dictionary.load('something.dict')
    print('Used saved dataset')
else:
    print('Please generate data set')


# Initialize tfidf model
tfidf = models.TfidfModel(corpus)

# Use tfidf model to transform vectors
corpus_tfidf = tfidf[corpus]

# Perform LSI tranformation
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2) #Initialize an LSI transformation
corpus_lsi = lsi[corpus_tfidf] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi

lsi.save('model.lsi')
lsi = models.LsiModel.load('model.lsi')

# Print topics
print('LSI topics: ')
lsi.print_topics(2)
print('\n')

print('Corpus LSI ')
with open(corpus_path, 'r') as file:
    for i, line in enumerate(file):
        topic_index = max(corpus_lsi[i], key=lambda item:abs(item[1]))[0]
        print('Topic : ', topic_index)
        print(corpus_lsi[i], " # " + line)
       
    print('\n')
