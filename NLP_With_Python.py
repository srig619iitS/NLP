####################### Regular Expression   ####################################################

import re

my_string = 'Lets write RegEx!  Wont that be fun?  I sure think so.  Can you find 4 sentences?  Or perhaps, all 19 words?'

#Split my_string on each sentence ending. To do this:Write a pattern called sentence_endings
sentence_endings = r"[.?!]"
#print(re.split(sentence_endings, my_string))

#Find all capitalized words in my_string and print the result
Capital_words = r"[A-Z]\w+"
#print(re.findall(Capital_words, my_string))

# Split my_string on spaces and print the result
Space_pattern = r"[\s+]"
#print(re.split(Space_pattern, my_string))

# Find all digits in my_string and print the result
digit_patten = r"[\d+]"
#print(re.findall(digit_patten, my_string))


####################### Word tokenization with NLTK   ####################################################
import pandas as pd
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.tokenize import regexp_tokenize,TweetTokenizer


#scene_one = pd.read_table(r'C:\Users\sg_cl\Desktop\Pythonproject\Final exam\ML\scene_one.txt', header = None)
scene_one = open(r'C:\Users\sg_cl\Desktop\Pythonproject\Final exam\ML\scene_one.txt')
scene_one = scene_one.read().replace("\n"," ")
#scene_one.close()
#print(type(scene_one))

#Split scene_one into sentences: sentences
sentences = sent_tokenize(scene_one)
#print(sentences[6])

## Use word_tokenize to tokenize the sixth sentence: tokenized_sen
words = word_tokenize(sentences[6])
#print(words)

## Make a set of unique tokens in the sixth sentence: unique_tokens
unique_tokens = set(word_tokenize(sentences[6]))
#print(unique_tokens)

####################### More Reg EX with Re.search() & re.match()  #############


# Search for the first occurrence of "coconuts" in scene_one: match
#print(re.match('coconuts',scene_one))
#print(re.search('coconuts',scene_one))

#Search for the occurrence of specific_tokens in scene_one
#print(re.search(r"\[.*\]",scene_one))

#Search for the occurrence of specific_tokens in scene_one
#print(re.search(r"[\w]:+", sentences[3]))

######################### Advanced Reg EX with NLTK toenization ############################################

tweets = "['This is the best #nlp exercise ive found online! #python', '#NLP is super fun! <3 #learning', 'Thanks @datacamp :) #nlp #python']"

# Define a regex pattern to find hashtags: pattern1
pattern1 = r"[\#]\w+"
hashtags = regexp_tokenize(tweets, pattern1)
#print(hashtags)

# Write a pattern that matches both mentions (@) and hashtags
pattern2 = r"([@|#][a-z]+)"
mentions_hashtags = regexp_tokenize(tweets, pattern2)
#print(mentions_hashtags)

#Use the TweetTokenizer to tokenize all tweets into one list

tknzr = TweetTokenizer(preserve_case=False, reduce_len=False, strip_handles=False)
all_tokens = list(tknzr.tokenize(tweets))
#print(all_tokens)


###########  Charting Practice ############################################################
#Load Holy_grail
holy_grail = open(r'C:\Users\sg_cl\Desktop\Pythonproject\Final exam\ML\holy_grail.txt')
holy_grail = holy_grail.read().replace("\n"," ")
'''
# Split the script into lines: lines
lines = holy_grail.split('\n')

# Replace all script lines for speaker
pattern = "[A-Z]{2,}(\s)?(#\d)?([A-Z]{2,})?:"
lines = [re.sub(pattern, "",l) for l in lines]

# Tokenize each line: tokenized_lines
tokenized_lines = [regexp_tokenize(s,"\W+") for s in lines]
print(tokenized_lines)

# Make a frequency list of lengths: line_num_words
line_num_words = [len(t) for t in tokenized_lines]
print(line_num_words)
'''

############# Building a Counter with Bag of words ########################################
####### Preprocessing
import collections
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#####Practice 1
Tokenization = [w for w in word_tokenize(scene_one.lower()) if w.isalpha()]
stop_words = [t for t in Tokenization if t not in stopwords.words('english')]
bow = Counter(stop_words)
#print(bow.most_common(5))

#####Practice 2
# Retain alphabetic words: alpha_only
alpha_only = [w for w in word_tokenize(scene_one.lower()) if w.isalpha()]
#[t for t in scene_one if t.isalpha()]

# Remove all stop words: no_stops
no_stops = [t for t in Tokenization if t not in stopwords.words('english')]
#[l for l in alpha_only if l not in stopwords.words('english')]

# Instantiate the WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

# Lemmatize all tokens into a new list: lemmatized
lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops]

#Bag of words
bow_N = Counter(lemmatized)
#print(bow_N.most_common(5))


############################ Gensim: Dictionary & Corpora ###################################################

from gensim.corpora.dictionary import Dictionary
scene_one = scene_one.lower()
sentences_1 = sent_tokenize(scene_one)
Tokenization_1 =  [word_tokenize(f) for f in sentences_1 ]
stop_words_1 = [t for t in Tokenization_1 if t not in stopwords.words('english')]

print(sentences_1[0:2])

dictionary = Dictionary(stop_words_1)
print(dictionary.token2id)

#Creating Corpus
corpus = [dictionary.doc2bow(doc) for doc in stop_words_1]
#print(corpus[0:2])


############## Gensim Bag of words ####################################################
import itertools


#   save second document
doc = corpus[0:2]
#print(doc)
#   Sort the doc for frequency: bow_doc
bow_doc = sorted(doc, key = lambda w : w[1], reverse = True)
print(bow_doc)

# print top 4 words of document

for word_id, word_count in itertools.chain.from_iterable(doc):
#in bow_doc.items():
    print(dictionary.get(word_id),word_count)

total_word_count = collections.defaultdict(int)
for word_id, word_count in itertools.chain.from_iterable(corpus):
    total_word_count[word_id] += word_count
print(total_word_count)

'''
############################ Gensim: Dictionary & Corpora ###################################################

from gensim.corpora.dictionary import Dictionary
scene_one = scene_one.lower()
Tokenization_1 =  [word_tokenize(scene_one) ] 
stop_words_1 = [t for t in Tokenization_1 if t not in stopwords.words('english')]
#print(Tokenization_1)
#print(stop_words_1)


dictionary = Dictionary(stop_words_1)
#print(dictionary.token2id)

#Creating Corpus
corpus = [dictionary.doc2bow(doc) for doc in stop_words_1]
#print(corpus)


############## Gensim Bag of words ####################################################


#   save second document
doc = corpus[0]
#print(doc)

#   Sort the doc for frequency: bow_doc
bow_doc = sorted(doc, key = lambda w : w[1], reverse = True)
#print(bow_doc)

# print top 4 words of document
for word_id, word_count in bow_doc[:5]:
    print(dictionary.get(word_id),word_count)
'''
######## TfIDF with Genism ################################################################

from gensim.models.tfidfmodel import TfidfModel

# Create a new TfidfModel using the corpus: tfidf
tfidf = TfidfModel(corpus)

# Calculate the tfidf weights of doc: tfidf_weights
tfidf_weights = tfidf[doc]

# Print the first five weights
for t in tfidf_weights:
    print(t)


