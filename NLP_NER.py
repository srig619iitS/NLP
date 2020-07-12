###### Named Entity Recognition with nltk ##############################
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize


nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')


article = open(r'C:\Users\sg_cl\Desktop\Pythonproject\Final exam\ML\scene_one.txt')
#open(r'C:\Users\sg_cl\Desktop\Pythonproject\Final exam\ML\article.txt', encoding="utf8")
article = article.read().replace("\n"," ")
#print(article)

sentence = nltk.sent_tokenize(article)
# Tokenize each sentence into words: token_sentences
token = [nltk.word_tokenize(f) for f in sentence]
#print(token)

# Tag each tokenized sentence into parts of speech: pos_sentences
pos_tag = [nltk.pos_tag(t) for t in token]

# Create the named entity chunks: chunked_sentences
chunked_sentences = nltk.ne_chunk_sents(pos_tag, binary = True)
#print(list(chunked_sentences))

# Test for stems of the tree with 'NE' tags
for sent in chunked_sentences:
    for chunk in sent:
        if hasattr(chunk, "label") and chunk.label() == "NE":
           print(chunk)
            

############# Charting practice #################################################

import collections

# Create the defaultdict: ner_categories
ner_categories = collections.defaultdict(int)

#fill up the dictionary with values
for sent in chunked_sentences:
    for chunk in sent:
        if hasattr(chunk, 'label'):
            print(ner_categories) 
            ner_categories[chunk.label()] += 1
#print(ner_categories)

#Create a list called labels from the keys of dictionary(ner_categories)
labels = list(ner_categories.keys())
#print(labels)

#Create list of values
values = [ner_categories.get(v) for v in labels]

########## spacy ##################################################################

import spacy

#instantiate english model
nlp = spacy.load("en_core_web_sm")

#create the doc
doc = nlp(article)

#print all of the found entities and their labels
for x in doc.ents:
    print(x.label_,x.text)
    
############ Polyglot ##############################################################
from polyglot.text import Text

test = 'telediario dijeron que tendríamos un nuevo día feriado nacional.'

ptext = Text(test)

# Print each of the entities found
for ent in ptext.entities:
    print(ent)
    
# Print the type of ent
print(type(ent))



