import pandas as pd 
from collections import defaultdict
import string
from gensim.models import CoherenceModel
import gensim
from pprint import pprint
import spacy, en_core_web_sm
from nltk.stem import PorterStemmer
import os
import json
from gensim.models import Word2Vec
import nltk
import re
import collections
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report
import numpy as np
from collections import defaultdict
import datetime
import csv
#from tfidf_basic_search import *
import gc
import os
# -*- encoding: utf-8 -*-

# Topic model 
class ProcessText:
    def __init__(self, filename):
        """Define varibles."""
        self.path_data = '/disk/data/share/s1690903/pandemic_anxiety/data/'
        self.data = pd.read_csv(self.path_data + filename)
        self.path_result = '/disk/data/share/s1690903/pandemic_anxiety/results/lda_results/'

    def __data_dict(self):
        """Convert df to dictionary. """
        mydict = lambda: defaultdict(mydict)
        data_dict = mydict()

        for pid, text, time in zip(self.data['post_id'], self.data['text'], self.data['time']):
            data_dict[pid]['text'] = text
            data_dict[pid]['time'] = time

        return data_dict

    def simple_preprocess(self):
        """Simple text process: lower case, remove punc. """

        data_dict = self.__data_dict()
        mydict = lambda: defaultdict(mydict)
        cleaned = mydict()
        for k, v in data_dict.items():
            sent = v['text']
            # remove line breaks
            sent = str(sent).replace('\n', '')
            sent = str(sent).replace('â€™', '')
            # lower case and remove punctuation
            sent = str(sent).lower().translate(str.maketrans('', '', string.punctuation))
            cleaned[k]['text'] = sent
            cleaned[k]['time'] = v['time']

        return cleaned

    def extract_entities(self, cleaned_text):
        """get noun trunks for the lda model,
        change noun and verb part to decide what
        you want to use as input for LDA"""
        ps = PorterStemmer()
      
        #find nound trunks
        nlp = en_core_web_sm.load()
        all_extracted = {}
        for k, v in cleaned_text.items():
            #v = v.replace('incubation period', 'incubation_period')
            doc = nlp(v['text'])
            nouns = ' '.join(ps.stem(str(v)) for v in doc if v.pos_ is 'NOUN').split()
            verbs = ' '.join(ps.stem(str(v)) for v in doc if v.pos_ is 'VERB').split()
            adj = ' '.join(str(v) for v in doc if v.pos_ is 'ADJ').split()
            all_w = nouns + verbs + adj
            all_extracted[k] = all_w
      
        return all_extracted

    

class LDATopic:
    def __init__(self, processed_text, topic_num, alpha, eta):
        """Define varibles."""
        self.path = '/disk/data/share/s1690903/pandemic_anxiety/data/'
        self.path_result = '/disk/data/share/s1690903/pandemic_anxiety/results/lda_results/'
        self.text = processed_text
        self.topic_num = topic_num
        self.alpha = alpha
        self.eta = eta

    def get_lda_score_eval(self, dictionary, bow_corpus):
        """LDA model and coherence score."""
        lda_model = gensim.models.ldamodel.LdaModel(bow_corpus, num_topics=self.topic_num, id2word=dictionary, passes=10,  update_every=1, random_state = 300, alpha=self.alpha, eta=self.eta)
        #pprint(lda_model.print_topics())

        # get coherence score
        cm = CoherenceModel(model=lda_model, corpus=bow_corpus, coherence='u_mass')
        coherence = cm.get_coherence()
        print('coherence score is {}'.format(coherence))

        return lda_model, coherence

    def get_score_dict(self, bow_corpus, lda_model_object):
        """
        get lda score for each document
        """
        all_lda_score = {}
        for i in range(len(bow_corpus)):
            lda_score = {}
            for index, score in sorted(lda_model_object[bow_corpus[i]], key=lambda tup: -1*tup[1]):
                lda_score[index] = score
                od = collections.OrderedDict(sorted(lda_score.items()))
            all_lda_score[i] = od
        return all_lda_score


    def topic_modeling(self):
        """Get LDA topic modeling."""
        # generate dictionary
        dictionary = gensim.corpora.Dictionary(self.text.values())
        bow_corpus = [dictionary.doc2bow(doc) for doc in self.text.values()]
        # modeling
        model, coherence = self.get_lda_score_eval(dictionary, bow_corpus)

        lda_score_all = self.get_score_dict(bow_corpus, model)

        all_lda_score_df = pd.DataFrame.from_dict(lda_score_all)
        all_lda_score_dfT = all_lda_score_df.T
        all_lda_score_dfT = all_lda_score_dfT.fillna(0)

        return model, coherence, all_lda_score_dfT, bow_corpus

    def format_topics_sentences(self, ldamodel, corpus):
        # Init output, get dominant topic for each document 
        sent_topics_df = pd.DataFrame()

        # Get main topic in each document
        for i, row_list in enumerate(ldamodel[corpus]):
            row = row_list[0] if ldamodel.per_word_topics else row_list            
            # print(row)
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = ldamodel.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
                else:
                    break
        
        sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
    

        #Add original text to the end of the output
        #contents = pd.Series(data['text'])     
        #contents = contents.reset_index()
        #sent_topics_df = sent_topics_df.reset_index()
        
        return sent_topics_df

    def get_ids_from_selected(self, text):
        """Get unique id from text """
        id_l = []
        for k, v in text.items():
            id_l.append(k)
            
        return id_l


def selected_best_LDA(path, text, num_topic, domTname):
        """Select the best lda model with extracted text """
        # convert data to dictionary format

        file_exists = os.path.isfile(path + 'lda_result_test.csv')
        f = open(path + 'lda_result_test.csv', 'a', encoding='utf-8')
        writer_top = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        if not file_exists:
            writer_top.writerow(['a'] + ['b'] + ['coherence'] + ['time'] + ['topics'] + ['num_topics'] )

        # optimized alpha and beta
        alpha = [0.1, 0.3, 0.5, 0.7, 0.9]
        beta = [0.1, 0.3, 0.5, 0.7, 0.9]

        # alpha = [0.3]
        # beta = [0.3]

        mydict = lambda: defaultdict(mydict)
        cohere_dict = mydict()
        for a in alpha:
            for b in beta:
                lda = LDATopic(text, num_topic, a, b)
                model, coherence, scores, corpus = lda.topic_modeling()
                cohere_dict[coherence]['a'] = a
                cohere_dict[coherence]['b'] = b

                
        # sort result dictionary to identify the best a, b
        # select a,b with the largest coherence score 
        sort = sorted(cohere_dict.keys())[0] 
        a = cohere_dict[sort]['a']
        b = cohere_dict[sort]['b']
        
        # run LDA with the optimized values
        lda = LDATopic(text, num_topic, a, b)
        model, coherence, scores_best, corpus = lda.topic_modeling()
        #pprint(model.print_topics())

        #f = open(path + 'result/lda_result.csv', 'a')
        result_row = [[a, b, coherence, str(datetime.datetime.now()), model.print_topics(), num_topic]]
        writer_top.writerows(result_row)

        f.close()
        gc.collect()

        #select merge ids with the LDA topic scores 
        #store result model with the best score
        id_l = lda.get_ids_from_selected(text)
        scores_best['post_id'] = id_l

        # get topic dominance
        #t = LDATopicModel()
        df_topic_sents_keywords = lda.format_topics_sentences(model, corpus)
        df_dominant_topic = df_topic_sents_keywords.reset_index()
        #print(df_dominant_topic.shape)

        sent_topics_df = pd.concat([df_dominant_topic, scores_best], axis=1)
        #sent_topics_df = sent_topics_df[['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']]
        sent_topics_df.to_csv(path + 'dominance_{}.csv'.format(domTname))

        return sent_topics_df


def loop_lda(inputfile):
    """ 
    inputfile: files for running LDA
    """
    pt = ProcessText(inputfile)
    cleaned_text = pt.simple_preprocess()
    entities = pt.extract_entities(cleaned_text)
    sent_topics_df = selected_best_LDA(pt.path_result, entities, 2, 'test')


loop_lda('posts/test.csv')

# pt = ProcessText('posts/test.csv')
# cleaned_text = pt.simple_preprocess()


pt = ProcessText('posts/test.csv')
cleaned_text = pt.simple_preprocess()
entities = pt.extract_entities(cleaned_text)




























