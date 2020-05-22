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
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report
import numpy as np
import datetime 
from datetime import datetime 
import csv
#from tfidf_basic_search import *
import gc
import os
# type hints
from typing import Dict, Tuple, Sequence
import typing
from ruamel import yaml

# -*- encoding: utf-8 -*-

def load_experiment(path_to_experiment):
    """load experiment"""
    data = yaml.safe_load(open(path_to_experiment))
    return data


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

    def simple_preprocess(self) -> typing.Dict[str, str]: 
        """Simple text process: lower case, remove punc. """

        data_dict = self.__data_dict()
        mydict = lambda: defaultdict(mydict)
        cleaned = mydict()
        for k, v in data_dict.items():
            sent = v['text']
            # remove line breaks
            sent = str(sent).replace('\n', '')
            sent = str(sent).replace(' â€™', 'i') 
            sent = str(sent).replace('\u200d', '')
            # lower case and remove punctuation
            sent = str(sent).lower().translate(str.maketrans('', '', string.punctuation))
            cleaned[k]['text'] = sent
            cleaned[k]['time'] = v['time']

        return cleaned

    def extract_entities(self, cleaned_text: typing.Dict[str, str]) -> typing.Dict[str, str]:
        """get noun, verbs and adj for the lda model,
        change the parts of speech to decide what
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


    def split_timeline(self, simple_prepro_text:typing.Dict[str, str], start:str, end:str) -> typing.Dict[str, str]: 
        """Here we segment the dataset according to time. 
        start date = (m/d/y)
        """
        mydict = lambda: defaultdict(mydict)
        target_time = mydict()

        for k, v in simple_prepro_text.items():
            time = datetime.strptime(v['time'], '%m/%d/%Y/%H:%M:%S').date()
            start_day = datetime.strptime(start, '%m/%d/%Y').date()
            end_day = datetime.strptime(end, '%m/%d/%Y').date()

            if (time > start_day and time < end_day):
                target_time[k]['text'] = v['text']
                target_time[k]['time'] = v['time']

        return target_time


class LDATopic:
    def __init__(self, processed_text: typing.Dict[str, str], topic_num: int, alpha: int, eta:int):
        """Define varibles."""
        self.path = '/disk/data/share/s1690903/pandemic_anxiety/data/'
        self.path_result = '/disk/data/share/s1690903/pandemic_anxiety/results/lda_results/'
        self.text = processed_text
        self.topic_num = topic_num
        self.alpha = alpha
        self.eta = eta

    def get_lda_score_eval(self, dictionary: typing.Dict[str, str], bow_corpus) -> list:
        """LDA model and coherence score."""
        lda_model = gensim.models.ldamodel.LdaModel(bow_corpus, num_topics=self.topic_num, id2word=dictionary, passes=10,  update_every=1, random_state = 300, alpha=self.alpha, eta=self.eta)
        #pprint(lda_model.print_topics())

        # get coherence score
        cm = CoherenceModel(model=lda_model, corpus=bow_corpus, coherence='u_mass')
        coherence = cm.get_coherence()
        print('coherence score is {}'.format(coherence))

        return lda_model, coherence

    def get_score_dict(self, bow_corpus, lda_model_object) -> pd.DataFrame:
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
        
        return sent_topics_df

    def get_ids_from_selected(self, text: typing.Dict[str, str]):
        """Get unique id from text """
        id_l = []
        for k, v in text.items():
            id_l.append(k)
            
        return id_l


def selected_best_LDA(path, text: typing.Dict[str, str], num_topic:int, domTname:str, subreddit):
        """Select the best lda model with extracted text 
        text: entities dictionary
        domTname:file name for the output
        """

        # convert data to dictionary format

        file_exists = os.path.isfile(path + 'lda_result_{}_{}.csv'.format(domTname, subreddit))
        f = open(path + 'lda_result_{}_{}.csv'.format(domTname, subreddit), 'a', encoding='utf-8-sig')
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
        result_row = [[a, b, coherence, str(datetime.now()), model.print_topics(), num_topic]]
        writer_top.writerows(result_row)

        f.close()
        gc.collect()

        #select merge ids with the LDA topic scores 
        #store result model with the best score
        id_l = lda.get_ids_from_selected(text)
        scores_best['post_id'] = id_l

        # get topic dominance
        df_topic_sents_keywords = lda.format_topics_sentences(model, corpus)
        df_dominant_topic = df_topic_sents_keywords.reset_index()

        sent_topics_df = pd.concat([df_dominant_topic, scores_best], axis=1)
        sent_topics_df.to_csv(path + 'dominance_{}_{}_{}.csv'.format(domTname, num_topic, subreddit), encoding='utf-8-sig')

        return sent_topics_df


def loop_lda(inputfile):
    """ 
    inputfile: files for running LDA
    """
    pt = ProcessText(inputfile)
    cleaned_text = pt.simple_preprocess()
    entities = pt.extract_entities(cleaned_text)
    sent_topics_df = selected_best_LDA(pt.path_result, entities, 2, 'test')


#loop_lda('posts/test.csv')

def get_dominant_topic(topic_df):
    """ """

    dt = topic_df['Dominant_Topic'].value_counts().to_frame()
    dt['num'] = dt.index
    # get the most dominant topic
    dt_num = int(dt['num'].head(1))
    print(dt_num)
    topic_kw = topic_df['Topic_Keywords'][topic_df['Dominant_Topic'] == dt_num][0]
    return dt_num, topic_kw



def get_topic_season(subreddit, year: int, num_topic: int) -> pd.DataFrame:
    """get topics according to season timeline 
    result saved in results/lda_results/
    """
    pt = ProcessText('posts/{}_postids_posts.csv'.format(subreddit))
    cleaned_text = pt.simple_preprocess()

    # here we can set the seasons
    spring = pt.split_timeline(cleaned_text, '3/1/{}'.format(year), '5/31/{}'.format(year))
    summer = pt.split_timeline(cleaned_text, '6/1/{}'.format(year), '8/31/{}'.format(year))
    fall = pt.split_timeline(cleaned_text, '9/1/{}'.format(year), '11/30/{}'.format(year))
    winter = pt.split_timeline(cleaned_text, '12/1/{}'.format(year), '2/29/{}'.format(year))
    
    # run lda for each season 
    if len(spring) > 1:
        entities = pt.extract_entities(spring)
        sent_topics_spring = selected_best_LDA(pt.path_result, entities, num_topic, 'spring_{}'.format(year), subreddit)
        dt_num_spring, topic_kw_spring = get_dominant_topic(sent_topics_spring)

    if len(summer) > 1:
        entities = pt.extract_entities(summer)
        sent_topics_summer = selected_best_LDA(pt.path_result, entities, num_topic, 'summer_{}'.format(year), subreddit)
        dt_num_summer, topic_kw_summer = get_dominant_topic(sent_topics_summer)
    else:
        dt_num_summer = None
        topic_kw_summer = None

    if len(fall) > 1:
        entities = pt.extract_entities(fall)
        sent_topics_fall = selected_best_LDA(pt.path_result, entities, num_topic, 'fall_{}'.format(year), subreddit)
        dt_num_fall, topic_kw_fall = get_dominant_topic(sent_topics_fall)
    else:
        dt_num_fall = None
        topic_kw_fall = None

    if len(winter) > 1:
        entities = pt.extract_entities(winter)
        sent_topics_winter = selected_best_LDA(pt.path_result, entities, num_topic, 'winter_{}'.format(year),subreddit)
        dt_num_winter, topic_kw_winter = get_dominant_topic(sent_topics_winter)
    else:
        dt_num_winter = None
        topic_kw_winter = None
    
    # save most dominant topics for the four season results so that we can put it on the big table
    
    f = open(pt.path_result + 'domance_output_{}_{}_{}.csv'.format(subreddit, year, num_topic), 'w', encoding='utf-8-sig')
    writer_top = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    writer_top.writerow(['topic_number_spring'] + ['keywords_spring'] + ['topic_number_summer'] + ['keywords_summer'] + ['topic_number_fall'] + ['keywords_fall'] + ['topic_number_winter'] + ['keywords_winter'])
    result_row = [[dt_num_spring, topic_kw_spring, dt_num_summer, topic_kw_summer, dt_num_fall, topic_kw_fall, dt_num_winter, topic_kw_winter]]
    writer_top.writerows(result_row)
    f.close()

    return sent_topics_spring


#sent_topics_spring = get_topic_season('HealthAnxiety', 2020, 10) #year, num_topic

#again, we can totally loop through a list of subreddit names
evn_path = '/disk/data/share/s1690903/pandemic_anxiety/evn/'
evn = load_experiment(evn_path + 'experiment.yaml')
subreddits = evn['subreddits']['subs']
for sub in subreddits:
    sent_topics_spring = get_topic_season(sub, 2020, 10) #year, num_topic
























