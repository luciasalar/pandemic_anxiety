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
import gc
import os
# type hints
from typing import Dict, Tuple, Sequence
import typing
from ruamel import yaml
import contractions
from gensim.sklearn_api import LdaTransformer
import lda

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

    def __processed_data(self):
        """Remove duplicates and deleted posts and nan"""
        data = self.data.drop_duplicates(subset='post_id', keep='first', inplace=False)
        data = data[~data['text'].isin(['[removed]'])]
        data = data[~data['text'].isin(['[deleted]'])]
        data = data[~data['text'].str.lower().isin(['deleted'])]
        data = data[~data['text'].str.lower().isin(['removed'])]
        data['text'].replace('', np.nan, inplace=True)
        data = data.dropna(subset=['text'])
        return data


    def data_dict(self):
        """Convert df to dictionary. """ 

        data = self.__processed_data()
        mydict = lambda: defaultdict(mydict)
        data_dict = mydict()

        if 'time' in data.columns:
            for pid, text, time in zip(data['post_id'], data['text'], data['time']):
                data_dict[pid]['text'] = text
                data_dict[pid]['time'] = time
        else:
            for pid, text in zip(data['post_id'], data['text']):
                data_dict[pid]['text'] = text

        return data_dict

    def simple_preprocess(self) -> typing.Dict[str, str]: 
        """Simple text process: lower case, remove punc. """

        data_dict = self.data_dict()
        mydict = lambda: defaultdict(mydict)
        cleaned = mydict()
        #count = 0
        for k, v in data_dict.items():
            sent = v['text']
            # url
            sent = re.sub(r'http\S+', '', sent)
            #sent = re.sub(r'^https?:\/\/.*[\r\n]*', '', sent, flags=re.MULTILINE)
            #remove contractions
            sent = contractions.fix(sent)
            # remove line breaks
            sent = str(sent).replace('\n', ' ')
            #sent = str(sent).replace(' â€™', 'i')
            sent = str(sent).replace('\u200d', ' ')
            # lower case and remove punctuation
            sent = str(sent).lower().translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
            if len(sent.split()) > 2:
                cleaned[k]['text'] = sent

            if v['time'] is not None:
                cleaned[k]['time'] = v['time']
                #print(cleaned[k]['time'])
            # count = count + 1
            # if count == 1000:
            #     break

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
            if bool(v['text']) == True:
            #v = v.replace('incubation period', 'incubation_period')
                doc = nlp(v['text'])
                nouns = ' '.join(ps.stem(str(v)) for v in doc if v.pos_ is 'NOUN').split()
                verbs = ' '.join(ps.stem(str(v)) for v in doc if v.pos_ is 'VERB').split()
                adj = ' '.join(str(v) for v in doc if v.pos_ is 'ADJ').split()
                #noun_tr = ' '.join(str(v) for v in doc.noun_chunks).split()
                all_w = nouns + adj + verbs
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

    def get_lda_score_eval2(self, dictionary: typing.Dict[str, str], bow_corpus) -> list:
        """LDA model and coherence score."""
        # lda_model = gensim.models.ldamodel.LdaModel(bow_corpus, num_topics=self.topic_num, id2word=dictionary, passes=10,  update_every=1, random_state = 300, alpha=self.alpha, eta=self.eta)
        # the trained model
        lda_model = LdaTransformer(num_topics=self.topic_num, id2word=dictionary, iterations=10, random_state=300, alpha=self.alpha, eta=self.eta, scorer= 'mass_u')

        #The topic distribution for each input document.
        docvecs = lda_model.fit_transform(bow_corpus)
        #pprint(lda_model.print_topics())

        # get coherence score
        #cm = CoherenceModel(model=lda_model, corpus=bow_corpus, coherence='u_mass')
        #coherence = cm.get_coherence()
        #print('coherence score is {}'.format(coherence))

        return lda_model, docvecs

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

    def topic_modeling2(self):
        """Get LDA topic modeling."""
        # generate dictionary
        dictionary = gensim.corpora.Dictionary(self.text.values())
        bow_corpus = [dictionary.doc2bow(doc) for doc in self.text.values()]
        # modeling
        model = self.get_lda_score_eval2(dictionary, bow_corpus)

        return model

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


def selected_best_LDA(path, text: typing.Dict[str, str], num_topic:int, domTname:str, subreddit = None):
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
        # beta = [0.9]

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



def get_dominant_topic(topic_df):
    """Get the most dominant topic."""

    dt = topic_df['Dominant_Topic'].value_counts().to_frame()
    dt['num'] = dt.index
    # get the most dominant topic
    dt_num = int(dt['num'].head(1))
    print(dt_num)
    topic_kw = topic_df['Topic_Keywords'][topic_df['Dominant_Topic'] == dt_num]
    topic_kw = topic_kw.iloc[0]
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
    winter = pt.split_timeline(cleaned_text, '12/1/{}'.format(year), '2/28/{}'.format(year + 1))
    
   #run lda for each season 
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
    
    #save most dominant topics for the four season results so that we can put it on the big table
    
    f = open(pt.path_result + 'domance_output_{}_{}_{}.csv'.format(subreddit, year, num_topic), 'w', encoding='utf-8-sig')
    writer_top = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    writer_top.writerow(['topic_number_spring'] + ['keywords_spring'] + ['topic_number_summer'] + ['keywords_summer'] + ['topic_number_fall'] + ['keywords_fall'] + ['topic_number_winter'] + ['keywords_winter'])
    result_row = [[dt_num_spring, topic_kw_spring, dt_num_summer, topic_kw_summer, dt_num_fall, topic_kw_fall, dt_num_winter, topic_kw_winter]]
    writer_top.writerows(result_row)
    f.close()

def get_topic_covid_timeline(subreddit, year: int, num_topic: int) -> pd.DataFrame:
    """get topics according to season timeline 
    result saved in results/lda_results/
    """
    pt = ProcessText('posts/{}_postids_posts.csv'.format(subreddit))
    cleaned_text = pt.simple_preprocess()

    # here we can set the seasons
    precovid = pt.split_timeline(cleaned_text, '1/1/{}'.format(year), '12/31/{}'.format(year))
    precovid2 = pt.split_timeline(cleaned_text, '1/1/{}'.format(year - 1), '12/31/{}'.format(year-1))
    covid = pt.split_timeline(cleaned_text, '2/1/2020', '8/30/2020')
    precovid.update(precovid2)
 
   #run lda for each period
    if len(precovid) > 1:
        entities = pt.extract_entities(precovid)
        sent_topics_precovid = selected_best_LDA(pt.path_result, entities, num_topic, 'precovid_{}_{}'.format(year, year-1), subreddit)
        dt_num_precovid, topic_kw_precovid = get_dominant_topic(sent_topics_precovid)
    else:
        dt_num_precovid = None
        topic_kw_precovid = None

    if len(covid) > 1:
        entities = pt.extract_entities(covid)
        sent_topics_covid = selected_best_LDA(pt.path_result, entities, num_topic, 'covid', subreddit)
        dt_num_covid, topic_kw_covid = get_dominant_topic(sent_topics_covid)
    else:
        dt_num_covid = None
        topic_kw_covid = None

    #save most dominant topics for the four season results so that we can put it on the big table
    
    f = open(pt.path_result + 'domance_output_{}_{}_{}_{}.csv'.format(subreddit, year, year-1, num_topic), 'w', encoding='utf-8-sig')
    writer_top = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    writer_top.writerow(['topic_number_precovid'] + ['keywords_precovid'] + ['topic_number_covid'] + ['keywords_covid'])
    result_row = [[dt_num_precovid, topic_kw_precovid, dt_num_covid, topic_kw_covid]]
    writer_top.writerows(result_row)
    f.close()
    

def get_topic_month_timeline(subreddit, year: int, num_topic: int) -> pd.DataFrame:
    """get topics according to season timeline 
    result saved in results/lda_results/
    """
    pt = ProcessText('posts/{}_postids_posts.csv'.format(subreddit))
    cleaned_text = pt.simple_preprocess()

    # here we can set the seasons
    covidFeb = pt.split_timeline(cleaned_text, '2/1/2020', '3/31/2020')
    covidApr = pt.split_timeline(cleaned_text, '4/1/2020', '4/30/2020')
    covidMay = pt.split_timeline(cleaned_text, '5/1/2020', '5/31/2020')
   
 
   #run lda for each period
    entities1 = pt.extract_entities(covidFeb)
    sent_topics_covidFeb = selected_best_LDA(pt.path_result, entities1, num_topic, 'covid_Feb_15', subreddit)
    dt_num_covidFeb, topic_kw_covidFeb = get_dominant_topic(sent_topics_covidFeb)

    entities2 = pt.extract_entities(covidApr)
    sent_topics_covidApr = selected_best_LDA(pt.path_result, entities2, num_topic, 'covid_Apr_15', subreddit)
    dt_num_covidApr, topic_kw_covidApr = get_dominant_topic(sent_topics_covidApr)

    entities3 = pt.extract_entities(covidMay)
    sent_topics_covidMay = selected_best_LDA(pt.path_result, entities3, num_topic, 'covid_May_15', subreddit)
    dt_num_covidMay, topic_kw_covidMay = get_dominant_topic(sent_topics_covidMay)
   

    # #save most dominant topics for the four season results so that we can put it on the big table
    
    f = open(pt.path_result + 'domance_output_{}_support_{}.csv'.format(subreddit, num_topic), 'w', encoding='utf-8-sig')
    writer_top = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    writer_top.writerow(['topic_number_covidFeb'] + ['keywords_covidFeb'] + ['topic_number_covidApr'] + ['keywords_covidApr'] + ['topic_number_covidMay'] + ['keywords_covidMay'])
    result_row = [[dt_num_covidFeb, topic_kw_covidFeb, dt_num_covidApr, topic_kw_covidApr, dt_num_covidMay, topic_kw_covidMay]]
    writer_top.writerows(result_row)
    f.close()



if __name__ == "__main__":

    #get_topic_season('Anxiety', 2020, 10) #year, xnum_topiclda_
#again, we can totally loop through a list of subreddit names
    evn_path = '/disk/data/share/s1690903/pandemic_anxiety/evn/'
    evn = load_experiment(evn_path + 'experiment.yaml')
    subreddits = evn['subreddits']['subs']
    for sub in subreddits:
        #covidApr, covidMay = get_topic_month_timeline(sub, 2020, 15) #year, num_topic
        get_topic_covid_timeline(sub, 2019, 15)


    # evn_path = '/disk/data/share/s1690903/pandemic_anxiety/evn/'
    # evn = load_experiment(evn_path + 'experiment.yaml')
    # subreddits = evn['subreddits']['subs']
    # get_topic_covid_timeline('HealthAnxiety', 2019, 15)  #here define precovid or covid dataset
# data = pd.read_csv('/disk/data/share/s1690903/pandemic_anxiety/data/posts/OCD_postids_posts.csv')
















