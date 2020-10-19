#!/usr/bin/env python
import pandas as pd
import numpy as np
import re
import sklearn
from sklearn.model_selection import train_test_split

# import nltk
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.metrics import classification_report
from collections import defaultdict

# the features
import readability 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from lda_topic import *

from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

# the classifiers
# from sklearn.metrics import confusion_matrix, f1_score, precision_score,\
# recall_score, confusion_matrix, classification_report, accuracy_score 
# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.preprocessing import FunctionTransformer
# from sklearn.feature_selection import SelectFromModel
# from sklearn.linear_model import LogisticRegression

# from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.dummy import DummyClassifier

from ruamel import yaml
# import datetime
# import matplotlib.pyplot as plt
import os
import time
# import logging
import csv
import gc
import datetime
import glob
from typing import Dict, Tuple, Sequence
import typing



#predict anxiety and fear entity
# we use combined text and title as input  combine_text.csv
# combine columns in line 393
# need to change lda feature when input is changed, use the optimized function to get the best lda model
# also change LIWC  testliwc.csv check /annotations for feature files


def load_experiment(path_to_experiment):
    #load experiment 
    data = yaml.safe_load(open(path_to_experiment))
    return data



class Read_raw_data:
    def __init__(self):
        self.path = '/disk/data/share/s1690903/pandemic_anxiety/data/annotations/post_anno/'

    def read_all_files(self) -> pd.DataFrame:
        """ Read all the annotation files. """

        all_files = []
        for file in glob.glob(self.path + "*.csv"):
            file_pd = pd.read_csv(file)
            all_files.append(file_pd)

        all_files_pd = pd.concat(all_files)

        # Drop those without annotations.
        all_files_pd = all_files_pd[all_files_pd['anxiety'].notna()]

        # Replace Nan with 0.
        all_files_pd = all_files_pd.replace(np.nan, 0)
        liwc_file = all_files_pd[['title', 'text', 'post_id']]
        liwc_file.to_csv('/disk/data/share/s1690903/pandemic_anxiety/data/annotations/test.csv')

        return all_files_pd

    def combine_columns(self, newcol, col1, col2, col3=None):
        """Combine column labels """
        all_files = self.read_all_files()
        if col3 == None:
            all_files[newcol] = all_files[col1] + all_files[col2]
            all_files.loc[all_files[newcol] > 1, newcol] = 1
        else:
            all_files[newcol] = all_files[col1] + all_files[col2] + all_files[col3]
            all_files.loc[all_files[newcol] > 1, newcol] = 1

        return all_files




class PrepareData:

    def __init__(self, raw_data, labelcol):
        '''define the main path'''
        self.path = '/disk/data/share/s1690903/pandemic_anxiety/data/annotations/'

        self.file = raw_data
        # join the title and text 
        self.file['text'] = self.file['text'].str.cat(self.file['title'], sep=" ")
        self.labelcol = labelcol

    def save_combined_text(self):
        self.file.to_csv(self.path + 'combined_text.csv')


    def convert_text_dict(self, file):
        """Convert text to dictionary format. """

        mydict = lambda: defaultdict(mydict)
        text_dict = mydict()

        for text, post_id in zip(file['text'], file['post_id']):
            text_dict[post_id] = text

        return text_dict


    def get_readability(self, text_dict):
        """Get readability """

        mydict = lambda: defaultdict(mydict)
        read_dict = mydict()

        for k, text in text_dict.items():
            result = readability.getmeasures(text, lang='en')
            readability_score = result['readability grades']['FleschReadingEase']

            read_dict[k] = readability_score

        return read_dict


    def get_sentiment(self, text_dict, senti):
        """Get sentiment scores """

        mydict = lambda: defaultdict(mydict)
        senti_dict = mydict()
        analyzer = SentimentIntensityAnalyzer()

        for k, text in text_dict.items(): 
            result = analyzer.polarity_scores(text) 

            senti_dict[k] = result[senti]

        return senti_dict
            

    def get_liwc(self)-> pd.DataFrame:
        """Merge all the features. The LIWC runs both text and title"""

        # Read LIWC features.
        liwc = pd.read_csv(self.path + 'test_liwc.csv', encoding="ISO-8859-1")
        liwc.columns = [str(col) + '_liwc' for col in liwc.columns]
        liwc = liwc.rename(columns={"post_id_liwc": "post_id", "text_liwc": "text", "title_liwc": "title"})

        return liwc

    def convert_dict_df(self, dictionary, newcolname):
        """Covert dictionary to df."""

        new_df = pd.DataFrame.from_dict(dictionary, orient='index')
        new_df['post_id'] = new_df.index
        new_df.columns = [newcolname, 'post_id']

        return new_df

    def optimize_lda(self):
        """Check the optimized lda scores """
        pt = ProcessText('annotations/combined_text.csv')
        cleaned_text = pt.simple_preprocess()
        entities = pt.extract_entities(cleaned_text)
        
        # get the optimized parameter 
        sent_topics_df = selected_best_LDA(pt.path_result, entities, 10, 'lda_test.csv')


    def merge_features(self)-> pd.DataFrame:
        """Merge features with labels """
        
        # get liwc
        liwc = self.get_liwc()

        # get readability
        text_dict = self.convert_text_dict(liwc)
        readability_score = self.get_readability(text_dict)
        readability_df = self.convert_dict_df(readability_score, 'FleschReadingEase')    

        # get sentiment
        neg = self.get_sentiment(text_dict, 'neg')
        neg_df = self.convert_dict_df(neg, 'neg')
        pos = self.get_sentiment(text_dict, 'pos')
        pos_df = self.convert_dict_df(pos, 'pos')
        neu = self.get_sentiment(text_dict, 'neu')
        #neu_df = self.convert_dict_df(neu, 'neu')
        senti = neg_df.merge(pos_df, on='post_id')
        #senti = senti.merge(neu_df, on='post_id')

        # add feature tag
        senti.columns = [str(col) + '_senti' for col in senti.columns]
        senti = senti.rename(columns={"post_id_senti": "post_id"})

        # get topic modeling
        #self.optimize_lda()  # optimize model and generate lda_feature.csv
        lda = pd.read_csv('/disk/data/share/s1690903/pandemic_anxiety/results/lda_results/lda_feature.csv', encoding = "ISO-8859-1")
        lda = lda.drop(['index', 'Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords'], axis=1)
        lda.columns = [str(col) + '_lda' for col in lda.columns]
        lda = lda.rename(columns={"post_id_lda": "post_id"})


        # merge all features
        all_fea = liwc.merge(readability_df, on='post_id')
        all_fea = all_fea.merge(senti, on='post_id')
        all_fea = all_fea.merge(lda, on='post_id')

        # get labels
        labels = self.file[[self.labelcol, 'post_id']]
        all_data = all_fea.merge(labels, on='post_id')
        
        return all_data


    def pre_train(self)-> pd.DataFrame:
        """Merge data, get X, y and recode y."""

        all_data = self.merge_features()
        X = all_data.drop(columns=[self.labelcol])
        y = all_data[[self.labelcol]]
        return X, y

    def get_train_test_split(self) -> pd.DataFrame:
        '''split 10% holdout set, then split train test with the rest 90%, stratify splitting'''

        X, y = self.pre_train()
        # get 10% holdout set for testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 300, stratify = y)

        return X_train, X_test, y_train, y_test
        




class ColumnSelector:
    '''feature selector for pipline (pandas df format) '''
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)

        try:
            return X[self.columns]
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)

    def get_feature_names(self):
        return self.columns.tolist
            


class TrainingClassifiers: 
    
    def __init__(self, X_train, X_test, y_train, y_test, parameters, features_list, tfidf_words):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.parameters = parameters
        self.features_list = features_list
        self.tfidf_words = tfidf_words 
        self.path = '/disk/data/share/s1690903/pandemic_anxiety/'


    def get_other_feature_names(self):
        """Select all the merged features. """

        fea_list = []
        for fea in self.features_list: #select column names with keywords in dict
            f_list = [i for i in self.X_train.columns if fea in i]
            fea_list.append(f_list)
        #flatten a list
        flat = [x for sublist in fea_list for x in sublist]
        #convert to transformer object
        return flat


    def setup_pipeline(self, classifier):
        '''set up pipeline'''
        features_col = self.get_other_feature_names()


        pipeline = Pipeline([
        # ColumnSelector(columns = features_list),
            
            ('feats', FeatureUnion([
        # generate count vect features
                ('text', Pipeline([

                    ('selector', ColumnSelector(columns='text')),
                    # ('cv', CountVectorizer()),
                    ('tfidf', TfidfVectorizer(max_features=self.tfidf_words, ngram_range = (1,3), stop_words ='english', max_df = 0.50, min_df = 0.0025)),
                    # ('svd', TruncatedSVD(algorithm='randomized', n_components=300))
                     ])),
          # # select other features, feature sets are defines in the yaml file


                ('other_features', Pipeline([

                    ('selector', ColumnSelector(columns=features_col)),
                    ('impute', SimpleImputer(strategy='mean')),# impute nan with mean
                ])),

             ])),


               ('clf', Pipeline([
               # ('impute', SimpleImputer(strategy='mean')), #impute nan with mean
               ('scale', StandardScaler(with_mean=False)),  # scale features
                ('classifier', classifier),  # classifier
           
                 ])),
        ])
        return pipeline

    

    def training_models(self, pipeline):
        '''train models with grid search'''
        grid_search_item = GridSearchCV(pipeline, self.parameters, cv=5, scoring='accuracy')
        grid_search = grid_search_item.fit(self.X_train, self.y_train)
        
        return grid_search

    def test_model(self, classifier):
        '''test model and save data'''
        #start = time.time()
        #training model
        print('getting pipeline...')

        #the dictionary returns a list, here we extract the string from list use [0]
        pipeline = self.setup_pipeline(eval(classifier)())

        print('features', self.features_list)
        grid_search = self.training_models(pipeline)
        #make prediction
        print('prediction...')
      
        y_true, y_pred = self.y_test, grid_search.predict(self.X_test)
        report = classification_report(y_true, y_pred, digits=3)
        #store prediction result
        y_pred_series = pd.DataFrame(y_pred)
        result = pd.concat([y_true.reset_index(drop=True), y_pred_series], axis = 1)
        result.columns = ['y_true', 'y_pred']
       
        result.to_csv(self.path + 'results/best_result_{}.csv'.format(self.features_list) )
    
        return report, grid_search, pipeline
   

def loop_the_grid(label): #label column for prediction
    '''loop parameters in the environment file '''

    path = '/disk/data/share/s1690903/pandemic_anxiety/'
    experiment = load_experiment(path + 'evn/experiment.yaml')

    file_exists = os.path.isfile(path + 'results/classifier/test_result.csv')
    f = open(path + 'results/classifier/test_result.csv', 'a')
    writer_top = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    if not file_exists:
        writer_top.writerow(['best_scores'] + ['best_parameters'] + ['report'] + ['time'] + ['model'] +['feature_set'] +['tfidf_words'] + 'pre_label')
        f.close()
    
    read = Read_raw_data()
    all_files = read.read_all_files()
    # you can combine columns here
    #all_files = read.combine_columns('finance', 'financial_career', 'health_work')
    #all_files = read.combine_columns('health', 'health_infected', 'health_prev', 'death')

    # define how do you want to cimbine the predicted var
    #all_files = read.combine_columns('social', 'quarantine', 'socializing')


    for classifier in experiment['experiment']:

        # prepare environment
        prepare = PrepareData(all_files, label)
        prepare.save_combined_text()

        # split data
        X_train, X_test, y_train, y_test = prepare.get_train_test_split()
        X_train.to_csv(path + 'train_feature.csv')
        X_test.to_csv(path + 'test_feature.csv')
        X_train = X_train.drop(columns=['post_id'])
        X_test = X_test.drop(columns=['post_id'])
        for feature_set, features_list in experiment['features'].items():# loop feature sets
            for tfidf_words in experiment['tfidf_features']['max_fea']:# loop tfidf features
                

                f = open(path + 'results/classifier/test_result.csv', 'a')
                writer_top = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)

                parameters = experiment['experiment'][classifier]
                print('parameters are:', parameters)
                training = TrainingClassifiers(X_train=X_train, y_train=y_train, X_test=X_test, y_test =y_test, parameters =parameters, features_list =features_list, tfidf_words=tfidf_words)
                
                report, grid_search, pipeline = training.test_model(classifier)

                result_row = [[grid_search.best_score_, grid_search.best_params_, report, str(datetime.datetime.now()), classifier, features_list, tfidf_words, label]]

                writer_top.writerows(result_row)

                f.close()
                gc.collect()

# run lda                   
# read = Read_raw_data()
# all_files = read.read_all_files()
# p = PrepareData(all_files, 'anxiety')
# p.optimize_lda()
# all_data = p.merge_features() 


loop_the_grid('mental_health') # input which one you want to predict
#'focus', 'financial_career', 'information_gen',  'socializing' , 'infomation_sharing_private', 'health_infected', 'health_work', 'mental_health', 'rant', 'death', 'mourn_death', 'travelling',
# list_of_var = ['future']

# for col in list_of_var:
#     




# dummy classifier
def dummy_classifier(MoodslideWindow):
    '''loop parameters in the environment file '''

    path = '/disk/data/share/s1690903/predicting_depression_symptoms/data/'
    experiment = load_experiment(path + '../experiment/experiment.yaml')


    timewindow = 14
    step = 3
    # prepare environment 
    prepare = PrepareData(timewindow = timewindow, step = step)
                    
    #   # split data
    X_train, X_test, y_train, y_test = prepare.get_train_test_split()
    print(X_train.shape, X_test.shape)

    dummy_clf = DummyClassifier(strategy="stratified")
    dummy_clf.fit(X_test, y_test)
    y_pred = dummy_clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    print(report)
    return report 


