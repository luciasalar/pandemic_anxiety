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

from sklearn.dummy import DummyClassifier

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
        self.path = '/disk/data/share/s1690903/pandemic_anxiety/data/anno_test/'


    def read_all_files(self) -> pd.DataFrame:
        """ Read all the annotation files. Get data for liwc """

        all_anno = pd.read_csv(self.path + 'all_data_text.csv') # annotated data
        all_anno = all_anno[['post_id', 'anxiety', 'financial_career', 'quar_social', 'health_infected', 'break_guideline', 'health_work', 'mental_health', 'death', 'travelling', 'future']]

        all_covid = pd.read_csv(self.path + 'COVID19_support.csv') #covid19 support 

        # merge annotated data with all data
        all_files_pd = pd.merge(all_covid, all_anno, on='post_id', how='outer')

        # recode anxiety level
        all_files_pd['text'] = all_files_pd['text'].replace(np.nan, 0)
        all_files_pd['anxiety'] = all_files_pd['anxiety'].replace(0, 1)

        #drop duplication
        all_files_pd = all_files_pd.drop_duplicates(subset=['post_id'])

        # Replace Nan with 0.
        #liwc_file = all_files_pd[['title', 'text', 'post_id']]
       # liwc_file.to_csv('/disk/data/share/s1690903/pandemic_anxiety/data/annotations/test.csv', encoding='utf-8', index=False) # liwc file

        return all_files_pd

    #def increase_data(self):


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

    def __init__(self, data, labelcol):
        '''define the main path'''

        self.path = '/disk/data/share/s1690903/pandemic_anxiety/data/anno_test/'
        self.file = pd.read_csv(self.path + data)
        # join the title and text
        self.file['text'] = self.file['text'].str.cat(self.file['title'], sep=" ")
        self.labelcol = labelcol

    def save_combined_text(self):
        """combine title with text"""

        self.file.to_csv(self.path + 'combined_text_test.csv')


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
        
            #print(text)
            result = readability.getmeasures(str(text), lang='en')

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
        liwc = pd.read_csv(self.path + 'liwc_data_test.csv', encoding="ISO-8859-1")
        liwc.columns = [str(col) + '_liwc' for col in liwc.columns]
        liwc = liwc.rename(columns={"post_id_liwc": "post_id", "text_liwc": "text", "title_liwc": "title"})

        return liwc

    def convert_dict_df(self, dictionary, newcolname):
        """Covert dictionary to df."""

        new_df = pd.DataFrame.from_dict(dictionary, orient='index')
        new_df['post_id'] = new_df.index
        new_df.columns = [newcolname, 'post_id']

        return new_df

    def optimize_lda(self, topic_num):
        """Check the optimized lda scores """

        pt = ProcessText('/anno_test/combined_text_test.csv') # this is whole dataset
        cleaned_text = pt.simple_preprocess()
        entities = pt.extract_entities(cleaned_text)
        
        # get the optimized parameter 
        sent_topics_df = selected_best_LDA(pt.path_result, entities, topic_num, 'lda_test.csv')


    def merge_features(self)-> pd.DataFrame:
        """Here we generate all the features of the whole dataset, merge features with annotated labels """
        
        # get liwc
        liwc = self.get_liwc()
        
        # clean text data
        liwc = liwc.drop_duplicates(subset='post_id', keep='first', inplace=False)
        liwc = liwc[~liwc['text'].isin(['[removed]'])]
        liwc = liwc[~liwc['text'].isin(['[deleted]'])]
        liwc = liwc.dropna(subset=['text']) #5234 rows
        # combine text and title
        liwc['text'] = liwc['text'].str.cat(liwc['title'], sep=" ")

        # # get readability
        text_dict = self.convert_text_dict(liwc)
        readability_score = self.get_readability(text_dict)
        readability_df = self.convert_dict_df(readability_score, 'FleschReadingEase')    

        # # get sentiment
        neg = self.get_sentiment(text_dict, 'neg')
        neg_df = self.convert_dict_df(neg, 'neg')
        pos = self.get_sentiment(text_dict, 'pos')
        pos_df = self.convert_dict_df(pos, 'pos')
        neu = self.get_sentiment(text_dict, 'neu')
        #neu_df = self.convert_dict_df(neu, 'neu')
        senti = neg_df.merge(pos_df, on='post_id')
        #senti = senti.merge(neu_df, on='post_id')

        # # add feature tag
        senti.columns = [str(col) + '_senti' for col in senti.columns]
        senti = senti.rename(columns={"post_id_senti": "post_id"})

        # get topic modeling
        #self.optimize_lda()  # optimize model and generate lda_feature.csv
        lda = pd.read_csv('/disk/data/share/s1690903/pandemic_anxiety/results/lda_results/lda_feature.csv', encoding = "ISO-8859-1")
        lda = lda.drop(['index', 'Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords'], axis=1)
        lda.columns = [str(col) + '_lda' for col in lda.columns]
        lda = lda.rename(columns={"post_id_lda": "post_id"})


        # # merge all features
        all_fea = liwc.merge(readability_df, on='post_id')
        all_fea = all_fea.merge(senti, on='post_id')
        all_fea = all_fea.merge(lda, on='post_id')

        # merge with annotation labels
        all_anno = pd.read_csv(self.path + 'all_data_text.csv')# annotated data
        all_anno = all_anno[['post_id', 'anxiety', 'financial_career', 'quar_social', 'health_infected', 'break_guideline', 'health_work', 'mental_health', 'death', 'travelling', 'future', 'time']]
        
        ## merge annotated data with all data
        all_files_pd = pd.merge(all_fea, all_anno, on='post_id', how='outer')

        # recode anxiety level
        all_files_pd['anxiety'] = all_files_pd['anxiety'].replace(0, 1)


        # # get labels
        # labels = self.file[[self.labelcol, 'post_id']]
        # all_data = all_fea.merge(labels, on='post_id')
        
        return all_files_pd

    def select_time(self, data, start_date, end_date):
        """select the data according to time"""

        data['time_delta'] = pd.to_datetime(data['time'], format='%m/%d/%Y/%H:%M:%S').dt.date
        startdate = pd.to_datetime(start_date).date()
        enddate = pd.to_datetime(end_date).date()
        data2 = data.loc[data['time_delta'].between(startdate, enddate, inclusive=False)]
        # remove the time delta
        data2 = data2.drop(columns=['time_delta'])

        return data2


    def pre_train(self)-> pd.DataFrame:
        """Merge data, get X, y and recode y."""

        all_data = self.merge_features()
        # here we drop data without annotation to create train test set
        all_data2 = all_data.dropna(subset=['anxiety'])

        # here we create the prediction set (without annotation)
        prediction_sample = all_data.loc[all_data['anxiety'].isna()]
        prediction_sample = prediction_sample.drop(columns=[self.labelcol])

        # for the training set, we drop the label column, y label assign time
        X = all_data2.drop(columns=[self.labelcol])
        y = all_data2[[self.labelcol, 'time']]
        return X, y, prediction_sample

    def get_train_test_split(self) -> pd.DataFrame:
        '''use last month as test set'''

        X, y, prediction_sample = self.pre_train()
  
        X_train = self.select_time(X, '2/1/2020', '9/30/2020')
        X_test = self.select_time(X, '10/1/2020', '10/30/2020')

        y_train = self.select_time(y, '2/1/2020', '9/30/2020')
        y_test = self.select_time(y, '10/1/2020', '10/30/2020')
        
        # # remove time from train, test
        y_train = y_train.drop(columns=['time'])
        y_test = y_test.drop(columns=['time'])
        
        # convert y to categories
        y_train = y_train[self.labelcol]
        y_test = y_test[self.labelcol]

        # get 10% holdout set for testing
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 300, stratify = y)

        return X_train, X_test, y_train, y_test, prediction_sample

    def get_test_data(self):
        """Merge data, get X, y and recode y."""
        



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
    
    def __init__(self, X_train, X_test, y_train, y_test, parameters, features_list, tfidf_words, label, prediction_sample = None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.parameters = parameters
        self.features_list = features_list
        self.tfidf_words = tfidf_words 
        self.path = '/disk/data/share/s1690903/pandemic_anxiety/'
        self.label = label
        if prediction_sample is not None:
           self.prediction_sample = prediction_sample



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
        '''train model, use model to predict on new data and save results'''
       
        #training model
        print('getting pipeline...')

        #the dictionary returns a list, here we extract the string from list use [0]
        pipeline = self.setup_pipeline(eval(classifier)())

        # train model
        print('features', self.features_list)
        grid_search = self.training_models(pipeline)
        # make prediction on test set
        print('prediction...')
      
        y_true, y_pred = self.y_test, grid_search.predict(self.X_test)

        # store classification report
        report = classification_report(y_true, y_pred, digits=2)

        # store prediction result
        y_pred_series = pd.DataFrame(y_pred)
        result = pd.concat([y_true.reset_index(drop=True), y_pred_series, self.X_test['post_id'].reset_index(drop=True)], axis = 1)
        result.columns = ['y_true', 'y_pred', 'post_id']

        # make prediction on new sample
        prediction_sample = self.prediction_sample
        print('predict sample')
        prediction_sample['prediction_{}'.format(self.label)] = grid_search.predict(self.prediction_sample)

        #store the best prediction result
        result.to_csv(self.path + 'results/test_result_{}.csv'.format(self.label))

        #store the prediction for new sample
        prediction_sample.to_csv(self.path + 'results/prediction_sample_{}.csv'.format(self.label))
    
        return report, grid_search, pipeline
   

def loop_the_grid(label): #label column for prediction
    '''loop parameters in the environment file '''

    path = '/disk/data/share/s1690903/pandemic_anxiety/'
    experiment = load_experiment(path + 'evn/experiment.yaml')

    file_exists = os.path.isfile(path + 'results/classifier/test_result.csv')
    f = open(path + 'results/classifier/test_result.csv', 'a')
    writer_top = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    if not file_exists:
        writer_top.writerow(['best_scores'] + ['best_parameters'] + ['report'] + ['time'] + ['model'] +['feature_set'] +['tfidf_words'] + ['pre_label'])
        f.close()
    
    read = Read_raw_data()
    #all_files = read.read_all_files()

    for classifier in experiment['experiment']:

        # prepare environment
        prepare = PrepareData('COVID19_support.csv', label)
        print('preparing data...')
        prepare.save_combined_text()

        # split data
        X_train, X_test, y_train, y_test, prediction_sample = prepare.get_train_test_split()
        #X_train.to_csv(path + 'train_feature.csv')
        print('save features')
        X_test.to_csv(path + 'test_feature.csv')
        #X_train = X_train.drop(columns=['post_id'])
        #X_test = X_test.drop(columns=['post_id'])

        for feature_set, features_list in experiment['features'].items():# loop feature sets
            for tfidf_words in experiment['tfidf_features']['max_fea']:# loop tfidf features

                # store test result
                f = open(path + 'results/classifier/test_result2.csv', 'a')
                writer_top = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                
                # loop parameters
                parameters = experiment['experiment'][classifier]
                print('parameters are:', parameters)
                # train classifier
                training = TrainingClassifiers(X_train=X_train, y_train=y_train, X_test=X_test, y_test =y_test, parameters =parameters, features_list =features_list, tfidf_words=tfidf_words, prediction_sample=prediction_sample, label=label)
                
                # store results
                report, grid_search, pipeline = training.test_model(classifier)

                result_row = [[grid_search.best_score_, grid_search.best_params_, report, str(datetime.datetime.now()), classifier, features_list, tfidf_words, label, 'last_month']]

                writer_top.writerows(result_row)

                f.close()
                gc.collect()


def baseline(label): #label column for prediction
    '''loop parameters in the environment file '''

    path = '/disk/data/share/s1690903/pandemic_anxiety/'
    experiment = load_experiment(path + 'evn/experiment.yaml')

    file_exists = os.path.isfile(path + 'results/classifier/test_result.csv')
    f = open(path + 'results/classifier/test_result.csv', 'a')
    writer_top = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    if not file_exists:
        writer_top.writerow(['best_scores'] + ['best_parameters'] + ['report'] + ['time'] + ['model'] +['feature_set'] +['tfidf_words'] + ['pre_label'])
        f.close()
    
    read = Read_raw_data()
    #all_files = read.read_all_files()

    for classifier in experiment['experiment']:

        # prepare environment
        prepare = PrepareData('COVID19_support.csv', label)
        print('preparing data...')
        prepare.save_combined_text()

        # split data
        X_train, X_test, y_train, y_test, prediction_sample = prepare.get_train_test_split()
        #X_train.to_csv(path + 'train_feature.csv')
        print('save features')
        X_test.to_csv(path + 'test_feature.csv')

        dummy_clf = DummyClassifier(strategy="stratified")
        # store test result

        dummy_clf = DummyClassifier(strategy="stratified")
        dummy_clf.fit(X_test, y_test)
        y_pred = dummy_clf.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)

        f = open(path + 'results/classifier/test_result2.csv', 'a')
        writer_top = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                
             
        result_row = [[None, None, report, str(datetime.datetime.now()), None, None, None, label, 'baseline']]

        writer_top.writerows(result_row)

        f.close()
        gc.collect()

# run lda                   
#read = Read_raw_data()
#all_file_df = read.read_all_files()
p = PrepareData('COVID19_support.csv', 'anxiety')
# p.save_combined_text()

# for i in [15, 20, 25, 30]:
#     p.optimize_lda(i)

#all_data = p.merge_features() 
#X_train, X_test, y_train, y_test, prediction_sample = p.get_train_test_split()


var_list = ['anxiety', 'financial_career', 'quar_social', 'health_infected', 'break_guideline', 'health_work', 'mental_health', 'death', 'travelling', 'future']


# for var in var_list:
#     loop_the_grid(var)# input which one you want to predict

for var in var_list:
    baseline(var)#

# dummy classifier
# def dummy_classifier(MoodslideWindow):
#     '''loop parameters in the environment file '''

#     path = '/disk/data/share/s1690903/predicting_depression_symptoms/data/'
#     experiment = load_experiment(path + '../experiment/experiment.yaml')

#     # prepare environment 
#     prepare = PrepareData(timewindow=timewindow, step = step)
                    
#     #   # split data
#     X_train, X_test, y_train, y_test = prepare.get_train_test_split()
#     print(X_train.shape, X_test.shape)

#     dummy_clf = DummyClassifier(strategy="stratified")
#     dummy_clf.fit(X_test, y_test)
#     y_pred = dummy_clf.predict(X_test)
#     report = classification_report(y_test, y_pred, output_dict=True)
#     print(report)
#     return report 


