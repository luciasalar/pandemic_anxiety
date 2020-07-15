import pandas as pd 
import numpy as np
import datetime 
from datetime import datetime 
from typing import Dict, Tuple, Sequence
import typing


class DataSelection:

    def __init__(self):
        """Define varibles."""
        self.path_data = '/disk/data/share/s1690903/pandemic_anxiety/data/posts/COVID19_support_postids_posts.csv'
        self.data = pd.read_csv(self.path_data)

    def clean_data(self):
        """Remove duplicates and deleted posts and nan"""

        data = self.data.drop_duplicates(subset='post_id', keep='first', inplace=False)
        data = data[~data['text'].isin(['[removed]'])]
        data = data[~data['text'].isin(['[deleted]'])]
        data = data[~data['text'].str.lower().isin(['deleted'])]
        data = data[~data['text'].str.lower().isin(['removed'])]

        #remove empty string
        data['text'].replace('', np.nan, inplace=True)
        data = data.dropna(subset=['text'])
        return data

    def filter_data(self):
        """filter data with certain tags"""
        clean_data = self.clean_data()
        filter_data = clean_data[~(clean_data['link_flair_text'] == 'Resources') | (clean_data['link_flair_text'] == 'Discussion')]

        return filter_data


    def get_monthly_data(self) -> pd.DataFrame:
        """get topics according to season timeline 
        result saved in results/lda_results/
        """

        #cleaned_text = self.simple_preprocess()
        filtered = self.filter_data()
        filtered['time'] = pd.to_datetime(filtered['time'])
        feb = filtered[(filtered['time'] > '2020-02-15') & (filtered['time'] < '2020-03-31')]
        apr = filtered[(filtered['time'] > '2020-04-1') & (filtered['time'] < '2020-04-30')]
        may = filtered[(filtered['time'] > '2020-05-1') & (filtered['time'] < '2020-05-31')]
        jun = filtered[(filtered['time'] > '2020-06-1') & (filtered['time'] < '2020-06-30')]

        return feb, apr, may, jun

    def random_sample(self, num):
        feb, apr, may, jun = self.get_monthly_data()
        feb = feb.sample(num)
        apr = apr.sample(num)
        may = may.sample(num)
        jun = jun.sample(num)

        return feb, apr, may, jun





ds = DataSelection()
# filtered = ds.filter_data()
# filtered.to_csv('/disk/data/share/s1690903/pandemic_anxiety/data/annotations/covid19_support.csv') 
feb, apr, may, jun = ds.get_monthly_data(100)
#divide_data

















