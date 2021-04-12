import pandas as pd 
import numpy as np
import datetime 
from datetime import datetime 
from typing import Dict, Tuple, Sequence
import typing

# select posts for annotation
class DataSelection:

    def __init__(self):
        """Define varibles."""
        self.path_data = '/disk/data/share/s1690903/pandemic_anxiety/data/posts/COVID19_support_postids_posts.csv'
        self.data = pd.read_csv(self.path_data)
        self.save_path_data = '/disk/data/share/s1690903/pandemic_anxiety/data/annotations/others/'

    def clean_data(self) -> pd.DataFrame:
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

    def filter_data(self) -> pd.DataFrame:
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
        july = filtered[(filtered['time'] > '2020-07-1') & (filtered['time'] < '2020-07-31')]
        aug = filtered[(filtered['time'] > '2020-08-1') & (filtered['time'] < '2020-08-31')]
        sep = filtered[(filtered['time'] > '2020-09-1') & (filtered['time'] < '2020-09-30')]
        octo = filtered[(filtered['time'] > '2020-10-1') & (filtered['time'] < '2020-10-30')]

        return feb, apr, may, jun, july, aug, sep, octo

    def random_sample(self, num) -> pd.DataFrame:
        """Select random sample """

        feb, apr, may, jun, july, aug, sep, octo = self.get_monthly_data()
        feb = feb.sample(num, random_state=123)
        apr = apr.sample(num, random_state=123)
        may = may.sample(num, random_state=123)
        jun = jun.sample(num, random_state=123)
        july = july.sample(num, random_state=123)
        Aug = aug.sample(num, random_state=123)
        Sep = sep.sample(num, random_state=123)
        Oct = octo.sample(num, random_state=123)


        feb.to_csv(self.save_path_data + 'feb1.csv')
        apr.to_csv(self.save_path_data + 'apr1.csv')
        may.to_csv(self.save_path_data + 'may1.csv')
        jun.to_csv(self.save_path_data + 'jun1.csv')
        july.to_csv(self.save_path_data + 'july1.csv')
        Aug.to_csv(self.save_path_data + 'Aug.csv')
        Sep.to_csv(self.save_path_data + 'Sep.csv')
        Oct.to_csv(self.save_path_data + 'Oct.csv')

        return feb, apr, may, jun, july, Aug, Sep, Oct

    def resample(self, feb_sel, apr_sel, may_sel, jun_sel, july_sel, num) -> pd.DataFrame:
        """Resample """

        feb, apr, may, jun, july = self.get_monthly_data()
        common_feb = feb.merge(feb_sel, on=['post_id'])
        common_apr = apr.merge(apr_sel, on=['post_id'])
        common_may = may.merge(may_sel, on=['post_id'])
        common_jun = jun.merge(jun_sel, on=['post_id'])
        common_july = jun.merge(jun_sel, on=['post_id'])

        new_feb = feb[(~feb.post_id.isin(common_feb.post_id))].sample(num, random_state=123)
        new_apr = apr[(~apr.post_id.isin(common_apr.post_id))].sample(num, random_state=123)
        new_may = may[(~may.post_id.isin(common_may.post_id))].sample(num, random_state=123)
        new_jun = jun[(~jun.post_id.isin(common_jun.post_id))].sample(num, random_state=123)
        new_july = july[(~july.post_id.isin(common_july.post_id))].sample(num, random_state=123)

        return new_feb, new_apr, new_may, new_jun, new_july


ds = DataSelection()
# filtered = ds.filter_data()
# filtered.to_csv('/disk/data/share/s1690903/pandemic_anxiety/data/annotations/covid19_support.csv')
feb_sel, apr_sel, may_sel, jun_sel, july_sel, Aug_sel, Sep_sel, Oct_sel = ds.random_sample(100)
# get another 100 posts
#new_feb, new_apr, new_may, new_jun, new_july = ds.resample(feb_sel, apr_sel, may_sel, jun_sel, july_sel 100)



#divide_data

















