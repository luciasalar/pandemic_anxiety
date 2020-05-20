import praw
import csv
import pprint
import os
import datetime
from praw.models import MoreComments
import pandas as pd
import datetime as dt
from psaw import PushshiftAPI
from ruamel import yaml


#Here we set to collect posts before set Date, you can set to collect posts after set Date
#by changing after to before in __get_post_ids

def load_experiment(path_to_experiment):
    """load experiment"""
    data = yaml.safe_load(open(path_to_experiment))
    return data

class CollectPostids:
    """Collect posts via pushshift."""
    def __init__(self, start_year, start_year_month, start_day, subreddit_name):
        '''define the main path'''
        self.datapath = '/disk/data/share/s1690903/pandemic_anxiety/data/postids/'
        self.start_year = start_year
        self.start_year_month = start_year_month
        self.start_day = start_day
        self.subreddit_name = subreddit_name

    def __get_post_ids(self):
        """Get list of post id."""
        start_epoch = int(dt.datetime(self.start_year, self.start_year_month, self.start_day).timestamp())
        result = list(api.search_submissions(before=start_epoch, subreddit=self.subreddit_name,filter=['subreddit']))
        print(result)
        return result

    def save_postids(self):
        """Save lists of postid to csv."""
        result_l = self.__get_post_ids()
        f = open(self.datapath + '{}_postids.csv'.format(self.subreddit_name), 'w')
        writer_top = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer_top.writerow(["post_id"])
        for i in range(0, len(result_l)):
            print(result_l[i].id)
            writer_top.writerows([[result_l[i].id]])
        f.close()


evn_path = '/disk/data/share/s1690903/pandemic_anxiety/evn/'
evn = load_experiment(evn_path + 'experiment.yaml')

reddit = praw.Reddit(client_id=evn['reddit_api_1']['client_id'],
                     client_secret=evn['reddit_api_1']['client_secret'],
                     user_agent=evn['reddit_api_1']['user_agent'],
                     username=evn['reddit_api_1']['username'],
                     password=evn['reddit_api_1']['password'])


#using pushshift to fetch IDS
api = PushshiftAPI(reddit)

# get lists of subreddits
subreddits = evn['subreddits']['subs']


for sub in subreddits:
    c = CollectPostids(2018, 1, 1, sub)
    c.save_postids()



