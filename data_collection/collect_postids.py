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
import time
from collect_posts import *


#Here we set to collect posts before set Date, you can set to collect posts after set Date
#by changing after to before in __get_post_ids

def load_experiment(path_to_experiment):
    """load experiment"""
    data = yaml.safe_load(open(path_to_experiment))
    return data

class CollectPostids:
    """Collect posts via pushshift."""
    def __init__(self, start_year, start_month, start_day, subreddit_name):
        '''define the main path'''
        self.datapath = '/disk/data/share/s1690903/pandemic_anxiety/data/postids/'
        self.start_year = start_year
        self.start_month = start_month
        self.start_day = start_day
        self.subreddit_name = subreddit_name

    def __get_post_ids(self):
        """Get list of post id."""
        start_epoch = int(dt.datetime(self.start_year, self.start_month, self.start_day).timestamp())
        result = list(api.search_submissions(before=start_epoch, subreddit=self.subreddit_name,filter=['subreddit'], limit=1000))
        #print(result)
        return result

    def __get_post_ids_ts(self, start_epoch):
        """Get list of post id."""
        result = list(api.search_submissions(after=start_epoch, subreddit=self.subreddit_name,filter=['subreddit'], limit=10))
        #print(result)
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

def get_posts(subreddit, year, day, month, iteration, end_year, end_day, end_month):
    evn_path = '/disk/data/share/s1690903/pandemic_anxiety/evn/'
    evn = load_experiment(evn_path + 'experiment.yaml')

    reddit = praw.Reddit(client_id=evn['reddit_api_3']['client_id'],
                         client_secret=evn['reddit_api_3']['client_secret'],
                         user_agent=evn['reddit_api_3']['user_agent'],
                         username=evn['reddit_api_3']['username'],
                         password=evn['reddit_api_3']['password'])

    #using pushshift to fetch IDS
    api = PushshiftAPI(reddit)

    count = 0
    while count < iteration:
        # get post ids
        c = CollectPostids(year, month, day, subreddit)
        c.save_postids()
        
        # get posts and continue the next iteration 
        cp = CollectPost()
        last_day = cp.collect_posts('postids/{}_postids.csv'.format(subreddit))
        print(last_day)
        
        year = int(last_day.split('/')[2])
        month = int(last_day.split('/')[0])
        day = int(last_day.split('/')[1])
        print(month, day)

        if (year == end_year and month == end_month and day == end_day):
            break

        count = count + 1
        time.sleep(30)


# now we loop through all the subs
# subreddits = evn['subreddits']['subs_all']
# for sub in subreddits:
#     get_posts(sub, 2020, 21, 5, 20, 2020, 6, 1)


# Axiety reach 2020, 2, 28
get_posts('COVID19_support', 2020, 10, 8, 100, 2020, 1, 7)


#Social skill starts from Dec 2018
# OCD 13/12/2017

#health anxiety 19/4/2018







