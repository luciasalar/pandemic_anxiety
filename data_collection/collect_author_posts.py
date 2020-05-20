import praw
import csv
import pprint
import os
import datetime
from praw.models import MoreComments
import pandas as pd
from ruamel import yaml

#this script collect all the posts from author

#for submission in reddit.subreddit(keyword).hot(limit=10):
#    print(submission.title)
def load_experiment(path_to_experiment):
    """load experiment"""
    data = yaml.safe_load(open(path_to_experiment))
    return data


class CollectPost:
    def __init__(self):
        '''define the main path'''
        self.datapath = '/disk/data/share/s1690903/pandemic_anxiety/data/'

    def __get_date(self, time):
        """Convert timestamp to string """
        t = datetime.datetime.fromtimestamp(time)
        return t.strftime('%m/%d/%Y/%H:%M:%S')


    def collect_author_Posts(self, input1, output):
        """Collect all the post from one user
        input1: list of authors
        output: output file
         """
        f = open(self.datapath + output, 'w')
        writer_top = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer_top.writerow(["author"] + ["text"] + ["subreddit"] + ["title"] + ["t_timestamp"] + ["post_id"] + ["num_comments"] + ["score"])

        subreddits = pd.read_csv(self.datapath + input1)
        for author in subreddits.author:
            if author != 'nan' or 'NaN':
                user = reddit.redditor(str(author))
                try:
                # New is just the listing type. Newest first, then descending in date order
                    comments_l = [[authorPosts.author, authorPosts.selftext, authorPosts.subreddit, authorPosts.title, self.__get_date(authorPosts.created), authorPosts.id, authorPosts.num_comments, authorPosts.score] for authorPosts in user.submissions.new(limit=None)]
                except:
                    continue
                writer_top.writerows(comments_l)


evn_path = '/disk/data/share/s1690903/pandemic_anxiety/evn/'
evn = load_experiment(evn_path + 'experiment.yaml')

# path = '/Users/lucia/phd_work/redditProject/'

reddit = praw.Reddit(client_id= evn['reddit_api_1']['client_id'],
                     client_secret=evn['reddit_api_1']['client_secret'],
                     user_agent=evn['reddit_api_1']['user_agent'],
                     username =evn['reddit_api_1']['username'],
                     password =evn['reddit_api_1']['password'])

collect_author_Posts(path + 'allP.csv', path + 'AuthorAllP.csv')





