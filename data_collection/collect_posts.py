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


def load_experiment(path_to_experiment):
    """load experiment"""
    data = yaml.safe_load(open(path_to_experiment))
    return data

class CollectPost:
    """Collect posts via pushshift."""

    def __init__(self):
        '''define the main path'''
        self.datapath = '/disk/data/share/s1690903/pandemic_anxiety/data/'


    def __get_date(self, time):
        """Collect all the post from one user
        input1: list of authors
        output: output file.
        """
        t = datetime.datetime.fromtimestamp(time)
        return t.strftime('%m/%d/%Y/%H:%M:%S')

    def ___postids(self, filename):
        """Read post id files"""
        ids = pd.read_csv(self.datapath + filename)#'Anxiety_postid.csv'
        return ids

    def collect_posts(self, postidsFile):
        """Collect all the post from one user
        input1: list of authors
        output: output file
         """
        postids = self.___postids(postidsFile)
        f = open(self.datapath + 'posts/{}_posts.csv'.format(postidsFile.split('/')[1]), 'w')
        writer_top = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer_top.writerow(["post_id"] + ['author_flair'] + ["title"] + ["text"] + ["author"]  + ["score"] + ['upvote'] + ['downvote']+["num_comments"]+["url"] + ["timestamp"] +["subreddit_id"] + ["subreddit"] +['removal_reason'] + ['report_reasons']+ ['num_reports']+ ['num_crossposts']+ ['link_flair_text']+ ['mod_reports']+['num_reports'])
        
        for subreddit_id in postids['post_id']:
            submission = reddit.submission(id=subreddit_id)
            post_l = [[subreddit_id, submission.author_flair_text, submission.title, submission.selftext, submission.author, submission.score, submission.ups, submission.downs, submission.num_comments, submission.url, self.__get_date(submission.created), submission.subreddit_id, submission.subreddit, submission.removal_reason, submission.report_reasons, submission.num_reports,submission.num_crossposts, submission.link_flair_text, submission.mod_reports, submission.num_reports]]
            
            writer_top.writerows(post_l)


evn_path = '/disk/data/share/s1690903/pandemic_anxiety/evn/'
evn = load_experiment(evn_path + 'experiment.yaml')

reddit = praw.Reddit(client_id=evn['reddit_api_1']['client_id'],
                     client_secret=evn['reddit_api_1']['client_secret'],
                     user_agent=evn['reddit_api_1']['user_agent'],
                     username=evn['reddit_api_1']['username'],
                     password=evn['reddit_api_1']['password'])


#using pushshift to fetch IDS 
api = PushshiftAPI(reddit)
#extractPosts = pd.read_csv('data/CleanData/allSIP_RM_DEL.csv')
c = CollectPost()
c.collect_posts('postids/Anxiety_postids.csv')





