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
import gc
import time


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
        file_exists = os.path.isfile(self.datapath + 'posts/{}_posts.csv'.format(postidsFile.split('/')[1].split('.')[0]))


        if not file_exists:
            f = open(self.datapath + 'posts/{}_posts.csv'.format(postidsFile.split('/')[1].split('.')[0], str(datetime.datetime.now())), 'a', encoding='utf-8-sig')
            writer_top = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)

            writer_top.writerow(["post_id"] + ["author_flair_text"] + ["title"] + ["text"] + ["author"]+["score"] + ["ups"] + ["downs"] +["num_comments"] +["url"] + ["time"] +['subreddit_id'] + ["subreddit"] + ["removal_reason"] + ["report_reasons"] + ["num_reports"] + ["num_crossposts"] + ["link_flair_text"] + ["mod_reports"] + ["num_reports"])
            f.close()
              
        count = 0
        last_date = None
        for subreddit_id in postids['post_id']:
            submission = reddit.submission(id=subreddit_id)
            post_l = [[subreddit_id, submission.author_flair_text, submission.title, submission.selftext, submission.author, submission.score, submission.ups, submission.downs, submission.num_comments, submission.url, self.__get_date(submission.created), submission.subreddit_id, submission.subreddit, submission.removal_reason, submission.report_reasons, submission.num_reports,submission.num_crossposts, submission.link_flair_text, submission.mod_reports, submission.num_reports]]

            f = open(self.datapath + 'posts/{}_posts.csv'.format(postidsFile.split('/')[1].split('.')[0], str(datetime.datetime.now())), 'a', encoding='utf-8-sig')
            writer_top = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
      
            writer_top.writerows(post_l)

            #sleep 10 sec every 10 query
            count = count + 1
            if isinstance(count / 10, int):
                time.sleep(5)

            last_date = self.__get_date(submission.created)
        
        #print(last_date)
            f.close()
        gc.collect()
        return last_date

evn_path = '/disk/data/share/s1690903/pandemic_anxiety/evn/'
evn = load_experiment(evn_path + 'experiment.yaml')

reddit = praw.Reddit(client_id=evn['reddit_api_3']['client_id'],
                         client_secret=evn['reddit_api_3']['client_secret'],
                         user_agent=evn['reddit_api_3']['user_agent'],
                         username=evn['reddit_api_3']['username'],
                         password=evn['reddit_api_3']['password'])


#using pushshift to fetch IDS 
api = PushshiftAPI(reddit)

if __name__ == "__main__":
    c = CollectPost()

    subreddits = evn['subreddits']['subs']

    for sub in subreddits:
        c.collect_posts('postids/{}_postids.csv'.format(sub))
        time.sleep(30)




































