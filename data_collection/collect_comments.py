import praw
import csv
import pprint
import os
import datetime
from praw.models import MoreComments
import pandas as pd


#send message to user
reddit = praw.Reddit(client_id='wyESgnQhXG9mNQ',
                     client_secret='079juG33U8gsRx9HUujEQngvHic',
                     user_agent='collect reddits',
                     username ='luciasalar',
                     password ='myjsybjwqt888')

# reddit = praw.Reddit(client_id='wyESgnQhXG9mNQ',
#                      client_secret='079juG33U8gsRx9HUujEQngvHic',
#                      user_agent='collect reddits',
#                      username ='luciasalar',
#                      password ='myjsybjwqt888')

def __get_date(time):
	
	t = datetime.datetime.fromtimestamp(time)
	return t.strftime('%m/%d/%Y/%H:%M:%S')

def collect_all_comments(input1,output):
	f = open(output , 'w')
	writer_top = csv.writer(f, delimiter = ',',quoting=csv.QUOTE_MINIMAL)
	writer_top.writerow(["author"] +["subreddit_id"] + ["t_level_comment"] + ["t_comment"] +["t_timestamp"] +["t_parent_id"] +["t_comment_score"] +["t_subreddit_id2"] +['author_flair_richtext'] + ['depth'] + ['distinguished'] + ['is_submitter'] + ['author_fullname'])

	subreddits = pd.read_csv(input1)
	for subreddit_id in subreddits.post_id:
		submission = reddit.submission(id = subreddit_id)
		try:
			comments_l = [[top_level_comment.author, subreddit_id, top_level_comment.id, top_level_comment.body, __get_date(top_level_comment.created), top_level_comment.parent_id, top_level_comment.score, top_level_comment.subreddit_id, top_level_comment.author_flair_richtext, top_level_comment.depth, top_level_comment.distinguished, top_level_comment.is_submitter, top_level_comment.author_fullname] for top_level_comment in submission.comments.list()]
		except AttributeError:
			continue
		writer_top.writerows(comments_l)


path = '/Users/lucia/phd_work/redditProject/'
collect_all_comments(path + 'data/CleanData/allPNew.csv', path + 'data/recollectNew/New_comments.csv' )



