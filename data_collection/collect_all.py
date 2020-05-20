import praw
import csv
import pprint
import os
import datetime
from praw.models import MoreComments
import pandas as pd




reddit = praw.Reddit(client_id='KvGkRjHQnvIbgA',
                     client_secret='YUROHLAqOV4l2ICEtnY0y9D7y1Q',
                     user_agent='collect reddits')

print(reddit.read_only)

# continued from code above

#for submission in reddit.subreddit(keyword).hot(limit=10):
#    print(submission.title)
def __get_date(time):
	
	t = datetime.datetime.fromtimestamp(time)
	return t.strftime('%m/%d/%Y/%H:%M:%S')

#search the new posts in a subreddit, return 1000 posts
def collect_subreddit_new(keyword, fname):

	subreddit_l = [[i.title, i.selftext, i.author, i.score, i.num_comments, i.id, i.url, __get_date(i.created), i.distinguished, i.stickied] for i in reddit.subreddit(keyword).hot(limit=1000)]

	with open(fname , 'w') as f:
	    writer = csv.writer(f, delimiter = ',',quoting=csv.QUOTE_MINIMAL)
	    writer.writerow(["title"] + ["text"] + ["author"] + ["score"] + ["num_comments"]+["subreddit_id"] + ["url"] + ["timestamp"] +["distinguished"] +["stickied"])
	    writer.writerows(subreddit_l)

	subreddits = pd.read_csv(fname)
	return subreddits

subreddit = collect_subreddit_new('relationship_advice', 'relationship_advice_hot.csv')

#subreddit = collect_subreddit_new('survivinginfidelity')

#search reddit by keywords, each query return 100 posts  
def collect_search(keyword, fname):

	subreddit_l = [[i.title, i.selftext, i.author, i.score, i.num_comments, i.id, i.url, __get_date(i.created), i.distinguished, i.stickied] for i in reddit.subreddit('all').search(keyword)]

	with open(fname , 'w') as f:
	    writer = csv.writer(f, delimiter = ',',quoting=csv.QUOTE_MINIMAL)
	    writer.writerow(["title"] + ["text"] + ["author"] + ["score"] + ["num_comments"]+["subreddit_id"] + ["url"] + ["timestamp"] +["distinguished"] +["stickied"])
	    writer.writerows(subreddit_l)

	subreddits = pd.read_csv(fname)
	return subreddits

subreddit = collect_search('gf+cheated', 'gf_cheated2.csv')
collect_comments(subreddit['subreddit_id'])

#kewords:  r/Infidelity. r/survivinginfidelity
####

def collect_comments(reddit_id_list):


	f1 = open('reddit_comments_top.csv' , 'w')
	f2 = open('reddit_comments_l2.csv' , 'w')
	f3 = open('reddit_comments_l3.csv' , 'w')
	f4 = open('reddit_comments_l4.csv' , 'w')
	f5 = open('reddit_comments_l5.csv' , 'w')
	f6 = open('reddit_comments_l6.csv' , 'w')

	writer_top = csv.writer(f1, delimiter = ',',quoting=csv.QUOTE_MINIMAL)
	writer_top.writerow(["author"] +["subreddit_id"] + ["t_level_comment"] + ["t_comment"] +["t_timestamp"] +["t_parent_id"] +["t_comment_score"] +["t_subreddit_id2"])

	writer_2 = csv.writer(f2, delimiter = ',',quoting=csv.QUOTE_MINIMAL)
	writer_2.writerow(["author"] +["subreddit_id_top"] + ["second_level_comment"] + ["comment_l2"] +["timestamp_l2"] +["parent_id_l2"] +["comment_score_l2"] +["subreddit_id2_l2"])

	writer_3 = csv.writer(f3, delimiter = ',',quoting=csv.QUOTE_MINIMAL)
	writer_3.writerow(["author"] +["subreddit_id_top"] + ["third_level_comment"] + ["comment_l3"] +["timestamp_l3"] +["parent_id_l3"] +["comment_score_l3"] +["subreddit_id2_l3"])

	writer_4 = csv.writer(f4, delimiter = ',',quoting=csv.QUOTE_MINIMAL)
	writer_4.writerow(["author"] +["subreddit_id_top"] + ["fourth_level_comment"] + ["comment_l4"] +["timestamp_l4"] +["parent_id_l4"] +["comment_score_l4"] +["subreddit_id2_l4"])

	writer_5 = csv.writer(f5, delimiter = ',',quoting=csv.QUOTE_MINIMAL)
	writer_5.writerow(["author"] +["subreddit_id_top"] + ["fifth_level_comment"] + ["comment_l5"] +["timestamp_l5"] +["parent_id_l5"] +["comment_score_l5"] +["subreddit_id2_l5"])

	writer_6 = csv.writer(f6, delimiter = ',',quoting=csv.QUOTE_MINIMAL)
	writer_6.writerow(["author"] +["subreddit_id_top"] + ["sixth_level_comment"] + ["comment_l6"] +["timestamp_l6"] +["parent_id_l6"] +["comment_score_l6"] +["subreddit_id2_l6"])


#top level comments
	for subreddit_id in reddit_id_list:
		submission = reddit.submission(id = subreddit_id)
		comments_l = None
		try:
			comments_l = [[top_level_comment.author, subreddit_id, top_level_comment.id, top_level_comment.body, __get_date(top_level_comment.created), top_level_comment.parent_id, top_level_comment.score, top_level_comment.subreddit_id] for top_level_comment in submission.comments.list()]
		except AttributeError:
			continue
		writer_top.writerows(comments_l)
		

		#second level comments
		for top_level_comment in submission.comments:
			try:
				comments_l2 = [[second_level_comment.author, subreddit_id, second_level_comment.id, second_level_comment.body, __get_date(second_level_comment.created), second_level_comment.parent_id, second_level_comment.score, second_level_comment.subreddit_id] for second_level_comment in top_level_comment.replies]							
			except AttributeError:
				continue
			writer_2.writerows(comments_l2)

			#third level comments	
			for second_level_comment in top_level_comment.replies:
				try:
					comments_l3 = [[third_level_comment.author, subreddit_id, third_level_comment.id, third_level_comment.body, __get_date(third_level_comment.created), third_level_comment.parent_id, third_level_comment.score, third_level_comment.subreddit_id] for third_level_comment in second_level_comment.replies]				
				except AttributeError:
					continue
				writer_3.writerows(comments_l3)

             #fourth level comments
				for third_level_comment in second_level_comment.replies:
					try:
						comments_l4 = [[fourth_level_comment.author, subreddit_id, fourth_level_comment.id, fourth_level_comment.body, __get_date(fourth_level_comment.created), fourth_level_comment.parent_id, fourth_level_comment.score, fourth_level_comment.subreddit_id] for fourth_level_comment in third_level_comment.replies]			
					except AttributeError:
						continue
					writer_4.writerows(comments_l4)
	
					#fifth level comments
					for fourth_level_comment in third_level_comment.replies:
						try:
							comments_l5 = [[fifth_level_comment.author, subreddit_id, fifth_level_comment.id, fifth_level_comment.body, __get_date(fifth_level_comment.created), fifth_level_comment.parent_id, fifth_level_comment.score, fifth_level_comment.subreddit_id] for fifth_level_comment in fourth_level_comment.replies]			
						except AttributeError:
							continue
						writer_5.writerows(comments_l5)

						for fifth_level_comment in fourth_level_comment.replies:
							try:
								comments_l6 = [[sixth_level_comment.author, subreddit_id, sixth_level_comment.id, fifth_level_comment.body, __get_date(sixth_level_comment.created), sixth_level_comment.parent_id, sixth_level_comment.score, sixth_level_comment.subreddit_id] for sixth_level_comment in fifth_level_comment.replies]				
							except AttributeError:
								continue
							writer_6.writerows(comments_l6)

	#comments = pd.read_csv('reddit_comments_l4.csv')

	f1.close()
	f2.close()
	f3.close()
	f4.close()
	f5.close()
	f6.close()

	print('comments collected!') 


subreddit = collect_subreddit_new('all-husband+cheated+on+me', 'husband_cheated.csv')
subreddit = collect_search('title:boyfriend+cheated+on+me', 'bf_cheated.csv')

#collect_comments(subreddit['subreddit_id'])


#keywords: infidelity, survivinginfidelity, relationship, relationship_asvice, wife+cheated+on+me,husband+cheated+on+me, girlfriend/boyfriend/ex/partner cheated on me


###collect most of the comments of a post ### use this one

def collect_all_comments(input1,output):
	f = open(output , 'w')
	writer_top = csv.writer(f, delimiter = ',',quoting=csv.QUOTE_MINIMAL)
	writer_top.writerow(["author"] +["subreddit_id"] + ["t_level_comment"] + ["t_comment"] +["t_timestamp"] +["t_parent_id"] +["t_comment_score"] +["t_subreddit_id2"])

	subreddits = pd.read_csv(input1)
	for subreddit_id in subreddits.subreddit_id:
		submission = reddit.submission(id = subreddit_id)
		try:
			comments_l = [[top_level_comment.author, subreddit_id, top_level_comment.id, top_level_comment.body, __get_date(top_level_comment.created), top_level_comment.parent_id, top_level_comment.score, top_level_comment.subreddit_id] for top_level_comment in submission.comments.list()]
		except AttributeError:
			continue
		writer_top.writerows(comments_l)


collect_all_comments('divorce.csv', 'divorce_comments.csv' )


#get all comments from a user, limit to 1000

def collect_author_comments(input1,output):
	f = open(output , 'w')
	writer_top = csv.writer(f, delimiter = ',',quoting=csv.QUOTE_MINIMAL)
	writer_top.writerow(["author"] +["comments"] + ["comment_id"] + ["parent_id"]+["t_timestamp"] +["likes"] +["num_comments"] +["score"] +['subreddit_id'] +['subreddit_title'] +['subreddit'])

	subreddits = pd.read_csv(input1)
	for author in subreddits.author:
		if author != 'nan' or 'NaN':
			user = reddit.redditor(str(author))
			print(user)
			try:
				comments_l = [[author_comment.author, author_comment.body, author_comment.id, author_comment.parent_id, __get_date(author_comment.created), author_comment.likes, author_comment.num_comments, author_comment.score, author_comment.subreddit_id,author_comment.link_title,author_comment.subreddit] for author_comment in user.comments.new(limit=None)]
			except:
				continue
			writer_top.writerows(comments_l)

collect_author_comments('data_liwc_emotions_search.csv', 'infidelity_author_comments.csv')

#getting all the posts from a user
def collect_author_Posts(input1,output):
	f = open(output , 'w')
	writer_top = csv.writer(f, delimiter = ',',quoting=csv.QUOTE_MINIMAL)
	writer_top.writerow(["author"] +["text"] + ["subreddit"] + ["title"]+["t_timestamp"] )

	subreddits = pd.read_csv(input1)
	for author in subreddits.author:
		if author != 'nan' or 'NaN':
			user = reddit.redditor(str(author))
			print(user)
			try:
				comments_l = [[authorPosts.author,authorPosts.selftext, authorPosts.subreddit, authorPosts.title, __get_date(authorPosts.created)] for authorPosts in user.submissions.new(limit=None)]
			except:
				continue
			writer_top.writerows(comments_l)

collect_author_Posts('SuicideWatch.csv', 'output.csv')



#send message to user
reddit = praw.Reddit(client_id='wyESgnQhXG9mNQ',
                     client_secret='079juG33U8gsRx9HUujEQngvHic',
                     user_agent='collect reddits',
                     username ='luciasalar',
                     password ='myjsybjwqt888')

name_list =['luciasalar']

for i in name_list:
	reddit.redditor(str(i)).message('hi from Lucia', 'Hey, how are you?')



for author in subreddits.author[1:3]:
	if author != 'nan' or 'NaN':
		user = reddit.redditor(str(author))
		for i in user.comments.new():
			print(i.parent_id)


import datetime as dt
from psaw import PushshiftAPI

api = PushshiftAPI(reddit)


start_epoch=int(dt.datetime(2019, 1, 1).timestamp())
result = list(api.search_submissions(before=start_epoch,
                            subreddit='SuicideWatch',
                            filter=['url','author', 'title', 'subreddit'],
                            limit=1000))


def collect_posts(input1,output):
	f = open(output , 'w')
	writer_top = csv.writer(f, delimiter = ',',quoting=csv.QUOTE_MINIMAL)
	writer_top.writerow(["title"] + ["text"] + ["author"] + ["score"] + ["num_comments"]+["url"] + ["timestamp"] +["subreddit_id"] + ["subreddit"])

	
	for subreddit_id in result:
		submission = reddit.submission(id = subreddit_id)
		
		post_l = [[submission.title, submission.selftext, submission.author, submission.score, submission.num_comments, submission.url, __get_date(submission.created), submission.id,submission.subreddit]]
		
		writer_top.writerows(post_l)


collect_posts(result, 'SuicideWatch.csv' )


