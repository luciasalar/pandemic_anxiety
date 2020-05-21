# Data Collection

## collect posts

### step 1: collect post ids run collect_postids.py 
Set the subreddit names in evn/experiment.yaml

### step 2: collect posts using the post id lists run collect_posts.py

the problem of the reddict api is that it breaks very often, therefore, we set a limit in the postid number, we only collect 500 ids at a time, save it to different files, and combine these files later on.

The script loop through a set of subreddit names, choose the most active one and check the timestampe of its last entry to decide the start date of the next loop

## collect comments

run collect_comments.py


## collect all the posts from one author

run collect_author_posts.py