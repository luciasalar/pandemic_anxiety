# Data Collection

## collect posts

### step 1: collect post ids run collect_postids.py 
Set subreddit names in evn/experiment.yaml

### step 2: collect posts using the post id lists run collect_posts.py

the problem of the reddict api is that it breaks very often, therefore, we set a limit to the postid count, we only collect 1000 ids for each iteration and we can set how many iterations we want

The script loop through a set of subreddit names, set these names in experiment.yaml

## collect comments

run collect_comments.py


## collect all the posts from one author

run collect_author_posts.py