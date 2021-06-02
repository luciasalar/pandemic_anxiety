# Project Description


# files 

## Folder

* Env: environment files
* Data Collection: scripts for data collection
* source: script files


## Data
AuthorAllP.csv: posts from all authors

AuthorAllP2.csv: posts from all authors

covid19_support_clean.csv


### annotations

annotation guideline: 
annotation_Guideline/annotation_anxiety.md

anno_test

posts: posts files from each subreddit

postid: collecting post ids according to certain timeline

anno_test/data: final annotation data

anno_text/all_data_text.csv: all annotated data with text

# scripts:

### data colection 

collect_author_posts.py: this script collect all the posts from author

collect_comments.py: collect all comments according to post id

collect_postid.py: collect posts according to set date, the script frist collect the post id, then collect the posts according to ids

collect_posts.py: collect post ids

### other /source

lda_topic.py: run LDA according to timeline

plot_lda.py: plot topic percentage before and after covid

seasonal_plot.rmd plot user activity on reddit

data_selection: select posts for annotation (we randomly selected 100 posts per month)

merge_pre_results.py: combine SOW prediction result

model_pipeline.py: prediction models on anxiety level and subject of anxiety

topic_percent.py: this script counts the percentage of topics in each doc

stats_activity.py: compute the activity level according to season timeline

### annotation

data/anno_test/post_anno_agreement.Rmd  compute post annotation agreement and highlight the differences

data/anno_test/post_analysis.Rmd  Plot anxiety level





