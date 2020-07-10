# Files

### LDA topic modeling
* step 1: clean text 
* step 2: divide dataset according to timeline, suggested season timeline:
	
	1. Spring runs from March 1 to May 31;
	2. runs from June 1 to August 31;
	3. Fall (autumn) runs from September 1 to November 30; and
	4. Winter runs from December 1 to February 28 (February 29 in a leap year).

* step 3: extract entities
* step 4: LDA topic modeling. The pipeline optimize paramters according to coherence score

### Time series

* step 1: stats_activity.py process data, convert timestamp, truncate data according to years
* step 2: run time series analysis with stats.Rmd


### Count keywords
* to understand the number of documents that mention covid, we count whether each document contain covid, coronavirus, corona, virus and compute the percentage   Count_keywords.py

### Count topic percentage
* we compute the percentage of certain topic in a dataset topic_percent.py