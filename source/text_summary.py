from gensim.summarization.summarizer import summarize
import pandas as pd 

"""Increase data by summary"""

path = '/disk/data/share/s1690903/pandemic_anxiety/data/anno_test/'
all_files_pd = pd.read_csv(path + 'all_data_text.csv', encoding="ISO-8859-1")

selected = all_files_pd.loc[all_files_pd['future'] > 0]

selected['text1'] = selected['text'].apply(lambda x: x.replace(",", "."))
selected['text1'] = selected['text1'].apply(lambda x: summarize(x))


for sent in selected['text'].head(10):
	sent = sent.replace(",", ".")
	s = summarize(sent)
	print(s)


print(summarize(text))



if len(str(x).split('.')) > 1 else False