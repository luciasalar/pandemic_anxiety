import pandas as pd
import csv
import numpy as np
from os import listdir
import glob
import os

# this script counts the percentage of topics in each doc

class TopicPercent:

    def __init__(self, filename):

        self.path_input = '/disk/data/share/s1690903/pandemic_anxiety/results/lda_results/dominance_topic_doc_level/'
        self.filename = filename.split('/')[9]
        self.lda_results = pd.read_csv(filename)
        self.path_output = '/disk/data/share/s1690903/pandemic_anxiety/results/lda_results/'

    def dominant_topic_percent(self):
        """Here we get the dominant topic of each doc and compute the percentage of dominant topic """
        count = self.lda_results['Dominant_Topic'].value_counts() 

        if 0 in count.index:
            topic0 = count[0] / len(self.lda_results)
        else:
            topic0 = None

        if 1 in count.index:
            topic1 = count[1] / len(self.lda_results)
        else:
            topic1 = None

        if 2 in count.index:
            topic2 = count[2] / len(self.lda_results)
        else:
            topic2 = None

        if 3 in count.index:
            topic3 = count[3] / len(self.lda_results)
        else:
            topic3 = None

        if 4 in count.index:
            topic4 = count[4] / len(self.lda_results)
        else:
            topic4 = None

        if 5 in count.index:
            topic5 = count[5] / len(self.lda_results)
        else:
            topic5 = None

        if 6 in count.index:
            topic6 = count[6] / len(self.lda_results)
        else:
            topic6 = None

        if 7 in count.index:
            topic7 = count[7] / len(self.lda_results)
        else:
            topic7 = None

        if 8 in count.index:
            topic8 = count[8] / len(self.lda_results)
        else:
            topic8 = None

        if 9 in count.index:
            topic9 = count[9] / len(self.lda_results)
        else:
            topic9 = None

        if 10 in count.index:
            topic10 = count[10] / len(self.lda_results)
        else:
            topic10 = None

        if 11 in count.index:
            topic11 = count[11] / len(self.lda_results)
        else:
            topic11 = None

        if 12 in count.index:
            topic12 = count[12] / len(self.lda_results)
        else:
            topic12 = None

        if 13 in count.index:
            topic13 = count[13] / len(self.lda_results)
        else:
            topic13 = None

        if 14 in count.index:
            topic14 = count[14] / len(self.lda_results)
        else:
            topic14 = None

        file_exists = os.path.isfile(self.path_output + 'dominance_topic_percent.csv')
        if not file_exists:
            f = open(self.path_output + 'dominance_topic_percent.csv', 'a')
            writer_top = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            writer_top.writerow(["filename"] + ["topic0"] + ["topic1"] + ["topic2"] + ["topic3"] + ["topic4"] + ["topic5"] + ["topic6"] + ["topic7"] + ["topic8"] + ["topic9"] + ["topic10"] + ["topic11"] + ["topic12"] + ["topic13"]+ ["topic14"] + ["total_num"])

        f = open(self.path_output + 'dominance_topic_percent.csv', 'a')
        writer_top = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer_top.writerows([[self.filename, topic0, topic1, topic2, topic3, topic4, topic5, topic6, topic7, topic8, topic9, topic10, topic11, topic12, topic13, topic14, len(self.lda_results)]])
        f.close()



    def get_topic_prevalence(self):
        """Compute the occurances of topics in the dataset and count the percentage """
        prevalence_dict = {}
        for i in range(5, 20):
            if tp.lda_results.shape[1] > 20:
                prevalence_dict["topic" + str(self.lda_results.columns[i])] = len(self.lda_results.loc[self.lda_results.iloc[:, i] > 0]) / len(self.lda_results)

        prevalence = pd.DataFrame.from_dict(prevalence_dict, orient='index')
        prevalence.to_csv(self.path_output + "topic_prevalence_{}.csv".format(self.filename))

        return prevalence
     
class getTopicPosts:
    """Get 100 example post for each topic """
    def __init__(self, top_n, filename):

        #self.data_path = '/disk/data/share/s1690903/pandemic_anxiety/results/lda_results/dominance_topic_doc_level/new_timeline/dominance_covid_15_Anxiety.csv'
        self.path_posts = '/disk/data/share/s1690903/pandemic_anxiety/data/posts/'
        self.path_result = '/disk/data/share/s1690903/pandemic_anxiety/results/lda_topic_doc_example/'
        self.data = pd.read_csv(filename)
        self.top_n = 100

    #append all the files
    def merge_all_docs(self):
        
        appended_data = []
        #allfiles = [f for f in listdir(tp.path_input) if isfile(join(tp.path_input, f))]
        allfiles = glob.glob(os.path.join(self.path_posts, '*'))
        for filename in allfiles:
            #print(filename)
            file = pd.read_csv(filename)
            file = file.drop_duplicates(subset='post_id', keep="last")
            appended_data.append(file)

        appended_data = pd.concat(appended_data)
        return appended_data

    def get_ids(self, topic):
        """get id for top x documents"""

        # sort documents
        # find docs contain this topic
        larger_than_zero = self.data[self.data[topic] > 0]

        if len(larger_than_zero) > self.top_n:
            sorted_df = larger_than_zero.sort_values(by=topic, ascending=False)
            top_n_df = sorted_df.head(self.top_n)
        else:
            sorted_df = larger_than_zero.sort_values(by=topic, ascending=False)
            top_n_df = sorted_df

        result = top_n_df[['post_id', 'index']]

        return result

    def loop_topics(self, all_d, filename_topic):
        """Get top_n documents for all the topics"""

        f = open(self.path_result + 'topic_doc_{}.csv'.format(filename_topic), 'w', encoding='utf-8-sig')
        writer_top = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer_top.writerow(['topic_num'] + ['text'])

        topic_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']
        for topic in topic_list:
            ids = self.get_ids(topic)
            result = pd.merge(ids, all_d, on='post_id', how='left')
            for text in result.text:
                result_row = [[topic, text]]
                writer_top.writerows(result_row)
        f.close()



#match topic from files
path_topic_result = '/disk/data/share/s1690903/pandemic_anxiety/results/lda_results/dominance_topic_doc_level/new_timeline/'

def loop_all_doc():
    """Loop all the topic docs """
    allfiles = glob.glob(os.path.join(path_topic_result, '*'))
    for file in allfiles:
        print(file)
        getp = getTopicPosts(100, file)
        all_d = getp.merge_all_docs()
        getp.loop_topics(all_d, file.split('/')[10].split('.')[0])


loop_all_doc()

#d = getp.get_ids()
#docs = pd.merge(d, all_d, on='post_id', how='left')






#
#tp = TopicPercent('dominance_fall_2019_10_Anxiety.csv')
if __name__ == "__main__":
    path_input = '/disk/data/share/s1690903/pandemic_anxiety/results/lda_results/dominance_topic_doc_level/'
    #allfiles = [f for f in listdir(tp.path_input) if isfile(join(tp.path_input, f))]
    allfiles = glob.glob(os.path.join(path_input, '*'))
    for file in allfiles:
        print(file)
        tp = TopicPercent(file)
        dominance = tp.dominant_topic_percent()
        prevalence = tp.get_topic_prevalence()































