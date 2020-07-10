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
     

        
        

#tp = TopicPercent()

#
#tp = TopicPercent('dominance_fall_2019_10_Anxiety.csv')

path_input = '/disk/data/share/s1690903/pandemic_anxiety/results/lda_results/dominance_topic_doc_level/'
#allfiles = [f for f in listdir(tp.path_input) if isfile(join(tp.path_input, f))]
allfiles = glob.glob(os.path.join(path_input, '*'))
for file in allfiles:
    print(file)
    tp = TopicPercent(file)
    dominance = tp.dominant_topic_percent()
    prevalence = tp.get_topic_prevalence()




#topic0 = len(tp.lda_results.loc[tp.lda_results['Dominant_Topic'] == 0]) / len(tp.lda_results)
# d = {}
# for i in range(5,15):
#     print(i)
#     d["topic" + str(i)] = len(tp.lda_results.loc[tp.lda_results.iloc[:, i] > 0]) / len(self.lda_results)
    





























