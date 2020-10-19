import pandas as pd 
from sklearn.metrics import cohen_kappa_score
import glob
import os
import csv
import krippendorff



class AnnotationAgreement:
    def __init__(self, filename):
        """Define varibles."""
        #self.path = '/disk/data/share/s1690903/pandemic_anxiety/data/annotations/'
        self.data = pd.read_csv(filename)

    def compute_kappa(self):
        """Compute Krippendorff's alpha is applicable to any number of coders, 
        each assigning one value to one unit of analysis, to incomplete (missing) data, 
        to any number of values available for coding a variable, to binary, nominal, 
        ordinal, interval, ratio, polar, and circular metrics """
        # recode data group 1, 2
        
        #self.data.iloc[:, [4]] = self.data.iloc[:, [4]].replace(1, 2)
        #self.data.iloc[:, [7]] = self.data.iloc[:, [7]].replace(1, 2)
       
        new_data = self.data[self.data.iloc[:, 4].notna()]
        new_data = self.data[self.data.iloc[:, 7].notna()]
        #print(new_data.iloc[:, [4]])
        

        kappa = cohen_kappa_score(new_data.iloc[:, [4]], new_data.iloc[:, [7]])
       
        return kappa

    def agreement(self):
        """Compute agreement """

        stat = self.data.agreement.value_counts()
        agg = stat[1] / (stat[1] + stat[0])
        return agg

#if __name__ == "__main__":
path_input = '/disk/data/share/s1690903/pandemic_anxiety/data/annotations/topics/'
path_result = '/disk/data/share/s1690903/pandemic_anxiety/results/'
#anxiety = pd.read_csv(path_input + 'Anxietyhelp.csv', header='infer')


allfiles = glob.glob(os.path.join(path_input, '*'))

f = open(path_result + 'annotation_aggreement_topic.csv', 'a', encoding='utf-8-sig')
writer_top = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
writer_top.writerow(['dim_kappa'] + ['topic_agreement'] + ['annotator1'] + ['annotator2'] + ['subreddit'] + ['kappa'])
for file in allfiles:
    # compute agreement
    ann = AnnotationAgreement(file)
    kappa = ann.compute_kappa()
    agreement = ann.agreement()
   
    
    # write result into file
    result_row = [[kappa, agreement, ann.data.columns[4], ann.data.columns[7], file.split('/')[8].split('.')[0], 'cohen_group12']]
    writer_top.writerows(result_row)

f.close()







