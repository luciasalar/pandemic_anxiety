


import pandas as pd 

from collections import defaultdict



class CombineResults:
    def __init__(self, var_list):
        self.path = '/disk/data/share/s1690903/pandemic_anxiety/results/prediction_result_2500/'
        self.var_list = var_list

    def read_file(self, col):
        """Read result """
   
        file = pd.read_csv(self.path + 'prediction_sample_{}.csv'.format(col))

        file = file.drop_duplicates(subset='post_id', keep='first', inplace=False)
        file = file[~file['text'].isin(['[removed]'])]
        file = file[~file['text'].isin(['[deleted]'])]
        file = file.dropna(subset=['text']) #5234 rows
        file = file[['post_id', 'prediction_{}'.format(col)]]

        return file

    def merge_results(self):

        d = defaultdict(dict)

        for col in self.var_list: 
            file = self.read_file(col)
            print(col)

            if not d:# check if dict is empty
                for post_id, prediction in zip(file['post_id'], file['prediction_{}'.format(col)]):
                    d[post_id]['prediction_{}'.format(col)] = prediction

            else:
                for post_id, prediction in zip(file['post_id'], file['prediction_{}'.format(col)]):
                    for k, v in d.items():
                        if post_id == k:
                            d[k]['prediction_{}'.format(col)] = prediction

        df = pd.DataFrame.from_dict(d, orient='index')
        df['post_id'] = df.index
        df.to_csv(self.path + 'all_prediction_results.csv', encoding = "ISO-8859-1")

        return df

var_list = ['anxiety', 'financial_career', 'quar_social', 'health_infected', 'break_guideline', 'health_work', 'mental_health', 'death', 'travelling', 'future']

#var_list = ['anxiety', 'financial_career']
c = CombineResults(var_list)
d = c.merge_results()
#df = pd.DataFrame.from_dict(d, orient='index')



