import pandas as pd 
import datetime
from datetime import datetime
from sklearn.metrics import classification_report
import os
import csv

# Here we want to check the false positive rate of results
class Check_result:

    def __init__(self, filename, resultFile, sel_col, time_from, time_until):
        self.path1 = '/disk/data/share/s1690903/pandemic_anxiety/data/anno_test/'
        self.path2 = '/disk/data/share/s1690903/pandemic_anxiety/results/test_set_results/'
        self.file = filename
        self.result = resultFile
        self.sel_col = sel_col
        self.time_from = time_from
        self.time_until = time_until

    def read_file(self):
        """Get test set id and time """

        # read annotated data
        file = pd.read_csv(self.path1 + self.file)
        # recode anxiety level
        file['anxiety'] = file['anxiety'].replace(0, 1)

        return file

    def read_result(self):
        # read result file
        result = pd.read_csv(self.path2 + self.result)

        return result

    def merge_time(self):
        # merge prediction result with time
        file = self.read_file()

        result = self.read_result()
        
        # merge results
        time = file[['post_id', 'time', self.sel_col]]
        time_result = time.merge(result, on='post_id')

        return time_result

    def select_time(self, start_date, end_date):
        #select the timeline to check result

        data = self.merge_time()
        data['time'] = pd.to_datetime(data['time'], format='%m/%d/%Y/%H:%M:%S').dt.date
        startdate = pd.to_datetime(start_date).date()
        enddate = pd.to_datetime(end_date).date()
        data2 = data.loc[data['time'].between(startdate, enddate, inclusive=False)]

        return data2

    def get_classification_report(self):
        # get classification report for results
       
        # get classication report according to time
        selected_time = self.select_time(self.time_from, self.time_until)
        selected_time = selected_time.fillna(0)
        report = classification_report(selected_time[self.sel_col], selected_time['y_pred'], digits=2)

        return report

    def write_result(self):

        file_exists = os.path.isfile(self.path2 + 'check_result.csv')
        f = open(self.path2 + 'check_result.csv', 'a')
        writer_top = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)

        report = self.get_classification_report()

        if not file_exists:
            writer_top.writerow(['time_from'] + ['time_until'] + ['report'] + ['class'])
            result_row = [[self.time_from, self.time_until, report, self.sel_col]]
            writer_top.writerows(result_row)
            f.close()

        else:
            result_row = [[self.time_from, self.time_until, report, self.sel_col]]
            writer_top.writerows(result_row)
            f.close()

        return report




cr = Check_result(filename='all_data_text.csv', resultFile='test_result_anxiety.csv', sel_col='anxiety', time_from = '2020-2-1', time_until = '2020-8-30')

report = cr.write_result()
print(report)

cr = Check_result(filename='all_data_text.csv', resultFile='test_result_anxiety.csv', sel_col='anxiety', time_from = '2020-9-1', time_until = '2020-10-30')

report = cr.write_result()
print(report)




































