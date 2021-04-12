import pandas as pd 
import datetime
import seaborn as sns

class PlotLDA:
    """Collect posts via pushshift."""
    def __init__(self, inputP, outputP, LDAfile, dataFile, subreddit):
        '''define the main path'''

        self.inputP = inputP
        self.LDAFile = LDAfile
        self.dataFile = dataFile
        self.subreddit = subreddit
        self.outputP = outputP
    
    def read_file(self):
        """Read LDA topic file."""

        LDAfile = pd.read_csv(self.inputP + self.LDAFile)
        datafile = pd.read_csv(self.inputP + self.dataFile)

        return LDAfile, datafile

    def match_time(self):
        """Match files with timeline."""

        LDAfile, datafile = self.read_file()
        datafile = datafile[['post_id', 'time', 'subreddit']]

        lda_time = pd.merge(LDAfile, datafile, on='post_id', how='inner')

        return lda_time

    def select_time_subreddit(self, year):
        """Select posts according to condition """

        lda_time = self.match_time()
        lda_time['time'] = pd.to_datetime(lda_time['time'], format='%m/%d/%Y/%H:%M:%S')

        filtered = lda_time
        filtered.index = pd.to_datetime(filtered['time'], format='%m/%d/%Y/%H:%M:%S')
        filtered = filtered[(filtered['time'] > '1/1/{}'.format(year)) & (filtered['time'] < '12/31/{}'.format(year))]

        filtered = filtered[filtered['subreddit'] == self.subreddit]

        return filtered

    def plot_topic(self, topic_name, year):

        filtered = self.select_time_subreddit(year)
        grouped = filtered.groupby(by=[filtered.index.month]).mean()
        grouped['month'] = grouped.index
        sns_plot = sns.lineplot(data=grouped, x="month", y=topic_name, label='topic_name_{}'.format(year))

        fig = sns_plot.get_figure()
        fig.savefig(self.outputP + "lda_{}_{}.png".format(topic_name, year))
        



# plot topic change over years



# plot precentage of symptom topic in each month
inputP = '/disk/data/share/s1690903/pandemic_anxiety/data/'
outputP = '/disk/data/share/s1690903/pandemic_anxiety/results/lda_plots/60_topics/'
LDAfile = 'dominance_lda_all_data.csv'
dataFile = 'all_sub_posts.csv'

subreddit = 'HealthAnxiety'

p = PlotLDA(inputP=inputP, LDAfile=LDAfile, dataFile=dataFile, subreddit=subreddit, outputP=outputP)

lda_time = p.match_time()
lda_time.to_csv(inputP + 'temp.csv')

lda_time['time'] = pd.to_datetime(lda_time['time'], format='%m/%d/%Y/%H:%M:%S')
filtered = lda_time[(lda_time['time'] > '1-1-2019')]

filtered1 = p.select_time_subreddit(2019)
filtered2 = p.select_time_subreddit(2020)

#grouped = filtered1.groupby(by=[filtered1.index.month]).mean()

for i in range(0, 61):
    p.plot_topic(i, 2019)
    p.plot_topic(i, 2020)
#grouped = lda_time.groupby(by=[lda_time.index.month]).mean()


#20, 26, 3, 29















