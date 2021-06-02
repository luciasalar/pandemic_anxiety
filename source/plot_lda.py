import pandas as pd 
import datetime
import seaborn as sns
import matplotlib.pyplot as plt

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

    def select_time_subreddit(self, year, subreddit):
        """Select posts according to condition """

        lda_time = self.match_time()
        lda_time['time'] = pd.to_datetime(lda_time['time'], format='%m/%d/%Y/%H:%M:%S')

        filtered = lda_time
        filtered.index = pd.to_datetime(filtered['time'], format='%m/%d/%Y/%H:%M:%S')
        #filtered = filtered[(filtered['time'] > '1/1/{}'.format(year)) & (filtered['time'] < '12/31/{}'.format(year))]
        filtered = filtered[(filtered['time'] > '1/1/2019') & (filtered['time'] < '12/31/{}'.format(year))]
        filtered = filtered.drop_duplicates(subset=['post_id'])

        filtered = filtered[filtered['subreddit'] == subreddit]

        return filtered

    def select_time_subreddit2(self, year):
        """Select posts according to condition """

        lda_time = self.match_time()
        lda_time['time'] = pd.to_datetime(lda_time['time'], format='%m/%d/%Y/%H:%M:%S')

        filtered = lda_time
        filtered.index = pd.to_datetime(filtered['time'], format='%m/%d/%Y/%H:%M:%S')
        #filtered = filtered[(filtered['time'] > '1/1/{}'.format(year)) & (filtered['time'] < '12/31/{}'.format(year))]
        filtered = filtered[(filtered['time'] > '1/1/2019') & (filtered['time'] < '12/31/{}'.format(year))]
        filtered = filtered.drop_duplicates(subset=['post_id'])

        filtered = filtered[filtered['subreddit'] != 'COVID19_support']

        return filtered

    def plot_topic(self, topic_name, year, subreddit):

        filtered = self.select_time_subreddit(year, subreddit)
        grouped = filtered.groupby(by=[(filtered.index.year)]).mean()
        grouped['month'] = grouped.index
        sns_plot = sns.lineplot(data=grouped, x="month", y=topic_name, label='topic_{}_{}_{}'.format(topic_name, year, subreddit))

        fig = sns_plot.get_figure()
        fig.savefig(self.outputP + "lda_{}_{}_{}.png".format(topic_name, year, subreddit))
        
    def plot_topic_subreddit(self, topic_num, subreddit, topic_name):
        """Plot topics in each sub """

        # filter data by year and group data
        #filtered = p.select_time_subreddit(2020, subreddit)
        filtered = p.select_time_subreddit2(2020)
        grouped = filtered.groupby(by=[(filtered.index.year), (filtered.index.month)]).mean()
        grouped['year'] = grouped.index
        grouped['year'] = grouped['year'].apply(lambda x: ''.join(str(x)))

        sns.set(rc={'figure.figsize': (11.7, 8.40)})
        #set axis labels
        #sns_plot = sns.lineplot(data=grouped, x='year', y=topic_num, label='{}'.format(subreddit))
        sns_plot = sns.lineplot(data=grouped, x='year', y=topic_num)
        sns_plot.set_xticklabels(labels=['Jan-2019', 'Feb-2019', 'Mar-2019', 'Apr-2019', 'May-2019', 'Jun-2019', 'Jul-2019', 'Aug-2019', 'Sep-2019', 'Oct-2019','Nov-2019', 'Dec-2019','Jan-2020', 'Feb-2020', 'Mar-2020', 'Apr-2020', 'May-2020', 'Jun-2020', 'Jul-2020', 'Aug-2020', 'Sep-2020', 'Oct-2020','Nov-2020', 'Dec-2020',], rotation=45)
        sns_plot.set(ylabel='mean topic score', xlabel='')
        sns_plot.set_title(topic_name)

        fig = sns_plot.get_figure()
        fig.savefig(p.outputP + "lda_{}.png".format(topic_name))

        return sns_plot

    def plot_topic_time_series(self, topic_name):
        """Plot topic in each subreddit and rolling average """

        filtered = self.select_time_subreddit2(2020)
        filtered = filtered.rename(columns={'47': 'anxiety', '14':'hygiene', '23':'panic_attack','26':'death','29':'doctor_visit','32':'social_interaction','4':'ailment','49':'travel','62':'drinking','63':'school','70':'dating','79':'media', '35':'sleep', '72':'drug'})

        grouped = filtered.groupby(by=[(filtered.index.year), (filtered.index.month), (filtered.subreddit)]).mean()
        grouped['year'] = grouped.index
        grouped['subreddit'] = grouped.index.get_level_values(2)
        grouped['year'] = grouped['year'].apply(lambda x: str(x[0]) + '-' + str(x[1]))

        sns.set(rc={'figure.figsize': (11.7, 8.40)})
        grouped['year2'] = pd.to_datetime(grouped['year'], format='%Y-%m')
        #set axis labels
        sns_plot = sns.lineplot(data=grouped, x='year2', y=topic_name, hue='subreddit', palette="husl", style='subreddit', linewidth=3)

        #plot sample mean
        grouped2 = filtered.groupby(by=[(filtered.index.year), (filtered.index.month)]).mean()
        grouped2['year'] = grouped2.index
        grouped2['year'] = grouped2['year'].apply(lambda x: str(x[0]) + '-' + str(x[1]))
        grouped2['year'] = pd.to_datetime(grouped2['year'], format='%Y-%m')
        #sns_plot = sns.lineplot(data=grouped2, x='year', y='70', linewidth=4, palette=['red'])
        #There are several ways to think about identifying trends in time series. One popular way is by taking a rolling average, which means that, for each time point, you take the average of the points on either side of it.
        sns_plot = grouped2.set_index('year')[topic_name].rolling(2).mean().plot(linewidth=6, color = 'blue', style='--', ax=sns_plot)
        #sns_plot.set(ylabel='mean topic score', xlabel='')
        sns_plot.tick_params(labelsize=12)
        sns_plot.set_xlabel('')
        sns_plot.set_ylabel('mean topic score', fontsize=18)
        sns_plot.set_title(topic_name, fontsize=20)
        #sns_plot.set_xticklabels(sns_plot.get_xticklabels(), rotation=45)

        fig = sns_plot.get_figure()
        fig.savefig(self.outputP + "lda_{}.png".format(topic_name))

        plt.close()

        return grouped2

    def plot_subreddit_topic(self, var_list, subreddit):
        """Plot all topics in one sub """

        filtered = p.select_time_subreddit(2020, subreddit)
        print(filtered.shape)
        
        filtered = filtered.rename(columns={'47': 'anxiety','14': 'hygiene', '23':'panic_attack','26':'death','29':'physical_symptoms','32':'social_interaction','4':'ailment','49':'travel','62':'drinking','63':'school','70':'dating','79':'media','35':'sleep','72':'drug'})

        melted = filtered.melt(id_vars=['post_id', 'time'], value_vars=var_list, var_name='topics', value_name='values')
        grouped2 = melted.groupby(by=[(melted['time'].dt.year), (melted['time'].dt.month), (melted['topics'])]).mean()
        grouped2['year'] = grouped2.index
        grouped2['year'] = grouped2['year'].apply(lambda x: str(x[0]) + '-' + str(x[1]))
        grouped2['topics'] = grouped2.index.get_level_values(2)
        grouped2['year'] = pd.to_datetime(grouped2['year'], format='%Y-%m')

        sns.set(rc={'figure.figsize': (11.7, 8.40)})
        sns_plot = sns.lineplot(data=grouped2, x='year', y='values', hue='topics', palette="rocket", style='topics', linewidth = 3)

        #sns_plot.set(ylabel='mean topic score', xlabel='')
        sns_plot.tick_params(labelsize=16, rotation=45)
        sns_plot.set_xlabel('')
        sns_plot.set_ylabel('mean topic score', fontsize=18)
        sns_plot.set_title(subreddit, fontsize=20)
        #sns_plot.set_xticklabels(sns_plot.get_xticklabels(), rotation=45)

        fig = sns_plot.get_figure()
        fig.savefig(p.outputP + "lda_{}.png".format(subreddit))

        plt.close()


# plot topic change over years


# plot precentage of symptom topic in each month
inputP = '/disk/data/share/s1690903/pandemic_anxiety/data/'
outputP = '/disk/data/share/s1690903/pandemic_anxiety/results/lda_plots/100_topics/'
LDAfile = 'dominance_lda_all_data.csv'
dataFile = 'all_sub_posts.csv'
subreddit = 'HealthAnxiety'

p = PlotLDA(inputP=inputP, LDAfile=LDAfile, dataFile=dataFile, subreddit=subreddit, outputP=outputP)


#figure_list = '47,14,23,26,29,32,4,49,62,63,70,79'
#topic_names = 'anxiety, hygiene, panic_attack, death, doctor_visit, social_interaction, ailment, travel, drinking, school, dating, media'

# plot each topic
# for name in topic_names.split(', '):
#     sns_plot = p.plot_topic_time_series(topic_name=name)
sns_plot = p.plot_topic_time_series(topic_name='anxiety')
#sns_plot = p.plot_topic_time_series(topic_name='sleep')

# plot each sub
# var_list = ['hygiene', 'panic_attack', 'physical_symptoms', 'school', 'dating', 'sleep']
var_list = ['hygiene']
subreddit_list = ['socialanxiety', 'AnxietyDepression', 'OCD', 'Anxietyhelp', 'HealthAnxiety', 'Anxiety']
for sub in subreddit_list:
    p.plot_subreddit_topic(var_list, sub)







