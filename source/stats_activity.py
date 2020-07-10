import pandas as pd 
from lda_topic import ProcessText
import datetime as dt
from datetime import datetime 
import csv
from ruamel import yaml
import os


def load_experiment(path_to_experiment):
    """load experiment"""
    data = yaml.safe_load(open(path_to_experiment))
    return data

class ActivityLevel:
    #here we count the activity level according to season timeline

    def __init__(self, filename):
        """Define varibles."""
        self.path_data = '/disk/data/share/s1690903/pandemic_anxiety/data/'
        self.path_result = '/disk/data/share/s1690903/pandemic_anxiety/results/stats/'
        #self.data = pd.read_csv(self.path_data + 'posts/{}_postids_posts.csv'.format(filename))
        
    def get_activity(self, subsetdata):
        """Here we compute the activity level, averaged number of posts per day """

        # remove duplicates
        data = subsetdata.drop_duplicates(subset='post_id', keep='first', inplace=False)
        # count post in each day
        data['time'] = pd.to_datetime(data['time'], format="%m/%d/%Y/%H:%M:%S")
        time = data['time'].dt.normalize().value_counts()
        # divide posts by the number of days
        activity = time.sum() / len(time)
        # get data for plotting
        plot_data = time.to_frame()
        plot_data.columns.values[0] = "count"
        plot_data['time'] = plot_data.index

        #
        return activity, plot_data

    def get_activity_timeline(self, filename, year):
        """Get activity level by timeline """
        pt = ProcessText('posts/{}_postids_posts.csv'.format(filename))
        data_dict = pt.data_dict()
        spring = pt.split_timeline(data_dict, '3/1/{}'.format(year), '5/31/{}'.format(year))
        summer = pt.split_timeline(data_dict, '6/1/{}'.format(year), '8/31/{}'.format(year))
        fall = pt.split_timeline(data_dict, '9/1/{}'.format(year), '11/30/{}'.format(year))
        winter = pt.split_timeline(data_dict, '12/1/{}'.format(year), '2/28/{}'.format(year + 1))
        
        # convert output to df format
        spring_df = pd.DataFrame.from_dict(spring, orient='index')
        spring_df['post_id'] = spring_df.index

        summer_df = pd.DataFrame.from_dict(summer, orient='index')
        summer_df['post_id'] = summer_df.index

        fall_df = pd.DataFrame.from_dict(fall, orient='index')
        fall_df['post_id'] = fall_df.index

        winter_df = pd.DataFrame.from_dict(winter, orient='index')
        winter_df['post_id'] = winter_df.index

        return spring_df, summer_df, fall_df, winter_df

  
def stats_loop(path_result, sub, year):
    ac = ActivityLevel(sub)
    spring_df, summer_df, fall_df, winter_df = ac.get_activity_timeline(sub, year)

    if len(spring_df) > 0:
        spring_activity, spring_plot_data = ac.get_activity(spring_df)
        spring_plot_data.to_csv(path_result + 'activity/plot_spring_{}_{}.csv'.format(sub, year))
    else:
        spring_activity = 0

    if len(summer_df) > 0:
        summer_activity, summer_plot_data = ac.get_activity(summer_df)
        summer_plot_data.to_csv(path_result + 'activity/plot_summer_{}_{}.csv'.format(sub, year))
        increase_spr_t_sum = (summer_activity - spring_activity) / summer_activity
    else:
        summer_activity = 0
        increase_spr_t_sum = 0

    if len(fall_df) > 0:
        fall_activity, fall_plot_data = ac.get_activity(fall_df)
        fall_plot_data.to_csv(path_result + 'activity/plot_fall_{}_{}.csv'.format(sub, year))
        increase_sum_t_fal = (fall_activity - summer_activity) / fall_activity
    else:
        fall_activity = 0
        increase_sum_t_fal = 0

    if len(winter_df) > 0:
        winter_activity, winter_plot_data = ac.get_activity(winter_df)
        winter_plot_data.to_csv(path_result + 'activity/plot_winter_{}_{}.csv'.format(sub, year))
        increase_fal_t_win = (winter_activity - fall_activity) / winter_activity
    else:
        winter_activity = 0
        increase_fal_t_win = 0

    file_exists = os.path.isfile(path_result + 'activity/activity.csv')
    if not file_exists:
        f = open(path_result + 'activity/activity.csv', 'a', encoding='utf-8-sig')
        writer_top = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer_top.writerow(['year'] + ['sub'] + ['spring'] + ['summer'] + ['fall'] + ['winter'] + ['increase_spr_t_sum'] + ['increase_sum_t_fal'] + ['increase_fal_t_win '])
        f.close()
  
    f = open(path_result + 'activity/activity.csv', 'a', encoding='utf-8-sig')
    writer_top = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    result_row = [[year, sub, spring_activity, summer_activity, fall_activity, winter_activity,increase_spr_t_sum, increase_sum_t_fal, increase_fal_t_win]]
    writer_top.writerows(result_row)
    f.close()

    return increase_spr_t_sum

#load experiment
evn_path = '/disk/data/share/s1690903/pandemic_anxiety/evn/'
evn = load_experiment(evn_path + 'experiment.yaml')

path_result = '/disk/data/share/s1690903/pandemic_anxiety/results/stats/'
#w = stats_loop(path_result, 'Anxiety', 2019)

#now we loop through all the subs
# subreddits = evn['subreddits']['subs_all']
# for sub in subreddits:
#     print(sub)
#     w = stats_loop(path_result, sub, 2017)



w = stats_loop(path_result, 'COVID19_support', 2020)







