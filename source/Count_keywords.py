from lda_topic import *


#here we count whether COVID / coronavirus appear in the documents

def load_experiment(path_to_experiment):
    """load experiment"""
    data = yaml.safe_load(open(path_to_experiment))
    return data

class keywordText:
    """Collect posts via pushshift."""
    def __init__(self, start_year, start_month, start_day, end_year, end_month, end_day, subreddit):
        '''define the main path'''
        self.datapath = '/disk/data/share/s1690903/pandemic_anxiety/data/postids/'
        self.subreddit = subreddit
        self.pt = ProcessText('posts/{}_postids_posts.csv'.format(self.subreddit))
        self.cleaned_text = self.pt.simple_preprocess()
        self.start_year = start_year
        self.start_month = start_month
        self.start_day = start_day
        self.end_year = end_year
        self.end_month = end_month
        self.end_day = end_day


    def get_keyword(self, text, keyword_list):

        """return dictionary contains keywords only """
        mydict = lambda: defaultdict(mydict)
        extracted_text = mydict()
        for keyword in keyword_list.split(','):
            for k, v in text.items():
                if keyword in v['text'].split():
                    extracted_text[k]['text'] = v['text']
                    extracted_text[k]['time'] = v['time']

        return extracted_text

    def get_percentage(self, cleaned_text, extracted_text):

        """Compute percentage of text contain keyword """
        percentage = len(extracted_text) / len(cleaned_text)

        return percentage


    def run_kw_percentage(self):

        #pt = ProcessText('posts/{}_postids_posts.csv'.format(subreddit)) #'Anxiety'
        #cleaned_text = pt.simple_preprocess()# Simple text process: lower case, remove punc.
        # we only need the spring data
        spring = self.pt.split_timeline(self.cleaned_text, '{}/{}/{}'.format(self.start_month, self.start_day, self.start_year), '{}/{}/{}'.format(self.end_month, self.end_day, self.end_year))
        extracted_text = self.get_keyword(spring, 'covid,coronavirus,corona,virus,covid-19')
        per = self.get_percentage(self.cleaned_text, extracted_text)
        print(per)

        return per





kw = keywordText(2020, 2, 1, 2020, 5, 31, 'HealthAnxiety')
per = kw.run_kw_percentage()

evn_path = '/disk/data/share/s1690903/pandemic_anxiety/evn/'
evn = load_experiment(evn_path + 'experiment.yaml')

#run it on all subreddits
subreddits = evn['subreddits']['subs_all']
for sub in subreddits:
    print(sub)
    kw = keywordText(2020, 2, 1, 2020, 5, 31, sub)
    per = kw.run_kw_percentage()



























