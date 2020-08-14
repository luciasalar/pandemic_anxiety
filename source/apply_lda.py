from lda_topic import *
import glob
import os


def load_experiment(path_to_experiment):
    """load experiment"""
    data = yaml.safe_load(open(path_to_experiment))
    return data


class LDAtrain:
    def __init__(self, train_text, alpha, beta, ProcessText):
        self.path_data = '/disk/data/share/s1690903/pandemic_anxiety/data/'
        self.train_text = train_text
        self.alpha = alpha
        self.beta = beta
        self.pt = ProcessText


    def run_LDA(self, path, text: typing.Dict[str, str], num_topic:int, alpha, beta):
        """Select the best lda model with extracted text 
        text: entities dictionary
        domTname:file name for the output
        """

        # convert data to dictionary format

        file_exists = os.path.isfile(path + 'lda_result_apply.csv')
        f = open(path + 'lda_result_apply.csv', 'a', encoding='utf-8-sig')
        writer_top = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        if not file_exists:
            writer_top.writerow(['a'] + ['b'] + ['coherence'] + ['time'] + ['topics'] + ['num_topics'])
       
        # run LDA with the reset values
        lda = LDATopic(text, num_topic, alpha, beta)
        model, coherence, all_lda_score_dfT, bow_corpus = lda.topic_modeling()
        #pprint(model.print_topics())

        #f = open(path + 'result/lda_result.csv', 'a')
        result_row = [[alpha, beta, str(datetime.now()), coherence, model.print_topics(num_topics=15), num_topic]]
        writer_top.writerows(result_row)

        f.close()
        gc.collect()
      
        return model


    def get_topic_covid_timeline(self, cleaned_text, year: int, num_topic: int, alpha_pre, beta_pre, alpha_co, beta_co) -> pd.DataFrame:
        """get topics according to season timeline 
        result saved in results/lda_results/
        """

        # here we can set the seasons
        precovid = self.pt.split_timeline(cleaned_text, '1/1/{}'.format(year), '12/31/{}'.format(year))
        precovid2 = self.pt.split_timeline(cleaned_text, '1/1/{}'.format(year - 1), '12/31/{}'.format(year-1))
        covid = self.pt.split_timeline(cleaned_text, '2/1/2020', '5/31/2020')
        precovid.update(precovid2)
     
       #run lda for each period
        
        entities = self.pt.extract_entities(covid)
        model_covid = run_LDA(self.pt.path_result, entities, num_topic, alpha_pre, beta_pre)

       
      
        return model_covid


    def lda_train(self):

        # extract entities
        #pt = ProcessText('posts/Anxiety_postids_posts.csv')  #which file doesnt matter in here
        entities_train = self.pt.extract_entities(self.train_text)
    
        #run model on train set
        model = self.run_LDA(self.pt.path_result, entities_train, 15, self.alpha, self.beta)

        return model, entities_train

class LDApredict:
    def __init__(self, entities_train, pre_text, ProcessText):
        self.path_data = '/disk/data/share/s1690903/pandemic_anxiety/data/'
        self.pre_text = pre_text
        self.entities_train = entities_train
        self.pt = ProcessText

      
    def lda_predict(self, model, subreddit, timeline):
         #which file doesnt matter in here
        entities_pre = self.pt.extract_entities(self.pre_text)
        #get the common corpus
        dictionary = gensim.corpora.Dictionary(self.entities_train.values())
        bow_corpus_common = [dictionary.doc2bow(doc) for doc in entities_pre.values()]

        #unseen_doc = bow_corpus_common
        #vector = model_precovid[0][unseen_doc]

        #Train the model with new documents, by EM-iterating over the corpus until the
        #topics converge, or until the maximum number of allowed iterations
        # is reached. corpus
        model.update(bow_corpus_common)
        #vector = model_covid[unseen_doc]

        #store result model with the best score
        lda = LDATopic(entities_pre, 15, 0.3, 0.9)
        id_l = lda.get_ids_from_selected(entities_pre)


        # get the score matrix
        lda_score_all = lda.get_score_dict(bow_corpus_common, model)

        all_lda_score_df = pd.DataFrame.from_dict(lda_score_all)
        all_lda_score_dfT = all_lda_score_df.T
        all_lda_score_dfT = all_lda_score_dfT.fillna(0)


        all_lda_score_dfT['post_id'] = id_l
        all_lda_score_dfT.to_csv(self.pt.path_result + 'lda_prediction/test_lda_predict_{}_{}.csv'.format(subreddit, timeline))

        return all_lda_score_dfT

def get_precovid_text(filename):
    """Get data from precovid timeline """

    pt = ProcessText('posts/{}_postids_posts.csv'.format(filename))
    cleaned_text = pt.simple_preprocess()
    precovid = pt.split_timeline(cleaned_text, '1/1/2019', '12/31/2019')
    precovid2 = pt.split_timeline(cleaned_text, '1/1/2018', '12/31/2018')
    precovid.update(precovid2)

    return precovid2

def get_covid_text(filename):
    """Get data from covid timeline """

    pt = ProcessText('posts/{}_postids_posts.csv'.format(filename))
    cleaned_text = pt.simple_preprocess()
    covid = pt.split_timeline(cleaned_text, '2/1/2020', '5/31/2020')

    return covid


def get_prediction_precovid(model, entities_train, train_file, predict_file):
    """using covid data for prediction, flip the training data to reverse it """
    pt = ProcessText('posts/{}_postids_posts.csv'.format(train_file))
    predict_data = get_precovid_text(predict_file)
    #train_data = get_covid_text(train_file)

    #train = LDAtrain(train_data, 0.3, 0.9, pt)
    #model, entities_train = train_model.lda_train()
    pre = LDApredict(entities_train, predict_data, pt)
    prediction = pre.lda_predict(model, predict_file, 'precovid')


def get_prediction_covid(model, entities_train, train_file, predict_file):
    """using covid data for prediction, flip the training data to reverse it """
    pt = ProcessText('posts/{}_postids_posts.csv'.format(train_file))
    predict_data = get_covid_text(predict_file)
    #train_data = get_covid_text(train_file)

    #train = LDAtrain(train_data, 0.3, 0.9, pt)
    #model, entities_train = train_model.lda_train()
    pre = LDApredict(entities_train, predict_data, pt)
    prediction = pre.lda_predict(model, predict_file, 'covid')


def get_topic_prevalence(lda_results, filename):
        """Compute the occurances of topics in the dataset and count the percentage """

        path_output = '/disk/data/share/s1690903/pandemic_anxiety/results/lda_results/'
        prevalence_dict = {}
        for i in range(1, 15):
            prevalence_dict["topic" + str(lda_results.columns[i])] = len(lda_results.loc[lda_results.iloc[:, i] > 0]) / len(lda_results)

        prevalence = pd.DataFrame.from_dict(prevalence_dict, orient='index')
        prevalence.to_csv(path_output + "predictlda_topic_prevalence/topic_prevalence_{}.csv".format(filename))

        return prevalence

def get_prevalence():
    path_prediction = '/disk/data/share/s1690903/pandemic_anxiety/results/lda_results/lda_prediction'
    path_output = '/disk/data/share/s1690903/pandemic_anxiety/results/lda_results/'
    allfiles = glob.glob(os.path.join(path_prediction, '*'))
    for file in allfiles:
        lda_results = pd.read_csv(file)
        #print(lda_results.shape)
        prevalence = get_topic_prevalence(lda_results, file.split('/')[9])


class LDAtrainPredict:
    def __init__(self, train_text, alpha, beta):
        self.evn_path = '/disk/data/share/s1690903/pandemic_anxiety/evn/'
        self.alpha = alpha
        self.beta = beta
        self.train_text = train_text
        #self.train_text = train_text

    def train_predict_loop(self):     
        evn = load_experiment(self.evn_path + 'experiment.yaml')
        subreddits = evn['subreddits']['subs']
        # get all the prediction result files
        train_data = get_covid_text(self.train_text)
        pt = ProcessText('posts/{}_postids_posts.csv'.format(self.train_text))
        train_model = LDAtrain(train_data, self.alpha, self.beta, pt)
        model, entities_train = train_model.lda_train()

        for sub in subreddits:
            get_prediction_precovid(model, entities_train, self.train_text, sub)
            get_prediction_covid(model, entities_train, self.train_text, sub)

ldatp = LDAtrainPredict('Anxiety', 0.3, 0.9)
ldatp.train_predict_loop()

get_prevalence()





