from lda_topic import *



def run_LDA(path, text: typing.Dict[str, str], num_topic:int, alpha, beta):
        """Select the best lda model with extracted text 
        text: entities dictionary
        domTname:file name for the output
        """

        # convert data to dictionary format

        file_exists = os.path.isfile(path + 'lda_result_apply.csv')
        f = open(path + 'lda_result_apply.csv', 'a', encoding='utf-8-sig')
        writer_top = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        if not file_exists:
            writer_top.writerow(['a'] + ['b'] + ['coherence'] + ['time'] + ['topics'] + ['num_topics'] )
       
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


def get_topic_covid_timeline(cleaned_text, year: int, num_topic: int, alpha_pre, beta_pre, alpha_co, beta_co) -> pd.DataFrame:
    """get topics according to season timeline 
    result saved in results/lda_results/
    """
    # pt = ProcessText('posts/Anxiety_postids_posts.csv')
    # cleaned_text = pt.simple_preprocess()

    # here we can set the seasons
    pt = ProcessText('posts/Anxiety_postids_posts.csv')

    precovid = pt.split_timeline(cleaned_text, '1/1/{}'.format(year), '12/31/{}'.format(year))
    precovid2 = pt.split_timeline(cleaned_text, '1/1/{}'.format(year - 1), '12/31/{}'.format(year-1))
    covid = pt.split_timeline(cleaned_text, '2/1/2020', '5/31/2020')
    precovid.update(precovid2)
 
   #run lda for each period
    
    entities = pt.extract_entities(covid)
    model_covid = run_LDA(pt.path_result, entities, num_topic, alpha_pre, beta_pre)

    # if len(covid) > 1:
    #     entities =rix pt.extract_entities(covid)
    #     model_covid = run_LDA(pt.path_result, entities, num_topic, alpha_co, beta_co)
  
    return model_covid




def lda_prediction(train_text, pre_text):

    # extract entities 
    entities_train = pt.extract_entities(train_text)
    entities_pre = pt.extract_entities(pre_text)

    #run model on train set
    model = run_LDA(pt.path_result, entities_train, 15, 0.3, 0.9)

    #get the common corpus
    dictionary = gensim.corpora.Dictionary(entities_train.values())
    bow_corpus_common = [dictionary.doc2bow(doc) for doc in entities_pre.values()]

    #unseen_doc = bow_corpus_common
    #vector = model_precovid[0][unseen_doc]

    #Update the model by incrementally training on the new corpus
    #model_covid.update(bow_corpus_common)
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
    all_lda_score_dfT.to_csv(pt.path_result + 'test_lda_predict.csv')

    # get topic dominance
    #df_topic_sents_keywords = lda.format_topics_sentences(model, entities_pre)
    #df_dominant_topic = df_topic_sents_keywords.reset_index()

    #sent_topics_df = pd.concat([df_dominant_topic, scores_best], axis=1)

    return all_lda_score_dfT


pt = ProcessText('posts/HealthAnxiety_postids_posts.csv')
cleaned_text = pt.simple_preprocess()

precovid = pt.split_timeline(cleaned_text, '1/1/2019', '12/31/2019')
#precovid2 = pt.split_timeline(cleaned_text, '1/1/2018', '12/31/2018')
covid = pt.split_timeline(cleaned_text, '2/1/2020', '5/31/2020')
#precovid.update(precovid2)

result = lda_prediction(covid, precovid)



#model = get_topic_covid_timeline(cleaned_text, 2019, 15, 0.1, 0.9, 0.3, 0.9)


#topics = model.transform(out_of_sample_docs)













