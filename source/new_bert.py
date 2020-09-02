
from sklearn.datasets import fetch_20newsgroups
import ktrain
#from ktrain import text
from ktrain import tabular
import numpy as np
import pandas as pd
import glob

# example from here https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/develop/tutorials/tutorial-08-tabular_classification_and_regression.ipynb

# categories = ['alt.atheism', 'soc.religion.christian',
#           'comp.graphics', 'sci.med']



# train_b = fetch_20newsgroups(subset='train',
#      categories=categories, shuffle=True, random_state=42)

# test_b = fetch_20newsgroups(subset='test',
#      categories=categories, shuffle=True, random_state=42)

class Read_raw_data:
    def __init__(self):
        self.path = '/disk/data/share/s1690903/pandemic_anxiety/data/annotations/post_anno/'

    def read_all_files(self) -> pd.DataFrame:
        """ Read all the annotation files. """

        all_files = []
        for file in glob.glob(self.path + "*.csv"):
            file_pd = pd.read_csv(file)
            all_files.append(file_pd)

        all_files_pd = pd.concat(all_files)

        # Drop those without annotations.
        all_files_pd = all_files_pd[all_files_pd['anxiety'].notna()]

        # Replace Nan with 0.
        all_files_pd = all_files_pd.replace(np.nan, 0)
        liwc_file = all_files_pd[['title', 'text', 'post_id']]


        #all_files_pd['text'] = all_files_pd['text'].apply(lambda x: x.str.slice(0, 500))
        all_files_pd['text'] = all_files_pd['text'].str.split(' ').str.slice(0,50)
        all_files_pd['text'] = all_files_pd['text'].apply(lambda x: ' '.join(str(v) for v in x))
        liwc_file.to_csv('/disk/data/share/s1690903/pandemic_anxiety/data/annotations/test.csv')

        return all_files_pd

# all_files['text'] = all_files['text'].apply(lambda x: x.str.split('').slice(0, 450))

    def combine_columns(self, newcol, col1, col2, col3=None):
        """Combine column labels """
        all_files = self.read_all_files()
        if col3 == None:
            all_files[newcol] = all_files[col1] + all_files[col2]
            all_files.loc[all_files[newcol] > 1, newcol] = 1
        else:
            all_files[newcol] = all_files[col1] + all_files[col2] + all_files[col3]
            all_files.loc[all_files[newcol] > 1, newcol] = 1

        return all_files

class Preprocess:
    def __init__(self, raw_data, labelcol):
        '''define the main path'''

        self.path = '/disk/data/share/s1690903/pandemic_anxiety/data/annotations/'

        self.file = raw_data
        # join the title and text 
        self.file['text'] = self.file['text'].str.cat(self.file['title'], sep=" ")
        self.file['text'] = self.file['text'].replace('\n', ' ', regex=True)
        self.file['text'] = self.file['text']  + ' ***lucia*** ' + '\n'
        
        #self.file['text'] = self.file['text'].replace(' ***lucia*** ','\n', regex=True)

        self.labelcol = labelcol

    def save_txt(self):
        '''separate text and labels '''
        
        np.savetxt(self.path + 'text.txt', self.file['text'], fmt='%s')

        return list(self.file['text']), self.file[self.labelcol]




read = Read_raw_data()
all_files = read.read_all_files()



# define train test
np.random.seed(42)
p = 0.3 # 10% for test set
prop = 1-p
#df = train_df.copy()
msk = np.random.rand(len(all_files)) < prop
train_df = all_files[msk]
test_df = all_files[~msk]


trn, val, preproc = tabular.tabular_from_df(train_df, label_columns=['mental_health'], random_state=42)

model = tabular.tabular_classifier('mlp', trn)

learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=32)

learner.lr_find(show_plot=True, max_epochs=2)

#train model
learner.fit_onecycle(2e-5, 5)

learner.evaluate(val, class_names=preproc.get_classes())

#learner.validate(val_data=(x_test, y_test), class_names=train_b.target_names)

predictor = ktrain.get_predictor(learner.model, preproc)

preds = predictor.predict(test_df, return_proba=True)

# print('test accuracy:')
# (np.argmax(preds, axis=1) == test_df['anxiety'].values).sum()/test_df.shape[0]






