import transformers
from datasets import load_dataset, load_metric
import random
import pandas as pd
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from collections import defaultdict
import csv
import gc
import os
import re
from transformers import RobertaTokenizer, RobertaModel





class Read_raw_data:

    def __init__(self, Anno_filename, model_checkpoint, label):
        self.path = '/disk/data/share/s1690903/pandemic_anxiety/data/anno_test/'
        self.file = pd.read_csv(self.path + Anno_filename)
        #self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
        self.label = label
        self.model_checkpoint = model_checkpoint

    def tokenizer_model(self):
        """Tokenize text """
        tokens = AutoTokenizer.from_pretrained(self.model_checkpoint, use_fast=True)
        #tokens = RobertaTokenizer.from_pretrained('roberta-base')

        return tokens


    def truncate_data(self):
        """Truncate and store datasets """
        file = self.file
        #do we need to clean the data?
        #file.loc[:, 'cleaned_text'] = file['text'].apply(lambda x: str.lower(x))
        #file.loc[:, 'cleaned_text'] = file['cleaned_text'].apply(lambda x: " ".join(re.findall('[\w]+',x)))

        file = file[['post_id', 'text', self.label]]
        file.columns = ['idx', 'sentence', 'label']
        f = np.array_split(file, [int(.1*len(self.file)), int(.3*len(self.file))])
        f[2].to_csv(self.path + 'trainset.csv')
        f[1].to_csv(self.path + 'testset.csv')
        f[0].to_csv(self.path + 'validationset.csv')

    def create_dataset(self):
        """Create dataset using huggingface Dataset framework """

        dataset1 = load_dataset('csv', data_files={'train': [self.path + 'trainset.csv'], 'validation': [self.path + 'validationset.csv'], 'test': [self.path + 'testset.csv']})

        return dataset1

    def preprocess_function(self, examples):
        # tokenize sentences
        tokenizer = self.tokenizer_model()

        tokens = tokenizer(examples['sentence'], truncation=True)


        return tokens


    def encode_dataset(self):
        """encode dataset """

        self.truncate_data()
        dataset1 = self.create_dataset()
        encoded_dataset = dataset1.map(read.preprocess_function, batched=True)

        return encoded_dataset


class Training_classifier:

    def __init__(self, model_checkpoint, encoded_dataset, batch_size):
        self.model_checkpoint = model_checkpoint
        self.encoded_dataset = encoded_dataset
        #self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
        self.batch_size = batch_size
        self.metric = load_metric('glue', 'mrpc')
        self.result_path = '/disk/data/share/s1690903/pandemic_anxiety/'

    def tokenizer_model(self):
        """Tokenize text """
        #use Bert tokenizer
        tokens = AutoTokenizer.from_pretrained(self.model_checkpoint, use_fast=True)
        # use Roberta Tokenizer
        #tokens = RobertaTokenizer.from_pretrained('roberta-base')

        return tokens

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        #if task != "stsb":
        predictions = np.argmax(predictions, axis=1)
        # else:
        #     predictions = predictions[:, 0]
        return self.metric.compute(predictions=predictions, references=labels)


    def model_init(self):

        return AutoModelForSequenceClassification.from_pretrained(self.model_checkpoint, num_labels=2)


    def define_trainer(self):
        """Define a trainer and do parameter search """

        #run tokenizer
        tokenizer = self.tokenizer_model()

        metric_name = "accuracy"

        model_name = self.model_checkpoint.split("/")[-1]

        args = TrainingArguments(
            "{}-finetuned".format(model_name),
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=5,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model=metric_name,
            push_to_hub=True,
        )


        trainer = Trainer(
        model_init=self.model_init,
        args=args,
        # find some good hyperparameter on a portion of the training dataset 
        train_dataset=self.encoded_dataset["train"].shard(index=1, num_shards=10), 
        eval_dataset=self.encoded_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=self.compute_metrics
        )

        best_run = trainer.hyperparameter_search(n_trials=5, direction="maximize")

        #reproduce the best training
        for n, v in best_run.hyperparameters.items():
            setattr(trainer.args, n, v)

        trainer.train()

        evaluation = trainer.evaluate()


        return best_run, trainer, evaluation


    def write_result(self, best_run, evalution, label_name):
        """ write result to file"""

        file_exists = os.path.isfile(self.result_path + 'results/bert_test_result.csv')
        f = open(self.result_path + 'results/bert_test_result.csv', 'a')
        writer_top = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        if not file_exists:
            writer_top.writerow(['best_run']+['eval_loss']+['eval_accuracy']+['eval_f1']+['eval_runtime']+['epoch'] + ['label'] + ['batch_size'] + ['pretrained_model'])

            result_row = [[best_run, evaluation['eval_loss'], evaluation['eval_accuracy'], evaluation['eval_f1'],evaluation['eval_runtime'], evaluation['epoch'], label_name, self.batch_size], self.model_checkpoint]

            writer_top.writerows(result_row)
            f.close()

        else:
            f = open(self.result_path + 'results/bert_test_result.csv', 'a')
            writer_top = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)

            result_row = [[best_run, evaluation['eval_loss'], evaluation['eval_accuracy'], evaluation['eval_f1'],evaluation['eval_runtime'], evaluation['epoch'], label_name, self.batch_size, self.model_checkpoint]]

            writer_top.writerows(result_row)

            f.close()

            gc.collect()

model_checkpoint = "distilbert-base-uncased"
batch_size = 4
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# read = Read_raw_data('all_data_text.csv', model_checkpoint=model_checkpoint, label='health_infected')

# encoded_dataset = read.encode_dataset()

# t = Training_classifier(model_checkpoint=model_checkpoint, encoded_dataset=encoded_dataset, batch_size=batch_size)

# best_run, trainer, evaluation = t.define_trainer()
# t.write_result(best_run, evaluation)


#labels = ['health_infected', 'financial_career', 'quar_social', 'break_guideline', 'health_work', 'mental_health', 'death', 'travelling', 'future']

labels = ['quar_social', 'break_guideline', 'health_work', 'mental_health', 'death', 'travelling', 'future']

for l in labels:
    read = Read_raw_data('all_data_text.csv', model_checkpoint=model_checkpoint, label=l)
    encoded_dataset = read.encode_dataset()

    t = Training_classifier(model_checkpoint=model_checkpoint, encoded_dataset=encoded_dataset, batch_size=batch_size)

    best_run, trainer, evaluation = t.define_trainer()
    gc.collect()

    t.write_result(best_run, evaluation, read.label)






# dataset1 = read.create_dataset()
# read.preprocess_function(dataset1['train'][:5])
# encoded_dataset = dataset1.map(read.preprocess_function, batched=True)

# # #We can now finetune our model by just calling the train method:
#trainer.train()

#trainer.evaluate()

