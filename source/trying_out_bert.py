import transformers
from datasets import load_dataset, load_metric
import random
import pandas as pd
from transformers import AutoTokenizer
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from collections import defaultdict
import csv
import gc
import os
import re






class Read_raw_data:

    def __init__(self, Anno_filename, model_checkpoint, label):
        self.path = '/disk/data/share/s1690903/pandemic_anxiety/data/anno_test/'
        self.file = pd.read_csv(self.path + Anno_filename)
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
        self.label = label

    # def preprocessing(self):
    #     cleaned_text = pt.simple_preprocess()


    def truncate_data(self):
        """Truncate and store datasets """
        file = self.file
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

        return self.tokenizer(examples['sentence'], truncation=True)


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
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
        self.batch_size = batch_size
        self.metric = load_metric('glue', 'mrpc')
        self.result_path = '/disk/data/share/s1690903/pandemic_anxiety/'

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

        #model = AutoModelForSequenceClassification.from_pretrained(self.model_checkpoint, num_labels=2)

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
        tokenizer=self.tokenizer,
        compute_metrics=self.compute_metrics
        )

        best_run = trainer.hyperparameter_search(n_trials=5, direction="maximize")

        #reproduce the best training
        for n, v in best_run.hyperparameters.items():
            setattr(trainer.args, n, v)

        trainer.train()

        evaluation = trainer.evaluate()

        return best_run, trainer, evaluation


    def write_result(self, best_run, evalution):
        """ write result to file"""

        file_exists = os.path.isfile(self.result_path + 'results/bert_test_result2.csv')
        f = open(self.result_path + 'results/bert_test_result2.csv', 'a')
        writer_top = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        if not file_exists:
            writer_top.writerow(['best_run']+['evaluation'])

            result_row = [[best_run + evaluation]]

            writer_top.writerows(result_row)

            f.close()

        else:
            f = open(self.result_path + 'results/bert_test_result2.csv', 'a')
            writer_top = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)

            result_row = [[best_run + evaluation]]

            writer_top.writerows(result_row)

            f.close()

            gc.collect()

model_checkpoint = "distilroberta-base"
batch_size = 8

read = Read_raw_data('all_data_text.csv', model_checkpoint=model_checkpoint, label='health_infected')

labels = ['financial_career', 'quar_social', 'health_infected','break_guideline', 'health_work', 'mental_health', 'death', 'travelling', 'future']

for l in labels:
    read = Read_raw_data('all_data_text.csv', model_checkpoint=model_checkpoint, label=l)
    encoded_dataset = read.encode_dataset()

    t = Training_classifier(model_checkpoint=model_checkpoint, encoded_dataset=encoded_dataset, batch_size=batch_size)

    best_run, trainer, evaluation = t.define_trainer()
    t.write_result(best_run, evaluation)





# read.truncate_data()
# dataset1 = read.create_dataset()
# read.preprocess_function(dataset1['train'][:5])
# encoded_dataset = dataset1.map(read.preprocess_function, batched=True)

# # #We can now finetune our model by just calling the train method:
#trainer.train()

#trainer.evaluate()

