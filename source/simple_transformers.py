import pandas as pd
from simpletransformers.classification import ClassificationModel


#train_df = pd.read_csv('/afs/inf.ed.ac.uk/user/s16/s1690903/share/simple_transformer_test/ag_news_csv/train.csv', header=None)
#train_df = pd.read_csv('/disk/scratch_big/homedirs/s1690903/simple_transformer_test/ag_news_csv/train.csv', header=None) # slurm path
train_df = pd.read_csv('/disk/nfs/ostrom/s1690903/simple_transformer_test/ag_news_csv/train.csv', header=None) # sbatch ilcc path
train_df['text'] = train_df.iloc[:, 1] + " " + train_df.iloc[:, 2]
train_df = train_df.drop(train_df.columns[[1, 2]], axis=1)
train_df.columns = ['label', 'text']
train_df = train_df[['text', 'label']]
train_df['text'] = train_df['text'].apply(lambda x: x.replace('\\', ' '))
train_df['label'] = train_df['label'].apply(lambda x:x-1)

eval_df = pd.read_csv('/disk/nfs/ostrom/s1690903/simple_transformer_test/ag_news_csv/test.csv', header=None)
eval_df['text'] = eval_df.iloc[:, 1] + " " + eval_df.iloc[:, 2]
eval_df = eval_df.drop(eval_df.columns[[1, 2]], axis=1)
eval_df.columns = ['label', 'text']
eval_df = eval_df[['text', 'label']]
eval_df['text'] = eval_df['text'].apply(lambda x: x.replace('\\', ' '))
eval_df['label'] = eval_df['label'].apply(lambda x:x-1)

# Create a ClassificationModel
model = ClassificationModel('roberta', 'roberta-base', num_labels=4)

#model = ClassificationModel('xlnet', 'path_to_model/', num_labels=4)

args = {
    "output_dir": "outputs/",
    "cache_dir": "cache_dir/",

    "fp16": True,
    "fp16_opt_level": "O1",
    "max_seq_length": 128,
    "train_batch_size": 8,
    "gradient_accumulation_steps": 1,
    "eval_batch_size": 8,
    "num_train_epochs": 1,
    "weight_decay": 0,
    "learning_rate": 4e-5,
    "adam_epsilon": 1e-8,
    "warmup_ratio": 0.06,
    "warmup_steps": 0,
    "max_grad_norm": 1.0,

    "logging_steps": 50,
    "save_steps": 2000,

    "overwrite_output_dir": False,
    "reprocess_input_data": False,
    "evaluate_during_training": False,

    ##"process_count": cpu_count() - 2 if cpu_count() > 2 else 1,
    #"process_count": 5,
    "n_gpu": 1,
}

#model = TransformerModel('roberta', 'roberta-base', num_labels=4, args={'learning_rate':1e-5, 'num_train_epochs': 2, 'reprocess_input_data': True, 'overwrite_output_dir': True})


model.train_model(train_df)

result, model_outputs, wrong_predictions = model.eval_model(eval_df)




































