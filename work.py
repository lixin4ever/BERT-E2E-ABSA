import argparse
import os
import torch
import numpy as np

from glue_utils import convert_examples_to_seq_features, compute_metrics_absa, ABSAProcessor
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer, XLNetConfig, XLNetTokenizer, WEIGHTS_NAME
from absa_layer import BertABSATagger
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
from seq_utils import ot2bieos_ts, bio2ot_ts, tag2ts

#ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig)), ())
ALL_MODELS = (
     'bert-base-uncased',
 'bert-large-uncased',
 'bert-base-cased',
 'bert-large-cased',
 'bert-base-multilingual-uncased',
 'bert-base-multilingual-cased',
 'bert-base-chinese',
 'bert-base-german-cased',
 'bert-large-uncased-whole-word-masking',
 'bert-large-cased-whole-word-masking',
 'bert-large-uncased-whole-word-masking-finetuned-squad',
 'bert-large-cased-whole-word-masking-finetuned-squad',
 'bert-base-cased-finetuned-mrpc',
 'bert-base-german-dbmdz-cased',
 'bert-base-german-dbmdz-uncased',
 'xlnet-base-cased',
 'xlnet-large-cased'
)


MODEL_CLASSES = {
    'bert': (BertConfig, BertABSATagger, BertTokenizer),
}


def load_and_cache_examples(args, task, tokenizer):
    # similar to that in main.py
    processor = ABSAProcessor()
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        'test',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file):
        print("cached_features_file:", cached_features_file)
        features = torch.load(cached_features_file)
        examples = processor.get_test_examples(args.data_dir, args.tagging_schema)
    else:
        #logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels(args.tagging_schema)
        examples = processor.get_test_examples(args.data_dir, args.tagging_schema)
        features = convert_examples_to_seq_features(examples=examples, label_list=label_list, tokenizer=tokenizer,
                                                    cls_token_at_end=bool(args.model_type in ['xlnet']),
                                                    cls_token=tokenizer.cls_token,
                                                    sep_token=tokenizer.sep_token,
                                                    cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
                                                    pad_on_left=bool(args.model_type in ['xlnet']),
                                                    pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0)
        torch.save(features, cached_features_file)
    total_words = []
    for input_example in examples:
        text = input_example.text_a
        total_words.append(text.split(' '))

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    # used in evaluation
    all_evaluate_label_ids = [f.evaluate_label_ids for f in features]
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset, all_evaluate_label_ids, total_words


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--absa_home", type=str, required=True, help="Home directory of the trained ABSA model")
    parser.add_argument("--ckpt", type=str, required=True, help="Directory of model checkpoint for evaluation")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="The incoming data dir. Should contain the files of test/unseen data")
    parser.add_argument("--task_name", type=str, required=True, help="task name")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                        "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument('--tagging_schema', type=str, default='BIEOS', help="Tagging schema, should be kept same with "
                                                                            "that of ckpt")

    args = parser.parse_args()

    return args


def main():
    # perform evaluation on single GPU
    args = init_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    if torch.cuda.is_available():
        args.n_gpu = torch.cuda.device_count()

    args.model_type = args.model_type.lower()
    _, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    # load the trained model (including the fine-tuned GPT/BERT/XLNET)
    print("Load checkpoint %s/%s..." % (args.ckpt, WEIGHTS_NAME))
    model = model_class.from_pretrained(args.ckpt)
    # follow the property of tokenizer in the loaded model, e.g., do_lower_case=True
    tokenizer = tokenizer_class.from_pretrained(args.absa_home)
    model.to(args.device)
    model.eval()
    predict(args, model, tokenizer)


def predict(args, model, tokenizer):
    dataset, evaluate_label_ids, total_words = load_and_cache_examples(args, args.task_name, tokenizer)
    sampler = SequentialSampler(dataset)
    # process the incoming data one by one
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=1)
    print("***** Running prediction *****")

    total_preds, gold_labels = None, None
    idx = 0
    if args.tagging_schema == 'BIEOS':
        absa_label_vocab = {'O': 0, 'EQ': 1, 'B-POS': 2, 'I-POS': 3, 'E-POS': 4, 'S-POS': 5,
                        'B-NEG': 6, 'I-NEG': 7, 'E-NEG': 8, 'S-NEG': 9,
                        'B-NEU': 10, 'I-NEU': 11, 'E-NEU': 12, 'S-NEU': 13}
    elif args.tagging_schema == 'BIO':
        absa_label_vocab = {'O': 0, 'EQ': 1, 'B-POS': 2, 'I-POS': 3,
        'B-NEG': 4, 'I-NEG': 5, 'B-NEU': 6, 'I-NEU': 7}
    elif args.tagging_schema == 'OT':
        absa_label_vocab = {'O': 0, 'EQ': 1, 'T-POS': 2, 'T-NEG': 3, 'T-NEU': 4}
    else:
        raise Exception("Invalid tagging schema %s..." % args.tagging_schema)
    absa_id2tag = {}
    for k in absa_label_vocab:
        v = absa_label_vocab[k]
        absa_id2tag[v] = k

    for batch in tqdm(dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                      # XLM don't use segment_ids
                      'labels': batch[3]}
            outputs = model(**inputs)
            # logits: (1, seq_len, label_size)
            logits = outputs[1]
            # preds: (1, seq_len)
            if model.tagger_config.absa_type != 'crf':
                preds = np.argmax(logits.detach().cpu().numpy(), axis=-1)
            else:
                mask = batch[1]
                preds = model.tagger.viterbi_tags(logits=logits, mask=mask)
            label_indices = evaluate_label_ids[idx]
            words = total_words[idx]
            pred_labels = preds[0][label_indices]
            assert len(words) == len(pred_labels)
            pred_tags = [absa_id2tag[label] for label in pred_labels]

            if args.tagging_schema == 'OT':
                pred_tags = ot2bieos_ts(pred_tags)
            elif args.tagging_schema == 'BIO':
                pred_tags = ot2bieos_ts(bio2ot_ts(pred_tags))
            else:
                # current tagging schema is BIEOS, do nothing
                pass
            p_ts_sequence = tag2ts(ts_tag_sequence=pred_tags)
            output_ts = []
            for t in p_ts_sequence:
                beg, end, sentiment = t
                aspect = words[beg:end+1]
                output_ts.append('%s: %s' % (aspect, sentiment))
            print("Input: %s, output: %s" % (' '.join(words), '\t'.join(output_ts)))
            # for evaluation
            if total_preds is None:
                total_preds = preds
            else:
                total_preds = np.append(total_preds, preds, axis=0)
            if inputs['labels'] is not None:
                # for the unseen data, there is no ``labels''
                if gold_labels is None:
                    gold_labels = inputs['labels'].detach().cpu().numpy()
                else:
                    gold_labels = np.append(gold_labels, inputs['labels'].detach().cpu().numpy(), axis=0)
        idx += 1
    if gold_labels is not None:
        result = compute_metrics_absa(preds=total_preds, labels=gold_labels, all_evaluate_label_ids=evaluate_label_ids,
                                      tagging_schema=args.tagging_schema)
        for (k, v) in result.items():
            print("%s: %s" % (k, v))


if __name__ == "__main__":
    main()

