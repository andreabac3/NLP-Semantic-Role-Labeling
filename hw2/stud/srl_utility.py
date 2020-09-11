# !/usr/bin/env python3

from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchtext.vocab import Vocab
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

'''
SOURCE CODE: 
the following functions: evaluate_argument_classification and evaluate_argument_identification
are taken from the utils.py provided by the task organizer in order to debug and test the model in the same way of the docker
'''


def evaluate_argument_classification(labels, predictions, null_tag='_'):
    true_positives, false_positives, false_negatives = 0, 0, 0
    for sentence_id in labels:
        gold = labels[sentence_id]['roles']
        pred = predictions[sentence_id]['roles']
        predicate_indices = set(gold.keys()).union(pred.keys())

        for idx in predicate_indices:
            if idx in gold and idx not in pred:
                false_negatives += sum(1 for role in gold[idx] if role != null_tag)
            elif idx in pred and idx not in gold:
                false_positives += sum(1 for role in pred[idx] if role != null_tag)
            else:  # idx in both gold and pred
                for r_g, r_p in zip(gold[idx], pred[idx]):
                    if r_g != null_tag and r_p != null_tag:
                        if r_g == r_p:
                            true_positives += 1
                        else:
                            false_positives += 1
                            false_negatives += 1
                    elif r_g != null_tag and r_p == null_tag:
                        false_negatives += 1
                    elif r_g == null_tag and r_p != null_tag:
                        false_positives += 1

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * (precision * recall) / (precision + recall)
    return {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def plot_conf_matrix(cm, name_label, dataset_name: str):
    # standard function to plot confusion matrix
    cm_df = pd.DataFrame(cm, index=name_label, columns=name_label)
    plt.figure(figsize=(30, 30))
    sns.heatmap(cm_df, annot=True)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Normalized Confusion Matrix ' + dataset_name)
    plt.show()


def calculate_confusion_matrix(labels, predictions, dataset_name: str = '', include_placeholder: bool = True):
    # function used to plot confusion matrix
    from sklearn.metrics import confusion_matrix
    pred_set = []
    label_set = []
    for sentence_id in labels:
        gold = labels[sentence_id]['roles']
        pred = predictions[sentence_id]['roles']
        pred_index_gold = gold.keys()
        for pred_id in pred_index_gold:
            for i in range(len(pred[pred_id])):
                if (pred[pred_id][i] == '_' or gold[pred_id][i] == '_') and include_placeholder:
                    # use to avoid the remove the placeholder tag from the confusion matrix
                    continue
                pred_set.append(pred[pred_id][i])
                label_set.append(gold[pred_id][i])
    name_label = list(set(label_set).union(set(pred_set)))
    cm = confusion_matrix(label_set, pred_set, name_label, normalize='true')
    plot_conf_matrix(cm, name_label, dataset_name)
    return cm, name_label


def evaluate_argument_identification(labels, predictions, null_tag='_'):
    true_positives, false_positives, false_negatives = 0, 0, 0
    for sentence_id in labels:
        gold = labels[sentence_id]['roles']
        pred = predictions[sentence_id]['roles']
        predicate_indices = set(gold.keys()).union(pred.keys())
        for idx in predicate_indices:
            if idx in gold and idx not in pred:
                false_negatives += sum(1 for role in gold[idx] if role != null_tag)
            elif idx in pred and idx not in gold:
                false_positives += sum(1 for role in pred[idx] if role != null_tag)
            else:  # idx in both gold and pred
                for r_g, r_p in zip(gold[idx], pred[idx]):
                    if r_g != null_tag and r_p != null_tag:
                        true_positives += 1
                    elif r_g != null_tag and r_p == null_tag:
                        false_negatives += 1
                    elif r_g == null_tag and r_p != null_tag:
                        false_positives += 1

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * (precision * recall) / (precision + recall)
    return {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def print_output_docker(model: nn.Module, dataloader_test: DataLoader, vocab_labels: Vocab) -> dict:
    '''
    This function take a model and return prediction as request in the docker.
    This function support batch evaluation
    '''
    output_dict = dict()
    model.eval()
    with torch.no_grad():
        for sample in tqdm(dataloader_test):
            batch_size: int = len(sample['predicate_id'])
            predictions = model(sample)  # get batched predictions

            if model.use_crf:
                predictions = torch.LongTensor(model.crf.decode(predictions))
            else:
                predictions = torch.argmax(predictions, -1)

            for batch_num in range(batch_size):
                # build the output in the right format
                batch_predictions = predictions[batch_num].view(-1)

                mask_padding = sample['words'][batch_num] != 0  # removing padding
                encoded_predictions = batch_predictions[mask_padding]
                list_predictions = encoded_predictions.tolist()

                decode_predictions: List[str] = [vocab_labels.itos[elem] for elem in list_predictions]  # back from vocabulary id List[int] to labels List[str]
                id_sentence: int = int(sample['sentence_id'][batch_num])
                index_pred: int = int(sample['predicate_id'][batch_num])
                if id_sentence in output_dict:
                    # add more than one list of roles at given id sentence
                    output_dict[id_sentence]['roles'][index_pred] = decode_predictions
                else:
                    # insert for the first time
                    output_dict[id_sentence] = {'roles': {index_pred: decode_predictions}}

        for id_sentence in dataloader_test.dataset.empty_predicates:
            # Case in which we don't have any index_pred
            output_dict[id_sentence] = {'roles': dict()}
        return output_dict


def plot_loss(data: dict) -> None:
    epochs_list = range(1, len(data['loss_train'])+1)
    loss_summary_train = data['loss_train']
    loss_summary_valid = data['loss_valid']
    f1_validation_summary_valid = data['f1_valid_identification']
    f1_classification_summary_valid = data['f1_valid_classification']

    plt.plot(epochs_list, loss_summary_train, label="Train")
    plt.plot(epochs_list, loss_summary_valid, label="Dev")
    best_loss_index = loss_summary_valid.index(min(loss_summary_valid))
    best_loss_value = loss_summary_valid[best_loss_index]
    f1_at_best_loss = f1_classification_summary_valid[best_loss_index]

    plt.axvline(x=best_loss_index + 1, label='Dev loss early stopping \n F1: {} \n Loss: {}'.format(round(f1_at_best_loss, 4), round(best_loss_value, 4)), c='red', linestyle=':')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Argument Classification Train vs Dev')
    plt.legend()
    plt.show()

    plt.plot(epochs_list, f1_validation_summary_valid, label="dev identification")
    plt.plot(epochs_list, f1_classification_summary_valid, label="dev classification")
    best_f1_index = f1_classification_summary_valid.index(max(f1_classification_summary_valid))
    best_f1_value = f1_classification_summary_valid[best_f1_index]
    loss_f1_best = loss_summary_valid[best_f1_index]
    plt.axvline(x=best_f1_index + 1, label='Classification f1 early stopping \n F1: {} \n Loss: {}'.format(round(best_f1_value, 4), round(loss_f1_best, 4)), c='red', linestyle=':')

    plt.xlabel('Epochs')
    plt.ylabel('F1')
    plt.title('F1 Score Identification vs Classification')
    plt.legend()
    plt.show()
