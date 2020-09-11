import json


def read_dataset(path: str):

    with open(path) as f:
        dataset = json.load(f)
    
    sentences, labels = {}, {}
    for sentence_id, sentence in dataset.items():
        sentence_id = int(sentence_id)
        sentences[sentence_id] = {
            'words': sentence['words'],
            'lemmas': sentence['lemmas'],
            'pos_tags': sentence['pos_tags'],
            'dependency_heads': [int(head) for head in sentence['dependency_heads']],
            'dependency_relations': sentence['dependency_relations'],
            'predicates': sentence['predicates'],
        }

        labels[sentence_id] = {
            'predicates': sentence['predicates'],
            'roles': {int(p): r for p, r in sentence['roles'].items()}
        }

    return sentences, labels


def evaluate_predicate_identification(labels, predictions, null_tag='_'):
    true_positives, false_positives, false_negatives = 0, 0, 0
    for sentence_id in labels:
        gold_predicates = labels[sentence_id]['predicates']
        pred_predicates = predictions[sentence_id]['predicates']
        for g, p in zip(gold_predicates, pred_predicates):
            if g != null_tag and p != null_tag:
                true_positives += 1
            elif p != null_tag and g == null_tag:
                false_positives += 1
            elif g != null_tag and p == null_tag:
                false_negatives += 1
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


def evaluate_predicate_disambiguation(labels, predictions, null_tag='_'):
    true_positives, false_positives, false_negatives = 0, 0, 0
    for sentence_id in labels:
        gold_predicates = labels[sentence_id]['predicates']
        pred_predicates = predictions[sentence_id]['predicates']
        for g, p in zip(gold_predicates, pred_predicates):
            if g != null_tag and p != null_tag:
                if p == g:
                    true_positives += 1
                else:
                    false_positives += 1
                    false_negatives += 1
            elif p != null_tag and g == null_tag:
                false_positives += 1
            elif g != null_tag and p == null_tag:
                false_negatives += 1
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
            else: # idx in both gold and pred
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
            else: # idx in both gold and pred
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


def _get_table_line(a, b, c):
    if isinstance(b, float):
        b = '{:0.2f}'.format(b)
    if isinstance(c, float):
        c = '{:0.2f}'.format(c)

    line = '{:^20}|{:^20}|{:^20}'.format(a, b, c)
    return line


def print_table(title, results):
    header = _get_table_line('', 'Gold Positive', 'Gold Negative')
    header_sep = '=' * len(header)

    first_line = _get_table_line('Pred Positive', results['true_positives'], results['false_positives'])
    second_line = _get_table_line('Pred Negative', results['false_negatives'], '')

    precision = 'Precision = {:0.4f}'.format(results['precision'])
    recall = 'Recall    = {:0.4f}'.format(results['recall'])
    f1 = 'F1 score  = {:0.4f}'.format(results['f1'])

    output = '{}\n\n{}\n{}\n{}\n{}\n\n\n{}\n{}\n{}\n\n\n'.format(title.upper(), header, header_sep, first_line, second_line, precision, recall, f1)
    return output
