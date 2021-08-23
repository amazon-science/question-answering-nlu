# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

import argparse
import copy
import json


def main(pred_file, test_file, cls):
    squad_nlu_test = json.load(open(test_file))
    test_pred = json.load(open(pred_file))
    scores_dict = {}
    d = {'nb_correct': 0, 'nb_pred': 0, 'nb_true': 0}
    intent_scores_dict = {}
    intent_dico = {'true_pos': 0, 'false_pos': 0, 'true_neg': 0, 'false_neg': 0, 'unrel': 0}

    for sentence in squad_nlu_test['data'][0]['paragraphs']:
        for qa in sentence['qas']:
            if not qa['id'].startswith("intent_"):
                if qa['slot'] not in scores_dict.keys():
                    scores_dict[qa['slot']] = copy.deepcopy(d)
                if not qa['is_impossible']:
                    if qa['answers'][0]['text'] == test_pred[qa['id']]:
                        scores_dict[qa['slot']]['nb_correct'] += 1
                        scores_dict[qa['slot']]['nb_pred'] += 1
                        scores_dict[qa['slot']]['nb_true'] += 1
                    else:
                        if test_pred[qa['id']]:
                            scores_dict[qa['slot']]['nb_pred'] += 1
                        scores_dict[qa['slot']]['nb_true'] += 1
                else:
                    if test_pred[qa['id']]:
                        scores_dict[qa['slot']]['nb_pred'] += 1
            else:
                if qa['intent'] not in intent_scores_dict.keys():
                    intent_scores_dict[qa['intent']] = copy.deepcopy(intent_dico)
                if qa['answers'][0]['text'] == test_pred[qa['id']]:
                    if qa['answers'][0]['text'] == "yes":
                        intent_scores_dict[qa['intent']]['true_pos'] += 1
                    elif qa['answers'][0]['text'] == "no":
                        intent_scores_dict[qa['intent']]['true_neg'] += 1
                    else:
                        raise ValueError("Wrong value of intent yes or no answer!")
                else:
                    if qa['answers'][0]['text'] == "yes":
                        if test_pred[qa['id']] == "no":
                            intent_scores_dict[qa['intent']]['false_neg'] += 1
                        else:
                            intent_scores_dict[qa['intent']]['unrel'] += 1
                    else:
                        assert qa['answers'][0]['text'] == "no"
                        if test_pred[qa['id']] == "yes":
                            intent_scores_dict[qa['intent']]['false_pos'] += 1
                        else:
                            intent_scores_dict[qa['intent']]['unrel'] += 1

    nb_correct = 0
    nb_pred = 0
    nb_true = 0

    for i in scores_dict.keys():
        nb_correct += scores_dict[i]['nb_correct']
        nb_pred += scores_dict[i]['nb_pred']
        nb_true += scores_dict[i]['nb_true']

    p = nb_correct / nb_pred if nb_pred > 0 else 0
    r = nb_correct / nb_true if nb_true > 0 else 0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0

    print("Precision: ", p)
    print("Recall: ", r)
    print("F1: ", f1)

    if cls:
        # Intent
        true_pos = 0
        false_pos = 0
        true_neg = 0
        false_neg = 0
        # Unrelated answer; any thing other than yes or no
        unrel = 0

        for intent in intent_scores_dict.keys():
            true_pos += intent_scores_dict[intent]['true_pos']
            false_pos += intent_scores_dict[intent]['false_pos']
            true_neg += intent_scores_dict[intent]['true_neg']
            false_neg += intent_scores_dict[intent]['false_neg']
            unrel += intent_scores_dict[intent]['unrel']

        cls_p = true_pos / (true_pos + false_pos + unrel) if (true_pos + false_pos + unrel) > 0 else 0
        cls_r = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        cls_f1 = 2 * cls_p * cls_r / (cls_p + cls_r) if (cls_p + cls_r) > 0 else 0

        intent_result = {'cls_p': cls_p, 'cls_r': cls_r, 'cls_f1': cls_f1}
        print("true_pos: ", true_pos, "false_pos: ", false_pos , "true_neg: ", true_neg, "false_neg: ", false_neg, "unrel: ", unrel)

        results = {"slot": {"Precision": p, "Recall": r, "F1": f1}, "intent": intent_result}
    else:
        results = {"slot": {"Precision": p, "Recall": r, "F1": f1}}
    print("Results: ", results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred_file",
        default=None,
        type=str,
        required=True,
        help="Full path to the file that contains predictions. Written by the HF's QA model")
    parser.add_argument(
        "--test_file",
        default=None,
        type=str,
        required=True,
        help="Full path to the QA test file")
    parser.add_argument(
        "--cls",
        default=False,
        type=bool,
        help="Whether there is classification too?"
    )
    args = parser.parse_args()

    main(args.pred_file, args.test_file, args.cls)
