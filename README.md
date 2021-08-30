## Question Answering NLU

Question Answering NLU (QANLU) is an approach that maps the NLU task into question answering, 
leveraging pre-trained question-answering models to perform well on few-shot settings. Instead of 
training an intent classifier or a slot tagger, for example, we can ask the model intent- and 
slot-related questions in natural language: 

```
Context : I'm looking for a cheap flight to Boston.

Question: Is the user looking to book a flight?
Answer  : Yes

Question: Is the user asking about departure time?
Answer  : No

Question: What price is the user looking for?
Answer  : cheap

Question: Where is the user flying from?
Answer  : (empty)
```

Thus, by asking questions for each intent and slot in natural language, we can effectively construct an NLU hypothesis. For more details,
please read the paper: 
[Language model is all you need: Natural language understanding as question answering](https://assets.amazon.science/33/ea/800419b24a09876601d8ab99bfb9/language-model-is-all-you-need-natural-language-understanding-as-question-answering.pdf).

This repository contains code to transform [MATIS++](https://github.com/amazon-research/multiatis) 
NLU data (e.g. utterances and intent / slot annotations) into [SQuAD 2.0 format](https://rajpurkar.github.io/SQuAD-explorer/explore/v2.0/dev/)
question-answering data that can be used by QANLU. MATIS++ includes
the original English version of ATIS and a translation into eight languages: German, Spanish, French, 
Japanese, Hindi, Portuguese, Turkish, and Chinese. 

To create a SQuAD-style dataset, we first need to create a list of questions for
each intent and a list of questions for each slot. Questions in English are saved in the `MATIS_questions.json` file.
In order to parse data in languages other than English, you need to provide questions in that language (or translate the English
questions we provide in this repository).

While we can have a number of questions for each intent and slot, sometimes QANLU will perform better if it sees
one question per intent and slot. We control this with the optional --single_q argument. If you call the
`atis.py` script using that argument, only the first question in the list will be chosen for each intent and slot.
In the opposite case, all questions for each intent and slot will be used.

Run the following to parse MATIS NLU data into SQuAD:

```
python atis.py \
       --data_path <path to the data> \
       --languages <de,en,es,fr,ja,hi,pt,tr,zh> \
       --qas_file <path to intent and slot questions json file> \
       --output_dir <path to where output files are stored> \
       [--single_q]
```

The output of this process will be in the exact format of SQuAD and can be used
to train question answering models. The next step would be to train a question answering model,
see [here](https://huggingface.co/transformers/master/custom_datasets.html#question-answering-with-squad-2-0)
for a guide. Alternatively, you can download a trained model directly from [huggingface](TBC).

In order to calculate precision, recall, and F1 for predictions done on QANLU test sets by the 
fine-tuned question answering model, you need to call:

```
python calculate_pr.py \
    --pred_file <full path to the predictions file created by transformers> \
    --test_file <full path to the test file that the predictions are for>
```

## Citation
If you use this work, please cite:

```
@inproceedings{namazifar2021language,
  title={Language model is all you need: Natural language understanding as question answering},
  author={Namazifar, Mahdi and Papangelis, Alexandros and Tur, Gokhan and Hakkani-T{\"u}r, Dilek},
  booktitle={ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={7803--7807},
  year={2021},
  organization={IEEE}
}
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the CC BY NC License.

