import difflib as dl
import pandas as pd
from jiwer import wer
import nltk


def sentence_change_density(s_before, s_after):
    # for altered sentences, do diff in word-level in order to check change-density (via WER metric)
    error = wer(s_before, s_after)
    return error


def update_row(df, text_before, text_after, change_density):
    # insider utility function
    j = df[df['text_before'] == text_before].index
    df.loc[j, 'text_after'] = text_after
    df.loc[j, 'change_density'] = change_density


def create_diff_df(text_before, text_after, sent_tokenizer=nltk.tokenize.sent_tokenize):
    sentences_before = sent_tokenizer(text_before)
    sentences_after = sent_tokenizer(text_after)
    df = pd.DataFrame({'text_before': sentences_before,
                     'text_after': None,
                     'change_density': None})

    diff = list(dl.Differ().compare(sentences_before, sentences_after))
    i = len(diff)-1
    while i >= 0:
        line = diff[i]
        if line[0] == '?':  # line altered
            if diff[i-2][0] in ['+', '-']:
                update_row(df, diff[i-2][2:], diff[i-1][2:], sentence_change_density(diff[i-2][2:], diff[i-1][2:]))
                i = i-3
            elif diff[i-2][0] == '?':
                update_row(df, diff[i-3][2:], diff[i-1][2:], sentence_change_density(diff[i-3][2:], diff[i-1][2:]))
                i = i-4
        elif line[0] == '+':  # new line added
            if diff[i-1][0] == '?' and diff[i-2][0] == '-':  # line altered
                update_row(df, diff[i-2][2:], line[2:], sentence_change_density(diff[i-2][2:], line[2:]))
                i = i-3
            else:
                df.loc[df.shape[0]] = {'text_before': None, 'text_after': line[2:], 'change_density': +1.0}
                i = i-1
                continue

        elif line[0] == '-':  # line deleted
            update_row(df, line[2:], None, -1.0)
            i = i-1

        else:  # remained same
            update_row(df, line[2:], line[2:], 0.0)
            i = i-1

    return df


def diff_score(text_before, text_after, sent_tokenizer=nltk.tokenize.sent_tokenize):
    df = create_diff_df(text_before, text_after, sent_tokenizer)
    return df['change_density'].abs().mean()

