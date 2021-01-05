import os.path as path
import re
import os
import sys
import pandas as pd
# from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


from inverted_index.VanillaInvertedIndex import *


def generate_logformat_regex(logformat):
    headers = []
    splitters = re.split(r'(<[^<>]+>)', logformat)
    regex = ''
    for k in range(len(splitters)):
        if k % 2 == 0:
            splitter = re.sub(' +', '\\\s+', splitters[k])
            regex += splitter
        else:
            header = splitters[k].strip('<').strip('>')
            regex += '(?P<%s>.*?)' % header
            headers.append(header)
    regex = re.compile('^' + regex + '$')
    return headers, regex


def log_to_dataframe(log_file, regex, headers):
    log_messages = []
    linecount = 0

    for line in log_file:
        try:
            match = regex.search(line.strip())
            message = [match.group(header) for header in headers]
            log_messages.append(message)
            linecount += 1
        except Exception as e:
            pass
    logdf = pd.DataFrame(log_messages, columns=headers)
    logdf.insert(0, 'LineId', None)
    logdf['LineId'] = [i + 1 for i in range(linecount)]
    return logdf


def my_tokenizer(text):
    return text


def replace_alpha_nums(preprocessed_log):
    for i, token in enumerate(preprocessed_log):
        alpha_numeric_regex = r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$'
        is_alpha_numeric = re.search(alpha_numeric_regex, token)
        if is_alpha_numeric:
            preprocessed_log[i] = re.sub(alpha_numeric_regex, '<*>', token)

    return preprocessed_log


def check_numeric(token):
    return_token = ""
    for i in range(0, len(token)):
        if not token[i].isnumeric():
            return_token += token[i]
        else:
            return_token += '<*>'
    wildcard_check = re.compile('(?:\<\*\>)+')
    return re.sub(wildcard_check, '<*>', return_token)


def replace_nums(preprocessed_log):
    for i, token in enumerate(preprocessed_log):
        preprocessed_log[i] = check_numeric(token)
    return preprocessed_log


def replace_only_nums(preprocessed_log):
    for i, token in enumerate(preprocessed_log):
        if token.isnumeric():
            preprocessed_log[i] = '<*>'

    return preprocessed_log


def get_tfidf(doc_ids, temp):
    # print("tfidf check: ",doc_ids,temp)
    corpus = [temp[i] for i in doc_ids]
    filtered_corpus = list(map(lambda x: filter_wildcards(x), corpus))
    vectorizer = TfidfVectorizer(lowercase=False, analyzer='word', stop_words=None, tokenizer=my_tokenizer,
                                 token_pattern=None)

    vectors = vectorizer.fit_transform(filtered_corpus).toarray()
    vectors = [vectors[i].tolist() for i in range(len(corpus))]
    return cosine_similarity(vectors)


class Vue4Logs:
    def __init__(self, conf, logs):
        self.conf = conf
        self.threshold = 0.82  # conf['threshold']
        self.templates = {}
        self.inverted_index = VanillaInvertedIndex()
        self.results = []
        self.logs = logs
        self.headers = []

    def get_new_template(self, temp_template):
        if len(self.templates.keys()) == 0:
            next_id = 0
        else:
            next_id = max(self.templates.keys()) + 1
        # print("NEXT TEMPLATE ID :", next_id)
        self.templates[next_id] = temp_template
        self.results.append(next_id)
        return next_id

    def write_results(self, df_log, headers):

        df_log['Log_line'] = [str(i) for i in self.logs]
        df_log['EventId'] = ["E" + str(i) for i in self.results]
        headers.remove('Content')
        # print('headers',headers)
        # headers.remove('Content')
        try:
            df_log['headers'] = df_log[headers].apply(
                lambda x: ' '.join(x), axis=1)
        except (TypeError, KeyError):
            pass

        templates_df = []
        for j in self.results:
            if int(j) > 2000:
                print("Error in result")
                sys.exit(0)
            else:
                templates_df.append(" ".join(self.templates[j]))
        df_log['EventTemplate'] = templates_df

        # print('df_log',df_log)

        return df_log

    def preprocess(self, line):
        rgx = {
            'HDFS': {
                'regex': [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?']
            },

            'Hadoop': {
                'regex': [r'(\d+\.){3}\d+']
            },

            'Spark': {
                'regex': [r'(\d+\.){3}\d+', r'\b[KGTM]?B\b', r'([\w-]+\.){2,}[\w-]+']
            },

            'Zookeeper': {
                'regex': [r'(/|)(\d+\.){3}\d+(:\d+)?']
            },

            'BGL': {
                'regex': [r'core\.\d+']
            },

            'HPC': {
                'regex': [r'=\d+']
            },

            'Thunderbird': {
                'regex': [r'(\d+\.){3}\d+']
            },

            'Windows': {
                'regex': [r'0x.*?\s']
            },

            'Linux': {
                'regex': [r'(\d+\.){3}\d+', r'\d{2}:\d{2}:\d{2}']
            },

            'Android': {
                'regex': [r'(/[\w-]+)+', r'([\w-]+\.){2,}[\w-]+', r'\b(\-?\+?\d+)\b|\b0[Xx][a-fA-F\d]+\b|\b[a-fA-F\d]{4,}\b']
            },

            'HealthApp': {
                'regex': []
            },

            'Apache': {
                'regex': [r'(\d+\.){3}\d+']
            },

            'Proxifier': {
                'regex': [r'<\d+\ssec', r'([\w-]+\.)+[\w-]+(:\d+)?', r'\d{2}:\d{2}(:\d{2})*', r'[KGTM]B']
            },

            'OpenSSH': {
                'regex': [r'(\d+\.){3}\d+', r'([\w-]+\.){2,}[\w-]+']
            },

            'OpenStack': {
                'regex': [r'((\d+\.){3}\d+,?)+', r'/.+?\s', r'\d+']
            },

            'Mac': {
                'regex': [r'([\w-]+\.){2,}[\w-]+']
            }
        }
        regex = rgx[self.conf['log_file']]['regex']
        for currentRex in regex:
            line = re.sub(currentRex, '<*>', line)
        return line

    def parse(self):

        headers, regex = generate_logformat_regex(self.conf['log_format'])
        # print(headers)
        df_log = log_to_dataframe(self.logs, regex, headers)
        for idx, line in df_log.iterrows():
            log_id = line['LineId']
            pre_processed_log = self.preprocess(
                line['Content']).strip().split()
            # print(logID, pre_processed_log)

            pre_processed_log = replace_nums(pre_processed_log)

            hits = self.inverted_index.search_doc(pre_processed_log)

            if len(hits) == 0:
                new_id = self.get_new_template(pre_processed_log)
                self.inverted_index.index_doc(new_id, self.templates[new_id])

            else:
                candidates = {key: self.templates[key] for key in hits}
                length_filtered_candidates = {key: candidates[key] for key in candidates if
                                              len(candidates[key]) == len(pre_processed_log)}
                remaining_hits = list(length_filtered_candidates.keys())

                if len(length_filtered_candidates) == 0:
                    new_id = self.get_new_template(pre_processed_log)
                    self.inverted_index.index_doc(
                        new_id, self.templates[new_id])
                else:

                    greedily_found = False
                    for hit in remaining_hits:
                        if pre_processed_log == self.templates[hit]:
                            # print("greedy catch")
                            self.results.append(hit)
                            greedily_found = True

                    if greedily_found:
                        continue
                    # print("more rules")

                    max_similarity = 0
                    selected_candidate_id = None

                    similarity_candidates = {
                        key: self.templates[key] for key in length_filtered_candidates}
                    similarity_candidates[-1] = pre_processed_log
                    doc_ids = [-1] + list(length_filtered_candidates.keys())

                    similarity = get_tfidf(doc_ids, similarity_candidates)[0]
                    # print("similarity",similarity)
                    # similarity = self.get_bm25(doc_ids)

                    for i in range(len(similarity)):
                        if i == 0:
                            continue
                        else:
                            current_similarity = similarity[i]
                            if current_similarity > max_similarity:
                                max_similarity = current_similarity
                                selected_candidate_id = remaining_hits[i - 1]

                    if max_similarity < self.threshold:
                        new_id = self.get_new_template(pre_processed_log)
                        self.inverted_index.index_doc(
                            new_id, self.templates[new_id])
                    else:
                        selected_candidate = self.templates[selected_candidate_id]
                        template_length = len(selected_candidate)
                        # print("SELECTED TEMPLATE IS not EQUAL TO LOG LINE")
                        temporary_tokens = []
                        changed_tokens = []

                        for index in range(template_length):
                            # if log_line_token_list[position] == candidate_token_list[position]:
                            if pre_processed_log[index] == selected_candidate[index] or \
                                    "<*>" in selected_candidate[index]:
                                temporary_tokens.append(
                                    selected_candidate[index])
                            else:
                                changed_tokens.append(
                                    selected_candidate[index])
                                temporary_tokens.append("<*>")

                        updated_template = temporary_tokens
                        self.inverted_index.update_doc(
                            selected_candidate_id, self.templates[selected_candidate_id], updated_template)

                        self.templates[selected_candidate_id] = updated_template
                        self.results.append(selected_candidate_id)
                assert len(self.results) == log_id
        # print(self.dataset)
        output_df = self.write_results(df_log, headers)
        # ground_truth_df = 'ground_truth/' + self.dataset + '_2k.log_structured.csv'
        # output = self.output_path + "/" + self.dataset + "_structured.csv"

        # print(self.dataset, pa)
        # print(self.inverted_index.dict)
        # print(self.dataset, len(self.templates), "\n")
        return output_df
