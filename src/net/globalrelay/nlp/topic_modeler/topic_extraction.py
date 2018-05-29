'''
Created on Mar 9, 2018

@author: wilson.penha
'''
import pyLDAvis.sklearn
import os
import base64
import codecs
import numpy as np
import pandas as pd
import warnings
import re
import guidedlda
from gensim import interfaces, utils, matutils
import logging

from collections import defaultdict
from itertools import groupby
from sklearn import datasets
from numpy import random
from scipy.stats import dirichlet, norm, poisson

from docutils.nodes import inline

from collections import Counter
from scipy import int64
from scipy.misc import imread
from scipy.sparse import (csr_matrix, lil_matrix, coo_matrix)
from gensim.models import LdaModel, Phrases
from gensim.models.phrases import Phraser
from gensim.corpora import Dictionary
from gensim import models, corpora, similarities

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.externals import joblib
from sklearn.externals.joblib import Parallel, delayed
from sklearn.utils import (check_random_state, check_array,
                     gen_batches, gen_even_slices, _get_n_jobs)
 
import configparser

import json 
import nltk
from cgitb import text

import itertools

from net.globalrelay.nlp.topic_modeler.my_display import MyDisplay
from net.globalrelay.nlp.topic_modeler.labels.text import LabelCountVectorizer
from net.globalrelay.nlp.topic_modeler.labels.label_finder import BigramLabelFinder
from net.globalrelay.nlp.topic_modeler.labels.label_ranker import LabelRanker
from net.globalrelay.nlp.topic_modeler.labels.pmi import PMICalculator
from net.globalrelay.nlp.topic_modeler.labels.corpus_processor import (
                                        CorpusWordLengthFilter,
                                        CorpusPOSTagger,
                                        CorpusStemmer)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

from nltk.stem import WordNetLemmatizer

from net.globalrelay.nlp.topic_modeler.lemma_count_vectorizer import LemmaCountVectorizer

CURDIR = os.path.dirname(os.path.realpath(__file__))

logger = logging.getLogger('guidedlda')
logger.propagate = False


class TunedLatentDirichletAllocation(LatentDirichletAllocation):
    # adding temp_folder to be able to setup parallel to dumpt to different tmp folder instead of /dev/shm
    def __init__(self, n_components=10, doc_topic_prior=None,
                 topic_word_prior=None, learning_method=None,
                 learning_decay=.7, learning_offset=10., max_iter=10,
                 batch_size=128, evaluate_every=-1, total_samples=1e6,
                 perp_tol=1e-1, mean_change_tol=1e-3, max_doc_update_iter=100,
                 n_jobs=1, verbose=0, random_state=None, n_topics=None, temp_folder=None):
        
        self.temp_folder=temp_folder
        
        super(TunedLatentDirichletAllocation, self).__init__(n_components=n_components, doc_topic_prior=doc_topic_prior,
                 topic_word_prior=topic_word_prior, learning_method=learning_method,
                 learning_decay=learning_decay, learning_offset=learning_offset, max_iter=max_iter,
                 batch_size=batch_size, evaluate_every=evaluate_every, total_samples=total_samples,
                 perp_tol=perp_tol, mean_change_tol=mean_change_tol, max_doc_update_iter=max_doc_update_iter,
                 n_jobs=n_jobs, verbose=verbose, random_state=random_state, n_topics=n_topics)
        
    def fit(self, X, y=None):
        """Learn model for the data X with variational Bayes method.

        When `learning_method` is 'online', use mini-batch update.
        Otherwise, use batch update.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Document word matrix.

        y : Ignored.

        Returns
        -------
        self
        """
        self._check_params()
        X = self._check_non_neg_array(X, "LatentDirichletAllocation.fit")
        n_samples, n_features = X.shape
        max_iter = self.max_iter
        evaluate_every = self.evaluate_every
        learning_method = self.learning_method
        if learning_method is None:
            warnings.warn("The default value for 'learning_method' will be "
                          "changed from 'online' to 'batch' in the release "
                          "0.20. This warning was introduced in 0.18.",
                          DeprecationWarning)
            learning_method = 'online'

        batch_size = self.batch_size

        # initialize parameters
        self._init_latent_vars(n_features)
        # change to perplexity later
        last_bound = None
        n_jobs = _get_n_jobs(self.n_jobs)
        with Parallel(n_jobs=n_jobs, verbose=max(0,
                      self.verbose - 1), temp_folder=self.temp_folder) as parallel:
            for i in range(max_iter):
                if learning_method == 'online':
                    for idx_slice in gen_batches(n_samples, batch_size):
                        self._em_step(X[idx_slice, :], total_samples=n_samples,
                                      batch_update=False, parallel=parallel)
                else:
                    # batch update
                    self._em_step(X, total_samples=n_samples,
                                  batch_update=True, parallel=parallel)

                # check perplexity
                if evaluate_every > 0 and (i + 1) % evaluate_every == 0:
                    doc_topics_distr, _ = self._e_step(X, cal_sstats=False,
                                                       random_init=False,
                                                       parallel=parallel)
                    bound = self._perplexity_precomp_distr(X, doc_topics_distr,
                                                           sub_sampling=False)
                    if self.verbose:
                        print('iteration: %d of max_iter: %d, perplexity: %.4f'
                              % (i + 1, max_iter, bound))

                    if last_bound and abs(last_bound - bound) < self.perp_tol:
                        break
                    last_bound = bound

                elif self.verbose:
                    print('iteration: %d of max_iter: %d' % (i + 1, max_iter))
                self.n_iter_ += 1

        # calculate final perplexity value on train set
        doc_topics_distr, _ = self._e_step(X, cal_sstats=False,
                                           random_init=False,
                                           parallel=parallel)
        self.bound_ = self._perplexity_precomp_distr(X, doc_topics_distr,
                                                     sub_sampling=False)

        return self

       
class TopicExtraction():
    
    def __init__(self):
        self.config = configparser.RawConfigParser()
        self.config.read('resources/config.properties')
        
        self.tf_feature_names=[]
        self.vectorizer=LemmaCountVectorizer()

    def find_labels(self, n_labels, label_min_df, tag_constraints, tagged_docs, n_cand_labels, docs):
        cand_labels = []
        while (len(cand_labels)<n_labels):
#             print("Generate candidate bigram labels(with POS filtering)...")
            finder = BigramLabelFinder('pmi', min_freq=label_min_df,
                                       pos=tag_constraints)
    
            if tag_constraints:
                cand_labels = finder.find(tagged_docs, top_n=n_cand_labels)
            else:  # if no constraint, then use untagged docs
                cand_labels = finder.find(docs, top_n=n_cand_labels)
        
            print("Finding Collected {} candidate labels".format(len(cand_labels)))
            if len(cand_labels)>=n_labels:
                break
            else:
                label_min_df -= 1
                if (label_min_df<1):
                    # build tags based on Singular Noun, Noun and Adjetive, Noun
                    label_tags = ['NNS,NN', 'NN,NN', 'JJ,NN']
                    label_min_df = 5
                    tag_constraints = []
                    for tags in label_tags:
                        tag_constraints.append(tuple(map(lambda t: t.strip(),
                                                             tags.split(','))))

        
        return cand_labels
        
    def load_stopwords(self):
        with codecs.open(CURDIR + '/resources/stopwords_en.txt', mode='r',encoding='utf8') as f:
            return map(lambda s: s.strip(),
                       f.readlines())
    # Define helper function to print top words
    def build_top_words(self, model, feature_names, n_top_words):
        for index, topic in enumerate(model.components_):
            topic_name = "Topic_#{}:".format(index)
#             message += " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1 :-1]])
            topic_json = {"topic" : topic_name,
                          "terms" : " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1 :-1]])  
                }
            self.topics["topics"].append(topic_json)
#             print(topic_json)
#         print("="*70)

    def print_top_words(self, model, feature_names, n_top_words):
        for topic_idx, topic in enumerate(model.components_):
            message = "Topic #%d: " % topic_idx
            message += " ".join([feature_names[i]
                                 for i in topic.argsort()[:-n_top_words - 1:-1]])
            print(message)
        print()
    
    def save_lda_topic_model(self):
        joblib.dump(self.lda, 'topic_model_small.lda') 
        
    def load_skl_lda_topic_model(self):
        print('Loading topics...')
        self.lda = TunedLatentDirichletAllocation()
        self.lda = joblib.load('topic_model_small.lda')
        self.vectorizer = self.load_term_vector_topic_model()
        self.tf_feature_names=self.vectorizer.get_feature_names()
        
    def load_glda_topic_model(self):
        print('Loading topics...') 
        self.glda = guidedlda.GuidedLDA(n_topics=1)
        self.glda = joblib.load('glda_topic_model.lda')
        self.glda_vectorizer = self.load_glda_term_vector_topic_model()
        self.glda_tf_feature_names=self.glda_vectorizer.get_feature_names()
        
    def load_glda_term_vector_topic_model(self):
        vectorizer = LemmaCountVectorizer()
        vectorizer = joblib.load('glda_vectorizer_topic_model.vtf')
        return vectorizer

    def load_gen_lda_topic_model(self):
        self.ldaGen = joblib.load('/data/gensim_topic_model.lda')
        
    def load_term_vector_topic_model(self):
        vectorizer = LemmaCountVectorizer()
        vectorizer = joblib.load('vectorizer_topic_model_small.vtf')
        return vectorizer

    def get_topic_extraction_skl(self, message, id):
#         self.load_lda_topic_model()
        
        tf_vectorizer = CountVectorizer(max_df=1, 
                                        min_df=1,
                                        vocabulary=self.tf_feature_names)

        docs = []

        message = re.sub('\n',' ', message)
        docs = self.message_corpus(message)
        
        print('Building BiGrams from the message...')
        bigram = Phrases(docs, min_count=2, threshold=2, delimiter=b' ')
        bigram_phraser = Phraser(bigram)
        
        texts = [bigram_phraser[line] for line in docs]

        bg_message = ' '.join(texts[0])
        
        tf = tf_vectorizer.fit_transform([bg_message])

        doc_topic = self.lda.transform(tf)

        print(doc_topic)
        
        for i in range(len(doc_topic)):
            print("top topic: {} Document: {}".format(doc_topic[i].argmax(),
                                                ', '.join(np.array(self.tf_feature_names))))
        return doc_topic


    def get_topic_extraction_glda(self, message, id):
#         self.load_lda_topic_model()
        
        tf_vectorizer = CountVectorizer(max_df=1, 
                                        min_df=1,
                                        vocabulary=self.glda_tf_feature_names)

        docs = []

        logger.propagate = False

        message = re.sub('\n',' ', message)
        docs = self.message_corpus(message)
        
        print('Building BiGrams from the message...')
        bigram = Phrases(docs, min_count=2, threshold=2, delimiter=b' ')
        bigram_phraser = Phraser(bigram)
        
        texts = [bigram_phraser[line] for line in docs]

        bg_message = ' '.join(texts[0])
        
        tf = tf_vectorizer.fit_transform([bg_message])

        doc_topic = self.glda.transform(tf)

        self.config_dict = dict(self.config.items('TOPIC_LABEL'))
        
        list_topic_names = eval(self.config_dict['list_topic_names']) 

        document_topics = [
                    (list_topic_names[topicid], topicvalue) for topicid, topicvalue in enumerate(doc_topic[0])
                    if topicvalue >= 0.01
                ]        
        
        document_topics = sorted(document_topics, key=lambda score: score[1], reverse=True) 
        
#         print(document_topics)



        return document_topics


    def get_topic_extraction_detail(self, message, id):
#         self.load_lda_topic_model()
        
        tf_vectorizer = CountVectorizer(max_df=1, 
                                        min_df=1,
                                        vocabulary=self.glda_tf_feature_names)

        docs = []

        message = re.sub('\n',' ', message)
        docs = self.message_corpus(message)
        
        print('Building BiGrams from the message...')
        bigram = Phrases(docs, min_count=2, threshold=2, delimiter=b' ')
        logger.propagate = False

        bigram_phraser = Phraser(bigram)
        
        texts = [bigram_phraser[line] for line in docs]

        bg_message = ' '.join(texts[0])
        
        tf = tf_vectorizer.fit_transform([bg_message])

        print('Extracting topics...')
        doc_topic = self.glda.transform(tf)

        document_topics = [
                    (topicid, topicvalue) for topicid, topicvalue in enumerate(doc_topic[0])
                    if topicvalue >= 0.01
                ]        
        
        document_topics = sorted(document_topics, key=lambda score: score[1], reverse=True) 
        
        doc_distribution = np.array([tup[0] for tup in document_topics])

#         print(doc_distribution)

        print('Extracting term per topic...')
        count_vec = np.asarray(tf.sum(axis=0)).ravel()
        zipped = list(zip(self.glda_tf_feature_names, count_vec))
        x, y = (list(x) for x in zip(*sorted(zipped, key=lambda x: x[1], reverse=True)))
        Y = np.concatenate([y[0:tf.indices.shape[0]], y[-1:-1]])
        X = np.concatenate([x[0:tf.indices.shape[0]], x[-1:-1]])
#         for i in range(len(X)):
#             print("Top to Bottom Frequent Words : {} , count: {}".format(X.tolist()[i],Y.tolist()[i]))

        self.config_dict = dict(self.config.items('TOPIC_LABEL'))
        
        list_topic_names = eval(self.config_dict['list_topic_names']) 

        doc_term_topic={"topics":[]}
        # print topics with words and score rank
        for i in doc_distribution[-40:][::-1]:
#             topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
#             print('Topic {}: {}'.format(i, ' '.join(topic_words)))
            print('Extracting term for topic: '+repr(i))
            topic_ = self.glda.topic_word_[i]
            topic_ = topic_ / topic_.sum()  # normalize to probability distribution
            bestn = matutils.argsort(topic_, topic_.shape[0], reverse=True)
            topic_ = sorted([(self.glda_tf_feature_names[id], topic_[id]) for id in bestn],reverse=True)
            
            topic_terms = {"terms":[]}
            
            ss = dict((k,1.0) for k in X)
            test=dict(topic_)
            d = {x:test[x] for x in test if x in ss}

            for term in X:
                topic_id = ''.join(['{"term":"%s" ,"score": %.6f}' % (term, d[term]*100)])
                topic_terms["terms"].append(json.loads(topic_id))

#                 print(topic_terms)
#                             for k, v in topic_:
#                                 {k: v for k, v in topic_}
#                             }
#                 }

            topic_terms["terms"] = sorted(topic_terms["terms"], key=lambda k: k['score'], reverse=True)
            topic = {"score":[score for k, score in document_topics if i == k][0],"terms":topic_terms["terms"],"topic":list_topic_names[i]}
            doc_term_topic["topics"].append(topic)
#             doc_term_topic_score.append(topic_id)
#             topic_ = ' + '.join(['%.4f*"%s"' % (v, k) for k, v in topic_])
#             print(topic)

        print('Sorting topics by score...')
        doc_term_topic["topics"] = sorted(doc_term_topic["topics"], key=lambda k: k['score'], reverse=True)

        terms_not_in_topics = json.dumps(list(set(texts[0]).difference(X)))
        doc_term_topic["terms_not_in_topics"] = terms_not_in_topics
#         print(document_topics)

        print('Done!')
        return doc_term_topic


    def get_topic_extraction_gen(self, message, id):
        docs = []

        message = re.sub('\n',' ', message)
        docs = self.message_corpus(message)
        
        print('Building BiGrams from the message...')
        bigram = Phrases(docs, min_count=2, threshold=2, delimiter=b' ')
        bigram_phraser = Phraser(bigram)
        
        texts = [bigram_phraser[line] for line in docs]

        dictionary = corpora.Dictionary(texts)
        
        bow = dictionary.doc2bow(texts[0])
        
        doc_topic = self.ldaGen.get_document_topics(bow=bow)

        doc_distribution = np.array([tup[0] for tup in doc_topic])

        print(doc_topic)
        
        for i in doc_distribution[-40:][::-1]:
            print(i, self.ldaGen.show_topic(topicid=i, topn=30), "\n")
            
#         for i in range(9):
#             print("top topic: {} Document: {}".format(doc_topic[i].argmax(),
#                                            ', '.join(np.array(self.tf_feature_names))))
            
        return doc_topic


    def process_emailthread(self, corpus, n_topics=10):

        n_components = n_topics
        n_top_words = 20     

        label_min_df = 5
        lda_random_state = 12345

        # max number of building labels
        n_labels = n_topics
        n_cand_labels = 300
        
        docs = []

        tokenize=True
        for l in corpus:
            if tokenize:
                sents = nltk.sent_tokenize(l.strip().lower())
                docs.append(list(itertools.chain(*map(
                    nltk.word_tokenize, sents))))
            else:
                docs.append(l.strip())
    
#         print("Stemming...")
#         stemmer = CorpusStemmer()
#         docs = stemmer.transform(docs)

        print("DOC tagging...")
        tagger = CorpusPOSTagger()
        tagged_docs = tagger.transform(docs)

        tag_constraints = []
        
        # build tags based on Singular Noun, Noun and Adjetive, Noun
        label_tags = ['NN,NN', 'JJ,NN', 'NNS,NN' ]
        for tags in label_tags:
            tag_constraints.append(tuple(map(lambda t: t.strip(),
                                                 tags.split(','))))
        
        cand_labels = self.find_labels(n_labels, label_min_df, tag_constraints, tagged_docs, n_cand_labels, docs)
        
        print("Collected {} candidate labels".format(len(cand_labels)))

        print("Calculate the PMI scores...")
    
        pmi_cal = PMICalculator(
            doc2word_vectorizer=CountVectorizer(
                max_df=.95, 
                min_df=5,
                lowercase=True,
                token_pattern= r'\b[a-zA-Z]{3,}\b',
                stop_words=self.load_stopwords()
                ),
            doc2label_vectorizer=LabelCountVectorizer())

        pmi_w2l = pmi_cal.from_texts(docs, cand_labels)
    
        # this example is to be used for future Topic Model builder 
#         try:
#             self.load_lda_topic_model() 
#         except FileNotFoundError:
            # if no lda topic_model file, build a new one
        self.lda = TunedLatentDirichletAllocation(
            n_jobs=40,
            n_components=n_components, max_iter=15,
            learning_method = 'online',
            learning_offset = 50.,
            random_state = 12)
    
        ##
        self.lda.fit(pmi_cal.d2w_)
        
#             self.save_lda_topic_model() 
        
        print("\nTopical words:")
#         print("-" * 20)
        topic_words_list = []
        for i, topic_dist in enumerate(self.lda.components_):
            top_word_ids = np.argsort(topic_dist)[:-2000:-1]
            topic_words = [pmi_cal.index2word_[id_]
                           for id_ in top_word_ids]
            print('Topic {}: {}'.format(i, ' '.join(topic_words)))
            topic_words_list.append(topic_words)
    
        ranker = LabelRanker(apply_intra_topic_coverage=False, apply_inter_topic_discrimination=True)
    
        labels = ranker.top_k_labels(topic_models=self.lda.components_,
                                   pmi_w2l=pmi_w2l,
                                   index2label=pmi_cal.index2label_,
                                   label_models=None,
                                   k=n_labels
                                   )

        ret = lil_matrix((len(topic_words_list), len(labels)),
                         dtype=int64)
        
        for i, d in enumerate(topic_words_list):
            for j, l in enumerate(labels[i]):
                cnt = self.label_relevance(l, d, i)
                if cnt > 0:
                    ret[i, j] = cnt
                else:
                    ret[i, j] = 1999

        print("\nTopical labels:")
        print("-" * 20)
        str_labels=[]
        for i, topic_labels in enumerate(labels):
            _list = ','.join(map(lambda l: ' '.join(l), topic_labels)).split(',')
            str_labels.append(_list)
                    
            print(u"Topic {}: {}\n".format(
                i,
                ', '.join(map(lambda l: ' '.join(l), topic_labels))
            ))
            if (len(str_labels)==n_components):
                break
    
        print("\nTopics in LDA model: ")
        self.tf_feature_names = pmi_cal._d2w_vect.get_feature_names()
        
        py_lda_vis = MyDisplay()

        
        self.topic_names = str_labels
        
        visualization = py_lda_vis.prepare(self.lda, pmi_cal.d2w_, pmi_cal._d2w_vect, 'tsne')
        json_data = visualization.to_json()
        
        list_topic_labels = []
        # processing the labels in the same order of topic relevance
        print(list(visualization[6:])[0])
        
        for i,topic in enumerate(list(visualization[6:])[0]):
            score=10000
            topic_label=''
            for label in ret[topic-1,].rows[0]:
                label_score=ret[topic-1,].data[0][label]
                if label_score<score:
                    try:
                        list_topic_labels.index(str_labels[topic-1][label])
                    except ValueError:
                        topic_label=str_labels[topic-1][label]
                        score=label_score
                
            list_topic_labels.append(topic_label)
            
        topic_name = {"topic.names" : list_topic_labels}
        
        visualization_html = py_lda_vis.prepared_data_to_html(visualization,
                                                              json_names=topic_name)
        py_lda_vis.save_html(visualization, 
                             'LDA_Visualization_labels.html',
                             json_names=topic_name)
        
        self.encoded_html = base64.b64encode(visualization_html.encode())
        
        self.topics = {}
        self.topics["topics"] = []

        self.build_top_words(self.lda, self.tf_feature_names, n_top_words)
    
        self.topic_names = list_topic_labels
        
    def label_relevance(self, label_tokens, context_tokens, topic_id):
        """
        Calculate the relevance position that the label appears
        in the context of the topics words frequency
        
        Parameter:
        ---------------

        label_tokens: list|tuple of str
            the label tokens
        context_tokens: list|tuple of str
            the sentence tokens

        Return:
        -----------
        int: the label frequency in the sentence
        """
        label_len = len(label_tokens)
        cnt = 0
        
        pos = []
        for i in range(0,len(context_tokens) - label_len + 1):
            for j in range(0,label_len):
                if label_tokens[j] == context_tokens[i+j]:
                    pos.append(i+1)
        
        relevance_index=int(sum(pos)/2)
        
        print({"topic_id":topic_id, "relevance_score": relevance_index, "topic_label":label_tokens})
        return relevance_index

    def get_topics(self):
        return self.topics
    
    def get_topic_name(self):
        return self.topic_names

    def get_topic_name_subject(self):
        return self.topic_names_subject

    def get_topic_visualization(self):
        return self.encoded_html

    def get_features_names(self):
        return self.tf_feature_names
    
    def get_lda(self):
        return self.lda
    
    def message_corpus(self, message):
        docs = []
        
        stop = list(self.load_stopwords())
        stop.append('off')
        stop.append('http')
        stop.append('www')
        stop.append('edt')
        stop.append('est')
        stop.append('mdt')
        stop.append('pst')
        stop.append('pt')
        
        line = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+.[a-zA-Z0-9-]+"," ",message.lower())
        line = re.sub(r'[^\x00-\x7f]',r' ', line) 
        line = re.sub(r"\b(https?://)?(www\.)?([a-z0-9\-.]+)(?:\.[a-z\.]+[\/]?)\w+/(?:\w+/)+(\S+)\b"," ", line)
        line = re.sub(r"(^|\W)\d+", " ", line)
        replaces = {"Subject:":"","\"":" ",",":" ",".":" ","-":" ",":":" ","/":" ","$":"","(":"",")":"","*":"","!":"",
                    "[":"","]":"","=":"","&":"","'":"","`":"","#":"","_":"","@":"","error occurred attempting initialize borland database engine error":"",
                    ";":"",">":"","<":"","?":"","  ":" ","\\":" ","\n":" ","\t":" ","%":" ","\'":" ","\"":" ","—":" "}
        line = self.replace_all(line, replaces)
#         line = re.sub(r'\W*\b\w{1,2}\b', " ", line)
        
        sents = nltk.sent_tokenize(line.strip())
        sents = list(itertools.chain(*map(nltk.word_tokenize, sents)))
        sents = [i for i in sents if i not in stop]
#                     lanste = LancasterStemmer()
#                     sents = [lanste.stem(i) for i in sents]

        docs.append(sents)

        return docs

    def get_doc_topics_lda(self, doc):
        tf_vectorizer = CountVectorizer(max_df=1, 
                                        min_df=1,
                                        vocabulary=self.tf_feature_names)
        tf = tf_vectorizer.fit_transform(doc)

#         feature_names = tf_vectorizer.get_feature_names()
#         count_vec = np.asarray(tf.sum(axis=0)).ravel()
#         zipped = list(zip(feature_names, count_vec))
#         x, y = (list(x) for x in zip(*sorted(zipped, key=lambda x: x[1], reverse=True)))
#         # Now I want to extract out on the top 15 and bottom 15 words
#         Y = np.concatenate([y[0:20], y[-21:-1]])
#         X = np.concatenate([x[0:20], x[-21:-1]])
#         
#         for i in range(len(X)):
#             print("LDA Match Words: {} , count: {}".format(X.tolist()[i],Y.tolist()[i]))
        
        doc_topic = self.lda.transform(tf)

#         for i in range(9):
#             print("top topic: {} Document: {}".format(doc_topic[i].argmax(),
#                                               ', '.join(np.array(self.tf_feature_names))))
        return doc_topic

    def get_doc_topics_nmf1(self, doc):
        tfidf_vectorizer = TfidfVectorizer(max_df=0.95, 
                                           min_df=2,
                                           vocabulary=self.tfidf_feature_names)
        nmf1 = tfidf_vectorizer.fit_transform([doc])
        
#         feature_names = tfidf_vectorizer.get_feature_names()
#         count_vec = np.asarray(nmf1.sum(axis=0)).ravel()
#         zipped = list(zip(feature_names, count_vec))
#         x, y = (list(x) for x in zip(*sorted(zipped, key=lambda x: x[1], reverse=True)))
#         # Now I want to extract out on the top 15 and bottom 15 words
#         Y = np.concatenate([y[0:20], y[-21:-1]])
#         X = np.concatenate([x[0:20], x[-21:-1]])
#         
#         for i in range(len(X)):
#             print("NMF1 Match Words: {} , count: {}".format(X.tolist()[i],Y.tolist()[i]))

        n_top_words=20

        doc_topic = self.nmf1.transform(nmf1)
        for index, topic in enumerate(doc_topic):
            print("top topic: {} Document: {}".format(doc_topic[index].argmax(),
                                              ', '.join([self.tfidf_feature_names[i] for i in self.nmf1.components_[doc_topic[index].argmax()].argsort()[:-n_top_words - 1 :-1]])))

#         for i in range(9):
#             print("top topic: {} Document: {}".format(doc_topic[i].argmax(),
#                                               ', '.join(np.array(self.tf_feature_names))))
        return doc_topic

    def get_doc_topics_nmf2(self, doc):
        tfidf_vectorizer = TfidfVectorizer(max_df=0.95, 
                                           min_df=2,
                                           vocabulary=self.tfidf_feature_names)
        nmf2 = tfidf_vectorizer.fit_transform([doc])
        
#         feature_names = tfidf_vectorizer.get_feature_names()
#         count_vec = np.asarray(nmf2.sum(axis=0)).ravel()
#         zipped = list(zip(feature_names, count_vec))
#         x, y = (list(x) for x in zip(*sorted(zipped, key=lambda x: x[1], reverse=True)))
#         # Now I want to extract out on the top 15 and bottom 15 words
#         Y = np.concatenate([y[0:20], y[-21:-1]])
#         X = np.concatenate([x[0:20], x[-21:-1]])
#         
#         for i in range(len(X)):
#             print("NMF2 Match Words: {} , count: {}".format(X.tolist()[i],Y.tolist()[i]))

        n_top_words=20
        doc_topic = self.nmf2.transform(nmf2)
        for index, topic in enumerate(doc_topic):
            print("top topic: {} Document: {}".format(doc_topic[index].argmax(),
                                              ', '.join([self.tfidf_feature_names[i] for i in self.nmf1.components_[doc_topic[index].argmax()].argsort()[:-n_top_words - 1 :-1]])))

        return doc_topic
        
    def replace_all(self,text, dic):
        for i, j in dic.items():
            text = text.replace(i, j)
        return text