from flask import Flask, jsonify, request
import json
from pyLDAvis import _display
import re
import requests
from textblob import TextBlob
import time

from net.globalrelay.nlp.topic_modeler.topic_extraction import TopicExtraction
from net.globalrelay.nlp.topic_modeler.lemma_count_vectorizer import LemmaCountVectorizer

from funcy.colls import none


app = Flask(__name__)

is_topic_sklear_loaded = False
is_topic_glda_loaded = False
is_topic_gensim_loaded = False
        
topicExtraction = TopicExtraction()

@app.route("/")
def hello_world():
  return "Hello, World!"


@app.route('/topic-extraction-glda', methods=['POST'])
def get_topic_extraction_glda():
    json_message  = request.json
    message_id = json_message['id']
    message = json_message['message']

    global is_topic_glda_loaded
    
    if is_topic_glda_loaded==False:
        topicExtraction.load_glda_topic_model()
        is_topic_glda_loaded = True
    
    doc_topic = topicExtraction.get_topic_extraction_glda(message, message_id)
    
    topic_name = []
    for id, score in doc_topic:
        topic = {"topic" : "{}".format(id),
                 "score" : repr(score)}
        topic_name.append(topic)

    print(topic_name)
    
    topic_json = {"_id" : message_id,
                  "topics" : topic_name  
        }
    
    return jsonify(topic_json)
    
@app.route('/topic-extraction-details', methods=['POST'])
def get_topic_extraction_detail():
    json_message  = request.json
    message_id = json_message['id']
    message = json_message['message']

    global is_topic_glda_loaded
    
    if is_topic_glda_loaded==False:
        topicExtraction.load_glda_topic_model()
        is_topic_glda_loaded = True
    
    doc_topic = topicExtraction.get_topic_extraction_detail(message, message_id)
    
    
    return jsonify(doc_topic)
    
@app.route('/topic-extraction-skl', methods=['POST'])
def get_topic_extraction_skl():
    json_message  = request.json
    message_id = json_message['id']
    message = json_message['message']

    global is_topic_sklear_loaded
    
    if is_topic_sklear_loaded==False:
        topicExtraction.load_skl_lda_topic_model()
        is_topic_sklear_loaded = True
    
    doc_topic = topicExtraction.get_topic_extraction_skl(message, message_id)
    
    topic_name = []
    for i in range(len(doc_topic)):
        topic_name.append("Topic_{}".format(doc_topic[i].argmax()))

    topic_json = {"_id" : message_id,
                  "topics" : topic_name  
        }
    
    return jsonify(topic_json)
    
@app.route('/topic-extraction-gen', methods=['POST'])
def get_topic_extraction_gen():
    json_message  = request.json
    message_id = json_message['id']
    message = json_message['message']

    global is_topic_gensim_loaded
    
    if is_topic_gensim_loaded==False:
        topicExtraction.load_gen_lda_topic_model()
        is_topic_gensim_loaded = True

    doc_topic = topicExtraction.get_topic_extraction_gen(message, message_id)

    topic_name = []
    for id, score in doc_topic:
        topic = {"topic" : "Topic_{}".format(id),
                 "score" : repr(score)}
        topic_name.append(topic)

    print(topic_name)
    
    topic_json = {"_id" : message_id,
                  "topics" : topic_name  
        }
    
    return jsonify(topic_json)
    
    
@app.route('/emailthread', methods=['POST'])
def get_emailthread():
    email_from_to  = request.json
    print('emails ', email_from_to)
    print('emails1 ', email_from_to['from'])
    print('emails2 ', email_from_to['to'])
    
    email_from = email_from_to['from']
    email_to = email_from_to['to']
    
    str_from=''
    for i in range(len(email_from)):
        str_from += 'emailAddresses.from.keyword:'+email_from[i]
        if i<(len(email_from)-1):
            str_from += ' OR '
        
    str_to=''
    for i in range(len(email_to)):
        str_to += 'emailAddresses.to.keyword:'+email_to[i]
        if i<(len(email_to)-1):
            str_to += ' OR '

    str_cc=''
    for i in range(len(email_to)):
        str_cc += 'emailAddresses.cc.keyword:'+email_to[i]
        if i<(len(email_to)-1):
            str_cc += ' OR '

    str_bcc=''
    for i in range(len(email_to)):
        str_bcc += 'emailAddresses.bcc.keyword:'+email_to[i]
        if i<(len(email_to)-1):
            str_bcc += ' OR '

    try:
            n_topics = email_from_to['numTopics']
    except KeyError:
            n_topics = 10
    try:
            year = email_from_to['year']
    except KeyError:
            year = 0
    try:
            month = email_from_to['month']
    except KeyError:
            month = 0
    try:
            day = email_from_to['day']
    except KeyError:
            day = 0

    date_filter = ""
    if year>0:
        date_filter = date_filter + " AND year:"+repr(year)
    if month>0:    
        date_filter = date_filter + " AND month:"+repr(month)
    if day>0:
        date_filter = date_filter + " AND day:"+repr(day)
    
    headers = {
        'Content-Type': 'application/json',
    }
    
    params = {}
    
    query = {"size": 1000, 
        "query":{
            "query_string":{
                "query":"("+str_from+") AND ("+str_to+" OR "+str_cc+" OR "+str_bcc+")"+
                date_filter
                }
            },
        "_source":["emailAddresses.from","emailAddresses.to","body","subject"]
    }
    
#     print('query ', query)

    start = time.time()

    url = 'http://10.178.21.51:9200/messages/_search'
#     url = 'http://dldev6-test.office.globalrelay.net:9200/messages/_search'
    response = requests.post(url, headers=headers, data=json.dumps(query))
    response.raise_for_status()

    emailthread_json = json.loads(response.content)
    
    end = time.time()
    elapsed = end - start
    print(elapsed)

    start = time.time()
    
    corpus = []
    for emails in emailthread_json['hits']['hits']:
        if not emails['_source']['subject'] == None:
            subject = 'Subject: '+emails['_source']['subject']
        else:
            subject = 'Subject: '
            
        body = subject + '\n' +emails['_source']['body']
        corpus.append(build_email_corpus(body.replace("Group", "")))
        
    if len(corpus)==0:
        result = {}
        return jsonify(result)
    
#     topicExtraction = TopicExtraction(corpus, n_topics)

    topics = topicExtraction.get_topics()

    end = time.time()
    elapsed = end - start
    print(elapsed)

    idx=0
    documents = []

    for emails in emailthread_json['hits']['hits']:
        body = build_email_corpus(emails['_source']['body'])
        documents.append(body)
    
#     print("\nTopics/Document in LDA model:")
    
    print('Topic extraction')
    start = time.time()
    doc_topic = topicExtraction.get_doc_topics_lda(documents)

    messages = {}
    messages["messages"] = []
    
    
    index = 0
    for emails in emailthread_json['hits']['hits']:
        email_id = emails['_id']

        topic_name = "{}".format(topicExtraction.get_topic_name()[doc_topic[index].argmax()])
        topic_json = {"_id" : email_id,
                      "topic" : topic_name  
            }
        
        messages["messages"].append(topic_json)
        index += 1
        #print(query)
         
#         url = 'http://localhost:8080/nlp-service/topic-ext/query'
#         response = requests.post(url, headers=headers, data=query)
#         response.raise_for_status()
 
#         n_top_words=20
#         
#         print("top topic: {} Document: {}".format(doc_topic[i].argmax(),
#                                       ', '.join([topicExtraction.get_features_names()[i] for i in topicExtraction.get_lda().components_[doc_topic[i].argmax()].argsort()[:-n_top_words - 1 :-1]])))
# 
#         print("Topic from CogNLP for doc#: "+repr(i)+" -> "+json.dumps(json.loads(response.content)))
#         
#         print("top topic: {} for email# {}".format(doc_topic[i].argmax(),documents[i]))
#         print("="*90)
    
    
    result = {"topic_visualization":topicExtraction.get_topic_visualization(),
              "messages" : messages["messages"] 
        }
    
    end = time.time()
    elapsed = end - start
    print(elapsed)
    print()

#     print("\nTopics/Document in NMF1 model:")
# 
#     for i in range(len(documents)):
#         doc_topic = topicExtraction.get_doc_topics_nmf1(documents[i])
#         print("top topic: {} for email# {}".format(doc_topic.argmax(),documents[i]))
#         print("="*90)
# 
#         
#     print()
# 
#     print("\nTopics/Document in NMF2 model:")
#     
#     for i in range(len(documents)):
#         doc_topic = topicExtraction.get_doc_topics_nmf2(documents[i])
#         print("top topic: {} for email# {}".format(doc_topic.argmax(),documents[i]))
#         print("="*90)
#     print()

#     for i in range(len(documents)):
#         query = '{"id":"123","message":'+json.dumps(documents[i])+'}'
#      
#         #print(query)
#          
#         url = 'http://localhost:8080/nlp-service/topic-ext/query'
#         response = requests.post(url, headers=headers, data=query)
#         response.raise_for_status()
#  
#         print("Topic from CogNLP for doc#: "+repr(i)+" -> "+json.dumps(json.loads(response.content)))
#      
#  
#     idx += 1   
    return jsonify(result)

# Define helper function to print top words
def build_email_corpus(body):
    isBody=True
    newBody=''
    for line in body.splitlines():
        if line.find('----Original Message----')>=0:
            isBody=False
        if line.find('To: ')>=0:
            isBody=False
        if line.find('From: ')>=0:
            isBody=False
        if line.find('cc: ')>=0:
            isBody=False
            
        if line.find('----Original Message----')>=0:
            isBody=False
            
        if line.find('Subject:')>=0:
#             print(line)
            isBody=True
        
        if isBody:    
            line = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+.[a-zA-Z0-9-]+"," ",line)
            line = re.sub(r"(^|\W)\d+", " ", line)
            replaces = {"Subject:":"","\"":" ",",":" ",".":" ","-":" ",":":" ","/":" ","$":"","(":"",")":"","*":"","!":"",
                        "[":"","]":"","=":"","&":"","'":"","`":"","#":"","_":"","@":"","error occurred attempting initialize borland database engine error":"",
                        ";":"",">":"","<":"","?":"","  ":" ","\\":" ","\n":" ","\t":" ","%":" "}
            line = replace_all(line, replaces)
            line = re.sub(r'\W*\b\w{1,2}\b', " ", line)
            newBody += line+' '
            
    return newBody
    
def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text

if __name__ == "__main__":
     app.run(host = "127.0.0.1", port = 5000)