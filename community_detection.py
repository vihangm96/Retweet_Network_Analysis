import json
import networkx as nx
import sys
import copy
import networkx.algorithms.community as nx_comm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(input_file):
    
    #create the graph
    G = nx.Graph()   
    
    #read from the json file
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            
            tweet = json.loads(line)
            
            if 'retweeted_status' in tweet:
            
                retweeter = tweet['user']['screen_name']
                original_tweeter = tweet['retweeted_status']['user']['screen_name']
                
                retweet_text = tweet['text']
                original_text = tweet['retweeted_status']['text']
                
                if G.has_edge(retweeter, original_tweeter):
                    
                    G[retweeter][original_tweeter]['weight'] += 1
                
                    G.nodes[retweeter]['tweets'] += ' ' + retweet_text
                    G.nodes[original_tweeter]['tweets'] += ' ' +original_text
                    
                else:
                    
                    if G.has_node(retweeter):
                        G.nodes[retweeter]['tweets'] += ' ' + retweet_text
                    else:
                        G.add_node(retweeter, tweets=retweet_text)
                        
                    if G.has_node(original_tweeter):
                        G.nodes[original_tweeter]['tweets'] += ' ' + original_text
                    else:
                        G.add_node(original_tweeter, tweets=original_text)

                    G.add_edge(retweeter, original_tweeter, weight=1)
                    
            else:
                
                tweeter = tweet['user']['screen_name']
                tweet_text = tweet['text']
                
                if G.has_node(tweeter):
                    G.nodes[tweeter]['tweets'] += ' ' + tweet_text
                else:
                    G.add_node(tweeter, tweets=tweet_text)
                
                continue
    return G

def betweenness(G):
    return nx.edge_betweenness_centrality(G, weight='weight')

def modularity(G, degrees_dict, two_m):

    modularity_score = 0
                        
    for s in nx.connected_components(G):
        
        if len(s)==1:
            continue
        
        for i in s:
            for j in s:
                
                if i==j:
                    continue
                
                if G.has_edge(i, j):
                    A_ij = G[i][j]['weight']
                else:
                    A_ij = 0
                
                k_i = degrees_dict[i]
                k_j = degrees_dict[j]
                
                modularity_score += (A_ij - (k_i*k_j)/(two_m))

    modularity_score /= two_m
    
    return modularity_score

def girvan_newman(G):

    max_modularity_yet = float('-inf')
            
    degrees_dict = dict()
    
    for node in G.nodes():
        degrees_dict[node] = G.degree(node, weight='weight')
            
    m = sum(nx.get_edge_attributes(G,'weight').values())
    two_m = 2*m
    
    while G.number_of_edges() > 0:
        
        prev_connected_components = nx.number_connected_components(G)
        
        edge_betweenness_scores = betweenness(G)
        
        max_betweeness = max(edge_betweenness_scores.items(),key = lambda x:x[1])[1]
        max_betweeness_list =[i[0] for i in edge_betweenness_scores.items() if i[1]==max_betweeness]
        G.remove_edges_from(max_betweeness_list)
        
        if prev_connected_components == nx.number_connected_components(G):
            continue

        current_modularity = modularity(G, degrees_dict, two_m)
        
        if max_modularity_yet < current_modularity:
            max_modularity_yet = current_modularity
            best_communities_yet = copy.deepcopy(G)        
    
    return max_modularity_yet, best_communities_yet

def count_vectorizer_classification(communities,all_tweets):
    
    max_community_1 = communities[-1]
    max_community_2 = communities[-2]
    
    Y = []
    label1_tweets = []
    label2_tweets = []
    
    for node in max_community_1:
        label1_tweets.append(all_tweets[node])
        Y.append(1)
    
    for node in max_community_2:
        label2_tweets.append(all_tweets[node])
        Y.append(2)
    
    #extract the countvectorizer features for the train data
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(label1_tweets + label2_tweets)
    
    #train a multinomial naive Bayes classifier
    clf = MultinomialNB().fit(X_train_counts, Y)

    for i in range(len(communities)-2):
        community = communities[i]
        for node in community:
            tweets_text = [all_tweets[node]]
            docs_test = count_vect.transform(tweets_text)
            predicted = clf.predict(docs_test)
            if predicted[0] == 1:
                max_community_1.append(node)
            else:
                max_community_2.append(node)

    return [sorted(max_community_1), sorted(max_community_2)]

def tfidf_classification(communities,all_tweets):
    
    max_community_1 = communities[-1]
    max_community_2 = communities[-2]
    
    Y = []
    label1_tweets = []
    label2_tweets = []
    
    for node in max_community_1:
        label1_tweets.append(all_tweets[node])
        Y.append(1)
    
    for node in max_community_2:
        label2_tweets.append(all_tweets[node])
        Y.append(2)
    
    #extract the TfidfVectorizer features for the train data
    
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(label1_tweets + label2_tweets)
    
    #train a multinomial naive Bayes classifier
    clf = MultinomialNB().fit(X, Y)
    
    for i in range(len(communities)-2):
        community = communities[i]
        for node in community:
            tweets_text = [all_tweets[node]]
            docs_test = vectorizer.transform(tweets_text)
            predicted = clf.predict(docs_test)
            if predicted[0] == 1:
                max_community_1.append(node)
            else:
                max_community_2.append(node)
    
    return [sorted(max_community_1), sorted(max_community_2)]

if __name__ == "__main__":

    # input_file = "toy_test/mini_mid_gamergate.json"
    # taskA_output_file = "taskA_output.txt"
    # taskB_output_file = "taskB_output.txt"
    # taskC_output_file = "taskC_output.txt"

    input_file = sys.argv[1]
    taskA_output_file = sys.argv[2]
    taskB_output_file = sys.argv[3]
    taskC_output_file = sys.argv[4]

    tweet_graph = load_data(input_file)

    # TASK 2 A
    max_modularity, communities = girvan_newman(tweet_graph)

    out_file = open(taskA_output_file, "w")  
    out_file.write('Best Modularity is: ')
    out_file.write(str(max_modularity))
    out_file.write("\n")
    S = [(len(s),sorted([x for x in s])) for s in nx.connected_components(communities)]
    sorted_communities = sorted(S)
    for s in sorted_communities: 
        out_file.write(','.join(['\'' + node + '\'' for node in s[1]]))
        out_file.write("\n")
    out_file.close()  

    all_communities = [community[1] for community in sorted_communities]
    all_tweets = nx.get_node_attributes(tweet_graph, "tweets")

    # TASK 2 B
    max_communities_B = tfidf_classification(copy.deepcopy(all_communities), all_tweets)

    out_file = open(taskB_output_file, "w")  
    S_B = [(len(s),s) for s in max_communities_B]
    sorted_communities_B = sorted(S_B)
    for sb in sorted_communities_B: 
        out_file.write(','.join(['\'' + node + '\'' for node in sb[1]]))
        out_file.write("\n")
    out_file.close()  

    # TASK 2 C
    max_communities_C = count_vectorizer_classification(copy.deepcopy(all_communities), all_tweets)

    out_file = open(taskC_output_file, "w")  
    S_C = [(len(s),s) for s in max_communities_C]
    sorted_communities_C = sorted(S_C)
    for sc in sorted_communities_C: 
        out_file.write(','.join(['\'' + node + '\'' for node in sc[1]]))
        out_file.write("\n")
    out_file.close()  