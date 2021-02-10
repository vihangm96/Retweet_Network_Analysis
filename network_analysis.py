import json
import networkx as nx
import sys

def load_data(input_file):
    
    #create the multigraph
    G = nx.DiGraph()      
    
    #read in the json file
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            tweet = json.loads(line)
            
            if 'retweeted_status' in tweet:
                retweeter = tweet['user']['screen_name']
                original_tweeter = tweet['retweeted_status']['user']['screen_name']
                if G.has_edge(retweeter, original_tweeter):
                    G[retweeter][original_tweeter]['weight'] += 1
                else:
                    G.add_node(original_tweeter)
                    G.add_node(retweeter)
                    G.add_edge(retweeter, original_tweeter, weight=1)
                    
            else:
                tweeter = tweet['user']['screen_name']
                if not G.has_node(tweeter):
                    G.add_node(tweeter)            

                continue
    
    return G

if __name__ == "__main__":
    
    input_file = sys.argv[1]
    graph_output_file = sys.argv[2]
    json_output_file = sys.argv[3]

    # input_file = "toy_test/mini_mid_gamergate.json"
    # json_output_file = "task11_output.json"
    
    G = load_data(input_file)
    
    #TASK A
    #save the graph as a gxef
    nx.write_gexf(G, graph_output_file)

    #TASK B
    n_nodes = G.number_of_nodes()

    #TASK C
    n_edges = G.number_of_edges()

    #TASK D E
    most_retweeted_data = sorted([(n, d) for n,d in G.in_degree(weight='weight')], key=lambda x: x[1], reverse=True)[0]
    max_retweeted_user = most_retweeted_data[0]
    max_retweeted_number = most_retweeted_data[1]

    #TASK F G
    most_retweeting_data = sorted([(n, d) for n,d in G.out_degree(weight='weight')], key=lambda x: x[1], reverse=True)[0]
    max_retweeter_user = most_retweeting_data[0]
    max_retweeter_number = most_retweeting_data[1]

    solution = {}

    solution['n_nodes'] = n_nodes
    solution['n_edges'] = n_edges
    solution['max_retweeted_user'] = max_retweeted_user
    solution['max_retweeted_number'] = max_retweeted_number
    solution['max_retweeter_user'] = max_retweeter_user
    solution['max_retweeter_number'] = max_retweeter_number

    solution_string = json.dumps(solution)
    out_file = open(json_output_file, "w")  
    out_file.write(solution_string)
    out_file.close()  