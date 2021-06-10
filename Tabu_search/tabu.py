import math
import time
from itertools import combinations
import sys
from random import randint
from random import shuffle
def MSA_to_TSP(sequences):
    node = []
    for i in range(len(sequences)):
        node.append(sequences[i])    
    graph = [[0.0]*len(node) for i in range(len(node))]
    max_score = 0.0
    for i in range(len(node)):
        graph[i][i] = 0.0
        for j in range(i+1,len(node)):
            score, align1, align2 = get_alignment_score(node[i],node[j]) 
            graph[i][j] = score
            graph[j][i] = score
            if max_score < score: 
                max_score = score                
    for i in range(len(node)):
        for j in range(i+1,len(node)):
            graph[i][j] = max_score - graph[i][j] + 1
            graph[j][i] = max_score - graph[j][i] + 1  
    return graph
def get_alignment_score(v, w, match_penalty=1, mismatch_penalty=-1, deletion_penalty=-1):
    n1 = len(v)
    n2 = len(w)
    s = [[0.0]*(n2+1) for i in range(n1+1)] 
    b =[[0.0]*(n2+1) for i in range(n1+1)]   
    for i in range(n1+1):
        s[i][0] = i * deletion_penalty
        b[i][0] = 1
    
    for j in range(n2+1):
        s[0][j] = j * deletion_penalty
        b[0][j] = 2
    for i in range(1,n1+1):
        for j in range(1,n2+1):
            if v[i-1] == w[j-1]:
                ms = s[i-1][j-1] + match_penalty
            else:
                ms = s[i-1][j-1] + mismatch_penalty
            
            test = [ms, s[i-1][j] + deletion_penalty, s[i][j-1] + deletion_penalty]
            p = max(test)
            s[i][j] = p
            b[i][j] = test.index(p)   
    i = n1
    j = n2
    sv = []
    sw = []
    while i != 0 or j != 0:
        p = b[i][j]
        if p==0:
            i-=1
            j-=1
            sv.append(v[i])
            sw.append(w[j])
        elif p == 1:
            i-=1
            sv.append(v[i])
            sw.append("-")
        elif p == 2:
            j-=1
            sv.append("-")
            sw.append(w[j])
        else:
            break
    
    sv.reverse()
    sw.reverse()
    
    return (s[n1][n2], "".join(sv), "".join(sw))
def parse_data(linkset):
    links = {}
    max_weight = 0
    
    for tmp in linkset:
        if int(tmp[2]) > max_weight:
            max_weight = int(tmp[2])
    for link in linkset:
        try:
            linklist = links[str(link[0])]
            linklist.append(link[1:])
            links[str(link[0])] = linklist
        except:
            links[str(link[0])] = [link[1:]]
        
    return links, max_weight
def getNeighbors(state):
    return two_opt_swap(state)        
def hill_climbing(state):
    node = randint(1, len(state)-1)
    neighbors = []
    
    for i in range(len(state)):
        if i != node and i != 0:
            tmp_state = state.copy()
            tmp = tmp_state[i]
            tmp_state[i] = tmp_state[node]
            tmp_state[node] = tmp
            neighbors.append(tmp_state)
            
    return neighbors

def two_opt_swap(state):
    global neighborhood_size
    neighbors = []
    
    for i in range(neighborhood_size):
        node1 = 0
        node2 = 0
        
        while node1 == node2:
            node1 = randint(1, len(state)-1)
            node2 = randint(1, len(state)-1)
    
        if node1 > node2:
            swap = node1
            node1 = node2
            node2 = swap   
        tmp = state[node1:node2]
        tmp_state = state[:node1] + tmp[::-1] +state[node2:]
        neighbors.append(tmp_state)
        
    return neighbors
def fitness(route, graph):
    path_length = 0
    
    for i in range(len(route)):
        if(i+1 != len(route)):
            dist = weight_distance(route[i], route[i+1], graph)
            if dist != -1:
                path_length = path_length + dist
            else:
                return max_fitness # there is no  such path
        else:
            dist = weight_distance(route[i], route[0], graph)
            if dist != -1:
                path_length = path_length + dist
            else:
                return max_fitness # there is no  such path
            
    return path_length
            
# not used in this code but some datasets has 2-or-more dimensional data points, in this case it is usable
def euclidean_distance(city1, city2):
    return math.sqrt((city1[0] - city2[0])**2 + ((city1[1] - city2[1])**2))

def weight_distance(city1, city2, graph):
    global max_fitness
    
    neighbors = graph[str(city1)]
    
    for neighbor in neighbors:
        if neighbor[0] == int(city2):
            return neighbor[1]
        
    return -1 #there can't be minus distance, so -1 means there is not any city found in graph or there is not such edge
def tabu_search(input_file_path):
    global max_fitness, start_node
    graph, max_weight = parse_data(input_file_path)
    
    ## Below, get the keys (node names) and shuffle them, and make start_node as start
    s0 = list(graph.keys())
    shuffle(s0)
    
    if int(s0[0]) != start_node:
        for i in range(len(s0)):
            if int(s0[i]) == start_node:
                swap = s0[0]
                s0[0] = s0[i]
                s0[i] = swap
                break
    # max_fitness will act like infinite fitness
    max_fitness = ((max_weight) * (len(s0)))+1
    sBest = s0
    vBest = fitness(s0, graph)
    bestCandidate = s0
    tabuList = []
    tabuList.append(s0)
    stop = False
    best_keep_turn = 0
    
    while not stop :
        sNeighborhood = getNeighbors(bestCandidate)
        bestCandidate = sNeighborhood[0]
        for sCandidate in sNeighborhood:
            if (sCandidate not in tabuList) and ((fitness(sCandidate, graph) < fitness(bestCandidate, graph))):
                bestCandidate = sCandidate

        if (fitness(bestCandidate, graph) < fitness(sBest, graph)):
            sBest = bestCandidate
            vBest = fitness(sBest, graph)
            best_keep_turn = 0

        tabuList.append(bestCandidate)
        if (len(tabuList) > maxTabuSize):
            tabuList.pop(0)
            
        if best_keep_turn == stoppingTurn:
            stop = True
            
        best_keep_turn += 1

    return sBest, vBest
    
maxTabuSize = 10000
neighborhood_size = 500
stoppingTurn = 500
max_fitness = 0
start_node = 0

def find_gap_indices(A, alignedA):
    i = 0
    j = 0
    pointer = []
    while j < len(alignedA):
        if alignedA[j] == '-' and (i > len(A) or A[i] != '-'):
            pointer.append(j)
            j += 1
        else:
            j += 1
            i += 1
    return pointer

def insert_gaps(S,gap_indices_for_A):
    copy_of_S = S
    if len(gap_indices_for_A) > 0 and len(gap_indices_for_A) > 0:
        gap_indices_for_A.sort()
        for i in gap_indices_for_A:
            copy_of_S = (copy_of_S[0:i]+'-')+copy_of_S[i:]
    return copy_of_S

def output_sequences(sequences, order) :
    aligned_sequences = []

    for i in order:
        aligned_sequences.append(sequences[i])

    for i in range(len(aligned_sequences)-1):
        A = aligned_sequences[i]
        B = aligned_sequences[i+1]
        score, alignedA, alignedB = get_alignment_score(A,B)
        gap_indices_for_A = find_gap_indices(A, alignedA)
        for j in range(i):
            S = aligned_sequences[j]
            newly_alinged_S = insert_gaps(S,gap_indices_for_A)
            aligned_sequences[j] = newly_alinged_S
        
        aligned_sequences[i] = alignedA
        aligned_sequences[i+1] = alignedB  

    return aligned_sequences

def getdist(data):
    out = []
    for i in range(len(data)):
        idx = len(data[i])
        for j in range(idx):
            if i != j:
                out.append(f"{i} {j} {data[i][j]}")
    return out

def score(sequences):
    t = len(sequences)
    k = len(sequences[0])
    score = 0
    for i in range(t):
        A = sequences[i]
        for j in range(i+1,t):
            B = sequences[j]
            for idx in range(k):
                if A[idx] != B[idx]:
                    score += 1
    return score

if __name__ == "__main__":
    x = []
    input_file = sys.argv[1]
    f = open(input_file, "r")
    seq = f.read().strip("\n").split("\n")
    tspdata = MSA_to_TSP(seq)
    newdata = getdist(tspdata)
    x, y = tabu_search(newdata)
    x = list(map(int, x))
    aligned_sequences = output_sequences(seq, x)
    for i in range(len(aligned_sequences)):
        print(aligned_sequences[i])
    print(f"Score: {score(aligned_sequences)}")


