
import itertools
from collections import OrderedDict
str=''
list_final=[]
stri=[]
strings=[]
fir=[]
V=[]
E=[]
listf=[]
out=[]
class Graph:
	def __init__(self, nodes=None, edges=None):
		self.nodes, self.adj = [],{}
		if nodes != None:
			self.add_nodes_from(nodes)
		if edges != None:
			self.add_edges_from(edges)
	def __iter__(self):
		return iter(self.nodes)
	def __getitem__(self, x):
		return iter(self.adj[x])
	def add_node(self, n):
		if n not in self.nodes:
			self.nodes.append(n)
			self.adj[n] = []
	def add_nodes_from(self, i):
		for n in i:
			self.add_node(n)
	def add_edge(self, u, v):   # undirected unweighted graph
		self.adj[u] = self.adj.get(u, []) + [v]
		self.adj[v] = self.adj.get(v, []) + [u]
	def add_edges_from(self, i):
		for n in i:
			self.add_edge(*n)
	def number_of_edges(self):
		return sum(len(l) for _, l in self.adj.items()) // 2
#removes the duplicates and its last element
def rem(strings):
	if(len(strings[-1])!=len(strings[0])):
		strings.pop()
	reads = list(OrderedDict.fromkeys(strings))
	return reads
def hierholzer(g):
	# for u in g:
		# if len(list(g[u])) % 2 == 1:
		# 	return None
	start = next(g.__iter__())  # choose the start vertex to be the first vertex in the graph
	circuit = [start]           # can use a linked list for better performance here
	traversed = {}
	ptr = 0
	while len(traversed) // 2 < g.number_of_edges() and ptr < len(circuit):
		subpath = []            # vertices on subpath
		dfs(g, circuit[ptr], circuit[ptr], subpath, traversed)
		if len(subpath) != 0:   # insert subpath vertices into circuit
			circuit = list(itertools.chain(circuit[:ptr+1], subpath, circuit[ptr+1:]))
		ptr += 1
	return circuit
def dfs(g, u, root, subpath, traversed):
	for v in g[u]:
		if (u,v) not in traversed or (v,u) not in traversed:
			traversed[(u,v)] = traversed[(v,u)] = True
			subpath.append(v)
			if v == root:
				return
			else:
				dfs(g, v, root, subpath, traversed)
######
for i in open("dna.txt","r"):
	strl=i.split()
	stri=stri+strl
for line in stri:
	str=str+line
k=int(input("Enter value of k: "))
for m in range(0,len(str),k):
	strings.append(str[m:m+k])
reads=rem(strings)
#make vertices and edges
for st in reads:
	fir.append(st[:-1])
	fir.append(st[1:])
	t=(st[:-1],)+(st[1:],)
	E.append(t)
V=rem(fir)
g = Graph(nodes=V, edges=E)
out=hierholzer(g)
first=out[0]
for i in out:
	list_final.append(i.rstrip())
finalstring=""
finalstring=finalstring+first
del list_final[0]
for i in list_final:
	finalstring=finalstring+i[len(first)-1]
listf.append(finalstring)
print("Eulerian Path :")
print(listf)