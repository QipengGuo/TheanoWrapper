import json
import gzip
import os
import re

def Amazon_data:
	def __init__(self, apr_name, en_list_name):
		f = open(en_list_name, 'r')
		self.en_list = f.readlines()
		f.close()
		self.read_json(apr_name)
		self.idx = 0
	
	def get_batch(self, batch_size):
		st = self.idx
		ed = self.idx + batch_size 
		l = len(self.one_hot_data)
		if ed > l:
			return NP.concatenate((self.one_hot_data[st:], self.one_hot_data[:ed-l]), axis=0), NP.concatenate((self.one_hot_label[st:], self.one_hot_label[:ed-l]), axis=0)
		else:
			return NP.asarray(self.one_hot_data[st:ed]), NP.asarray(self.one_hot_label[st:ed])

	def json_parse(self, filename):
		g = gzip.open(filename, 'r')
		for l in g:
			yield eval(l)['reviewText']

	def read_json(self, filename):
		max_itmes = 5e4
		slist = self.json_parse(filename)
		cnt=0
		f = open('temp.txt', 'w')
		for s in slist:
			if cnt>max_items-1:
				break
			cnt+=1
			f.write(s+'\n')
		f.close()
		os.system('')

		f = open('temp_token.txt', 'r')
		vocab_idx = 1
		max_len = 0
		num_S = 0
		num_W = 0
		slist = f.readlines()
		for s in slist:
			num_S += 1
			words = re.split(' ', s)
			if max_len < len(words):
				max_len = len(words)
			for w in words:
				num_W += 1
				if w in self.en_list:
					if w not in vocab_map.keys():
						vocab_map[w]=vocab_idx
						vocab_idx += 1
					else:
						data[num_S][num_W]=vocab_map[w]

		f.close()
		self.one_hot_data = NP.zeros((len(slist),max_len,vocab_idx+1))
		self.one_hot_label = NP.zeros((len(slist),max_len,vocab_dix+1))
		for i in xrange(num_S):
			for j in xrange(num_W-1):
				self.one_hot_data[data[i][j]] = 1
				self.one_hot_label[data[i][j+1]] = 1 

