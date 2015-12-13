import json
import gzip
import os
import re
import numpy as NP
class Amazon_data(object):
	def __init__(self, apr_name, en_list_name, cache=False):
                if cache and os.path.exists('token.npz'):
                        f = NP.load('token.npz')
                        self.emb_data = f['data']
                        self.emb_label = f['label']
                else:
		        f = open(en_list_name, 'r')
		        self.en_list = f.readlines()
		        f.close()
		        self.read_json(apr_name)
                        NP.savez('token.npz', data=self.emb_data, label=self.emb_label)
                self.idx = 0
                self.max_vocab = 20000
                self.t_idx = 0
                self.vocab_size = min(self.max_vocab, NP.max(self.emb_data))
                self.test_size = 5000
                self.test_emb_data = self.emb_data[-self.test_size:]
                self.test_emb_label = self.emb_label[-self.test_size:]
                self.emb_data = self.emb_data[:-self.test_size]
                self.emb_label = self.emb_label[:-self.test_size]
	
	def _get_batch(self, batch_size, test=False):
                if test:
                        st = self.t_idx
                        ed = self.t_idx + batch_size
                        l = len(self.test_emb_data)
                        self.t_idx = ed % l
                else:
		        st = self.idx
		        ed = self.idx + batch_size 
		        l = len(self.emb_data)
                        self.idx = ed % l
		if ed > l:
                        if test:         
			        return NP.concatenate((self.test_emb_data[st:], self.test_emb_data[:ed-l]), axis=0), NP.concatenate((self.test_emb_label[st:], self.test_emb_label[:ed-l]), axis=0)
                        else:
			        return NP.concatenate((self.emb_data[st:], self.emb_data[:ed-l]), axis=0), NP.concatenate((self.emb_label[st:], self.emb_label[:ed-l]), axis=0)
		else:
                        if test:
                                return NP.asarray(self.test_emb_data[st:ed]), NP.asarray(self.test_emb_label[st:ed])
                        else:
			        return NP.asarray(self.emb_data[st:ed]), NP.asarray(self.emb_label[st:ed])

        def get_batch(self, batch_size, test=False):
                data, label = self._get_batch(batch_size, test)
                one_hot_data = NP.zeros((batch_size, len(data[0]), self.vocab_size))
                one_hot_label = NP.zeros((batch_size, len(data[0]), self.vocab_size))
                for i in xrange(batch_size):
                    for j in xrange(len(data[0])):
                        one_hot_data[i][j][min(self.max_vocab, data[i][j])]=1.0
                        one_hot_label[i][j][min(label[i][j], self.max_vocab)]=1.0
                return one_hot_data, one_hot_label

	def json_parse(self, filename):
		g = gzip.open(filename, 'r')
		for l in g:
			yield eval(l)['reviewText']

	def read_json(self, filename):
		max_items = 1e4
                max_time = 200
		slist = self.json_parse(filename)
		cnt=0
		f = open('temp.txt', 'w')
		for s in slist:
			if cnt>max_items-1:
				break
			cnt+=1
			f.write(s+'\n')
		f.close()
		os.system('perl tokenizer.perl -threads 8 < temp.txt > temp_token.txt')

		f = open('temp_token.txt', 'r')
                vocab_map = {}
		vocab_idx = 1
		max_len = 0
		num_S = -1
		num_W = -1
                data = NP.zeros((max_items+1, max_time))
		slist = f.readlines()
		for s in slist:
			num_S += 1
                        #print num_S, 
			words = re.split(' |\n', s)
			if max_len < len(words):
				max_len = len(words)
                        num_W = 0
                        print 'Sent ', num_S
			for w in words:
                                if w in ['', '\n', ' ', '\t', '\r']:
                                        continue
				num_W += 1
                                if num_W >= max_time:
                                    break
                                flag =w.lower() in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ',', '.', ':', '!', '#', '%', ';', '?', '(', ')'] or w.lower()+'\n' in self.en_list
                                #print w, ' Has = ', flag
                                if flag:
					if w not in vocab_map.keys():
						vocab_map[w]=vocab_idx
						vocab_idx += 1
                                                data[num_S][num_W]=vocab_map[w]
					else:
						data[num_S][num_W]=vocab_map[w]

		f.close()
                max_len = min(max_time, max_len)
                self.emb_data = data[:,:-1]
                self.emb_label = data[:,1:]
                self.vocab_size = vocab_idx + 1

