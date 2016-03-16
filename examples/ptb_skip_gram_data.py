import numpy as NP
import re
def list_to_nparray(x, dim):
    y = NP.zeros((dim,), dtype=NP.int32)
    if x>=0:
        y[x]=1
    return y

def shuffle_union(a, b, c):
	p = NP.random.permutation(len(a))
	return a[p], b[p], c[p]

def shuffle(a):
	p = NP.random.permutation(len(a))
	return a[p]

class ptb_skip_gram(object):

	def __init__(self, train_fname='ptb.train.txt', test_fname='ptb.test.txt'):	
                self.rng = NP.random
		self.train_fname=train_fname
                self.test_fname=test_fname                        
		self.Wdict={}
		self.Wcnt={}
		self.wcnt=0
		self.MAX_TIME = 50
		self.MAX_NEG = 50
		self.train_dataA = []
		self.train_dataB = []
		self.train_label = []
		self.test_dataA = []
		self.test_dataB = []
		self.test_label = []
		self.count_txt(train_fname)
		self.count_txt(test_fname)
		self.word_num=0
		for i in self.Wdict.keys():
			self.word_num+=self.Wdict[i]
		self.read_txt(train_fname)
		self.read_txt(test_fname)
                self.read_txt(train_fname)
                self.read_txt(test_fname)
                #self.new_neg = self.find_neg_words(500)
		self.test_size = len(self.test_dataA)

        def get_dict(self, fname =''):
            self.wdict = []
            f=open(fname,'r')
            for s in f.readlines():
                for w in s.split(' '):
                    if w==' ' or w=='' or w=='\n':
                        continue
                    if w not in self.wdict:
                        self.wdict.append(w)
            f.close()

        def get_word_dict(self, fname = 'ptb_dict', train_fname='ptb.train.txt', test_fname='ptb.test.txt'):
                self.get_dict(train_fname)
                self.get_dict(test_fname)
                ws = []
                for s in self.wdict:
                    w = []
                    for c in s:
                        if c==' ':
                            continue
                        if c not in self.c_dict.keys():
                            print c
                        w.append(self.c_dict[c])
                    w.extend([-1]*(self.MAX_TIME-len(w)))
                    ws.append(w)
                data = NP.asarray([[list_to_nparray(x, self.c_dim) for x in batch] for batch in ws])
                return data, self.wdict


                
	def get_batch(self, batch_size=1, test=False):
		dataA = self.test_dataA if test else self.train_dataA
		dataB = self.test_dataB if test else self.train_dataB
		label = self.test_label if test else self.train_label
		dataA = NP.asarray(dataA)
		dataB = NP.asarray(dataB)
		label = NP.asarray(label)
		dataA, dataB, label = shuffle_union(dataA, dataB, label)
		dataA = dataA[:batch_size]
		dataB = dataB[:batch_size]
		label = label[:batch_size]
                #if self.rng.random()<0.1:
                #    self.new_neg = shuffle(self.new_neg)
                #    dataA = 1.0* (dataA==1).astype('int')*dataA+(dataA==0).astype*self.new_neg[:batch_size]
                #    if self.rng.random()<0.3:
                #        self.new_neg = self.find_neg_words(500)
		return dataA, dataB, label
		
	def count_txt(self, fname):
		f=open(fname, 'r')
		for s in f.readlines():
			ws = re.split('\s|(?<!\d)[,.](?!\d)', s)
			ws = filter(lambda a: a!='', ws)
			for w in ws:
				if w==' ' or w=='' or w=='\n':
					continue
				if w not in self.Wdict.keys():
					self.Wdict[w]=0
					self.Wcnt[w]=self.wcnt
					self.wcnt+=1
				self.Wdict[w]+=1
		f.close()

	def find_neg_words(self, size=50):
		cnt = []
		words = []
		for i in self.Wdict.keys():
			t = 1.0*(NP.sqrt(1.0*self.Wdict[i]/size*self.word_num)+1)*(size*self.word_num) / self.Wdict[i]
			if (t<self.rng.random()):
				continue
			cnt.append(self.Wdict[i])
			words.append(i)
		cnt = NP.asarray(cnt)
		words = NP.asarray(words)
		idx = NP.argsort(cnt)[::-1]
		words = words[idx]
		return words[:size]

	def read_txt(self, fname):
		f=open(fname, 'r')
		cnt = 0
		negs = self.find_neg_words(self.MAX_NEG)
		for s in f.readlines():
			cnt+=1
			if cnt % 5 == 3:
				negs = self.find_neg_words(self.MAX_NEG)
			ws = re.split('\s|(?<!\d)[,.](?!\d)', s)
			ws = filter(lambda a:a!='', ws)
			for i in xrange(len(ws)):
				w = ws[i]
				if w==' ' or w=='' or w=='\n':
					continue
				self.train_dataA.append(self.Wcnt[w])
				t=self.rng.randint(max(0, i-2), min(i+2, len(ws)))
				t = ws[t]
				self.train_dataB.append(self.Wcnt[t])
				self.train_label.append(1.0)
                                negs = shuffle(negs)
                                for k in negs[:20]:
                                        if k == w:
                                            continue
					self.train_dataA.append(self.Wcnt[k])
					self.train_dataB.append(self.Wcnt[t])
					self.train_label.append(-1.0)
					
				
		f.close()


