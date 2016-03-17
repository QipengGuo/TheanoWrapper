import numpy as NP

def list_to_nparray(x, dim):
    y = NP.zeros((dim,), dtype=NP.int32)
    if x>=0:
        y[x]=1
    return y

def shuffle_union(a, b):
	p = NP.random.permutation(len(a))
	return a[p], b[p]

class ptb_data(object):

	def __init__(self, train_fname='ptb_train', test_fname='ptb_test'):	
                self.train_fname=train_fname
                self.test_fname=test_fname
		self.c_dict = {}
		self.Wdict = {}
		self.num2c = {}
		self.w_dim = 0
		self.c_dict_cnt = 0
		self.MAX_TIME = 50
		self.train_data, self.train_segs = self.read_txt(train_fname)
		self.test_data, self.test_segs = self.read_txt(test_fname)
		self.c_dim = self.c_dict_cnt
		self.test_size = len(self.test_data)

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

        def get_word_dict(self, fname = 'ptb_dict'):
                self.get_dict(self.train_fname)
                self.get_dict(self.test_fname)
                ws = []
                for s in self.wdict:
                    w = []
                    for c in s:
                        if c==' ':
                            continue
                        w.append(self.c_dict[c])
                        w.extend([-1]*(self.MAX_TIME-len(w)))
                    ws.append(w)
                data = NP.asarray([[list_to_nparray(x, self.c_dim) for x in batch] for batch in ws])
                return data, slist


                
	def get_batch(self, batch_size=1, test=False):
		data = self.test_data if test else self.train_data
		segs = self.test_segs if test else self.train_segs
		data = NP.asarray(data)
		segs = NP.asarray(segs)
		data, segs = shuffle_union(data, segs)
		data = data[:batch_size]
		segs = segs[:batch_size]
		words = NP.zeros_like(data)
		data = NP.asarray([[list_to_nparray(x, self.c_dim) for x in batch] for batch in data])
		label = data
		mask = NP.zeros((batch_size, self.MAX_TIME))
		for b in xrange(batch_size):
			pos = 0 
			for i in xrange(len(segs[b])):
				t = segs[b][i]
				pos += t
				if pos>=self.MAX_TIME:
					break
				mask[b][pos]=1
				if (i<len(segs[b])-1):
					str1 = self.tostring(data[b][pos:pos+segs[b][i+1]])
					words[b][pos]= self.Wdict[str1]
#			if pos<self.MAX_TIME-1:
#				mask[b][pos+1]=-1
                        if pos>=self.MAX_TIME:
                                mask[b][-1] = 1
		words = NP.asarray([[list_to_nparray(x, self.w_dim) for x in batch] for batch in words])

		return data, label, mask, words
	def tostring(self, char_seq):
		str1 = ''
		for i in char_seq:
			str1 += self.num2c[i]
		return str1
		
	def read_txt(self, fname):
		f=open(fname, 'r')
		data = []
		segs = []
		slist = f.readlines()
		for s in slist:
			t = s.split('|||')
			if len(t)!=2:
				continue
			one_row = []
			one_seg = []
			x, z = t
			tt = 0
			for c in x.split(' '):
				if c == '' or c == ' ':
					continue 
				tt+=1
				if tt>=self.MAX_TIME:
					break
				if c not in self.c_dict.keys():
					self.c_dict[c]=self.c_dict_cnt
					self.c_dict_cnt += 1
					self.num2c[self.c_dict_cnt-1]=c
				one_row.append(self.c_dict[c])

			tt = 0
			for seg in z.split(' '):
				if not seg.isdigit():
					continue
				tt+=1
				if tt>=self.MAX_TIME:
					break
				one_seg.append(int(seg))

			last_pos = 0
			for ss in seg:
				str1 = self.tostring(one_row[last_pos:last_pos+ss+1])
				if str1 not in self.Wdict.keys():
					self.Wdict[str1]=self.w_dim
					self.w_dim += 1
					last_pos = last_pos + ss

			one_row.extend([-1]*(self.MAX_TIME-len(one_row)))
			data.append(one_row)
			segs.append(one_seg)
		f.close()
		return data, segs
