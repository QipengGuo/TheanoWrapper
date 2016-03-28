import numpy as NP

def list_to_nparray(x, dim):
    y = NP.zeros((dim,), dtype=NP.int32)
    if x>=0:
        y[x]=1
    return y

def shuffle_union(a, b):
	p = NP.random.permutation(len(a))
	return a[p], b[p]

def shuffle(a):
	p = NP.random.permutation(len(a))
	return a[p]

class ptb_data(object):

	def __init__(self, train_fname='ptb_train', test_fname='ptb_valid', write_dict=False, only_dict=False):	
                self.only_dict = only_dict
                self.train_fname=train_fname
                self.test_fname=test_fname
                if only_dict:
                    f = NP.load('PTB_dict.npz')
                    self.c_dict = f['c_dict'].tolist()
                    self.Wdict = f['Wdict'].tolist()
                    self.num2c = f['num2c'].tolist()
                    self.num2w = f['num2w'].tolist()
                    self.w_dim = f['w_dim'].tolist()
                    self.c_dict_cnt = f['c_dict_cnt'].tolist()
                else:
    		    self.c_dict = {}
		    self.Wdict = {}
		    self.num2c = {}
                    self.num2w = {}
		    self.w_dim = 1
		    self.c_dict_cnt = 1

                self.pcnt=0
                self.test_data_idx = 0
                self.train_data_idx = 0
                self.test_len_idx = 0
                self.train_len_idx = 0
                self.idx_spliter = [50, 100, 150, 200, 250, 300]
		self.MAX_TIME = 300
		self.train_data, self.train_segs, self.train_idx = self.read_txt(train_fname)
		self.test_data, self.test_segs, self.test_idx = self.read_txt(test_fname)
                self.train_idx_list = self.split_len(self.train_idx)
                self.test_idx_list = self.split_len(self.test_idx)
		self.c_dim = self.c_dict_cnt
		self.test_size = len(self.test_data)
                print 'Data Size ', self.w_dim
                if write_dict:
                    print 'Writing Dcit'
                    NP.savez('PTB_dict', Wdict=self.Wdict, c_dict=self.c_dict, num2c=self.num2c, num2w=self.num2w, c_dict_cnt=self.c_dict_cnt, w_dim=self.w_dim)

        def split_len(self, idx):
                idx_list = []
                for j in self.idx_spliter:
                    idx_list.append([])

                for i in xrange(len(idx)):
                    for j in xrange(len(self.idx_spliter)):
                        if idx[i]<self.idx_spliter[j]:
                            idx_list[j].append(i)
                            break

                return idx_list

	def get_batch(self, batch_size=1, test=False):
		data = self.test_data if test else self.train_data
		segs = self.test_segs if test else self.train_segs
                idx_list = self.test_idx_list if test else self.train_idx_list
                data_idx = self.test_data_idx if test else self.train_data_idx
                len_idx = self.test_len_idx if test else self.train_len_idx
                data_idx += batch_size
                if data_idx>=len(data):
                    for i in xrange(len(idx_list)):
                        idx_list[i]=shuffle(idx_list[i])
                    data_idx = batch_size
                    len_idx = 0
                if data_idx>=len(idx_list[len_idx]):
                    len_idx+=1
                    while len_idx<len(self.idx_spliter) and len(idx_list[len_idx])==0:
                        len_idx+=1
                    if len_idx>=len(self.idx_spliter):
                        len_idx = 0
                    data_idx = batch_size
                if test:
                    self.test_len_idx = len_idx
                    self.test_data_idx = data_idx
                else:
                    self.train_len_idx = len_idx
                    self.train_data_idx = data_idx
                idx = NP.asarray(idx_list[len_idx][data_idx-batch_size:data_idx])
                MAX_TIME = self.idx_spliter[len_idx]
		data = data[idx]
                tdata = []
                for i in xrange(len(data)):
                    tdata.append((NP.concatenate((data[i], NP.zeros(MAX_TIME-len(data[i]))-1), axis=0)).tolist())
                data = NP.asarray(tdata)
		segs = segs[idx]
		words = NP.zeros_like(data)-1
                batch_size = len(data)
		mask = NP.zeros((batch_size, MAX_TIME))
		for b in xrange(batch_size):
			pos = 0
                        #print self.tostring(data[b])
			for i in xrange(len(segs[b])):
				t = segs[b][i]
				pos += t
				if pos>=MAX_TIME:
					break
				if (i<len(segs[b])-1):
				        mask[b][pos]=1
				        str1 = self.tostring(data[b][pos:pos+segs[b][i+1]])
                                        str2 = self.tostring(data[b][pos-segs[b][i]:pos])
                                        #print str2, '|||', str1
					words[b][pos]= self.Wdict[str1]
#                        if pos>=MAX_TIME:
#                                mask[b][-1] = 1
                data = NP.asarray([[list_to_nparray(x, self.c_dim) for x in batch] for batch in data])
		label = data
		words = NP.asarray([[list_to_nparray(x, self.w_dim) for x in batch] for batch in words])

		return data, label, mask, words
	def tostring(self, char_seq):
		str1 = ''
		for i in char_seq:
                        #print self.num2c[i]
                        if i not in self.num2c.keys():
                            print 'ERROR ', i
                        else:
			    str1 += self.num2c[i]
                if str1 == '':
                    return ' '
		return str1
		
	def read_txt(self, fname):
		f=open(fname, 'r')
		data = []
		segs = []
                idx = []
		slist = f.readlines()
		for s in slist:
			t = s.split('|||')
			if len(t)!=2:
				continue
                        #print self.pcnt
                        self.pcnt+=1
			one_row = []
			one_seg = []
			x, z = t
			tt = 0
                        xx = x.split(' ')
                        xx = filter(lambda x:x!='' and x!=' ', xx)
                        for c in xx:
				tt+=1
				if tt>=self.MAX_TIME:
					break
				if c not in self.c_dict.keys():
                                        if self.only_dict:
                                            self.c_dict[c]=0
                                        else:
					    self.c_dict[c]=self.c_dict_cnt
					    self.c_dict_cnt += 1
					    self.num2c[self.c_dict_cnt-1]=c
				one_row.append(self.c_dict[c])

			tt = 0
			for seg in z.split(' '):
				if not seg.isdigit():
					continue
				tt+=int(seg)
				if tt>=self.MAX_TIME:
					break
				one_seg.append(int(seg))

			last_pos = 0
                        str1 = self.tostring(one_row)
                        #print str1
			for ss in one_seg:
				str1 = self.tostring(one_row[last_pos:last_pos+ss])
				if (str1 not in self.Wdict.keys()):
                                        #print self.w_dim, str1
                                        if self.only_dict:
                                            self.Wdict[str1]=0
                                        else:
					    self.Wdict[str1]=self.w_dim
                                            self.num2w[self.w_dim] = str1
					    self.w_dim += 1
				last_pos = last_pos + ss

			data.append(one_row)
			segs.append(one_seg)
                        idx.append(len(one_row))
		f.close()
		return NP.asarray(data), NP.asarray(segs), NP.asarray(idx)
