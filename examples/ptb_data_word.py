#the code may be quite ugly, but this version is using in now running experiments, the optimization will add in next version(such as speed up cross_entropy and remove abs in attention part)
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

# an example for data in ptb_train, easy for generate segmentation mask
# n o i t w a s n ' t b l a c k m o n d a y ||| 2 2 3 3 5 6 

	def __init__(self, train_fname='ptb_train', test_fname='ptb_valid', write_dict=False, only_dict=False):	
                self.only_dict = only_dict # only_dict means using exist dict, don't add new word when parsing file
                self.train_fname=train_fname
                self.test_fname=test_fname
                # c_dict, dictionary for character
                # Wdict, dictionary for word
                # num2c, index in dict to character
                # num2w, index in dict to word
                # w_dim, c_dim, c_dict_cnt are count of word and character, last one is redundant
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
                #w_dim, c_dict_cnt start from 1, actually it is useless, just keep the 0 to special case
                self.pcnt=0 #debug count, useless
                # follow index record the position which next batch start
                self.test_data_idx = 0
                self.train_data_idx = 0
                # the len_idx record which length group should be used
                self.test_len_idx = 0
                self.train_len_idx = 0
                self.idx_spliter = [50, 100, 150, 200, 250, 300] #split the sentence length in group to speed up, the 50 contains all sentences which length smaller than 50, the batch length will be 50
		self.MAX_TIME = 300 #one sentences length no more than 300 characters 
                #read from file and split into group
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

        #split the sample by length, only record index
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

        # the data structure may be stupid, ... anyway, i add these function step by step, i will reconstruct the code after checking the correctness
	def get_batch(self, batch_size=1, test=False):
		data = self.test_data if test else self.train_data
		segs = self.test_segs if test else self.train_segs
                idx_list = self.test_idx_list if test else self.train_idx_list
                data_idx = self.test_data_idx if test else self.train_data_idx
                len_idx = self.test_len_idx if test else self.train_len_idx
                data_idx += batch_size
                #shuffle the data when pass through the whole data once
                if data_idx>=len(data):
                    for i in xrange(len(idx_list)):
                        idx_list[i]=shuffle(idx_list[i])
                    data_idx = batch_size
                    len_idx = 0
                #enter next length group
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
                # get index from group
                idx = NP.asarray(idx_list[len_idx][data_idx-batch_size:data_idx]) 
                MAX_TIME = self.idx_spliter[len_idx]
                # extract data by index, all the extracted sentences' length smaller than local MAX_TIME now, so the batch length is MAX_TIME
		data = data[idx]
                tdata = []
                # data is still the index , not one-hot, i using -1 to represent all zeros, the following things compelete the sentences to batch length
                for i in xrange(len(data)):
                    tdata.append((NP.concatenate((data[i], NP.zeros(MAX_TIME-len(data[i]))-1), axis=0)).tolist())

                data = NP.asarray(tdata)
		segs = segs[idx]
                # -1 means all zeros in one-hot
		words = NP.zeros_like(data)-1
                # fitthe data when the length is not enough for batch_size
                batch_size = len(data)
		mask = NP.zeros((batch_size, MAX_TIME))
		for b in xrange(batch_size):
			pos = 0 # positions in character seqences
                        #print self.tostring(data[b])
			for i in xrange(len(segs[b])):
				t = segs[b][i] #the length of this word 
				pos += t
				if pos>=MAX_TIME: #useless checking ...
					break
				if (i<len(segs[b])-1): #stop before last word, because there no next word for last word
				        mask[b][pos]=1 #mask equal one means the end of one word
                                        #str1 is the next word, str2 is the current word
				        str1 = self.tostring(data[b][pos:pos+segs[b][i+1]]) 
                                        str2 = self.tostring(data[b][pos-segs[b][i]:pos])
                                        #print str2, '|||', str1
					words[b][pos]= self.Wdict[str1] # string to index
#                        if pos>=MAX_TIME:
#                                mask[b][-1] = 1
                # to one hot representation
                data = NP.asarray([[list_to_nparray(x, self.c_dim) for x in batch] for batch in data])
		label = data
		words = NP.asarray([[list_to_nparray(x, self.w_dim) for x in batch] for batch in words])

		return data, label, mask, words

        # convert character sequences to word string
	def tostring(self, char_seq):
		str1 = ''
		for i in char_seq:
                        #print self.num2c[i]
                        if i not in self.num2c.keys():
                            print 'ERROR ', i
                        else:
			    str1 += self.num2c[i]
                if str1 == '': # only for debug, space and null are not in our dict
                    return ' '
		return str1
		
	def read_txt(self, fname):
		f=open(fname, 'r')
		data = []
		segs = []
                idx = []
		slist = f.readlines()
		for s in slist:
			t = s.split('|||') # see the data format at the top
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
                        for c in xx: # add character 
				tt+=1
				if tt>=self.MAX_TIME:
					break
				if c not in self.c_dict.keys():
                                        if self.only_dict:
                                            self.c_dict[c]=0
                                        else:
					    self.c_dict[c]=self.c_dict_cnt
					    self.c_dict_cnt += 1
					    self.num2c[self.c_dict_cnt-1]=c # so ugly...
				one_row.append(self.c_dict[c])

			tt = 0
			for seg in z.split(' '): # add segmentation
				if not seg.isdigit():
					continue
				tt+=int(seg)
				if tt>=self.MAX_TIME:
					break
				one_seg.append(int(seg))

			last_pos = 0
                        str1 = self.tostring(one_row) # debug
                        #print str1
			for ss in one_seg:
				str1 = self.tostring(one_row[last_pos:last_pos+ss]) # add words 
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
