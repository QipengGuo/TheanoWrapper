#the code may be quite ugly, but this version is using in now running experiments, the optimization will add in next version(such as speed up cross_entropy and remove abs in attention part)
import numpy as NP
import re 
from collections import OrderedDict
import nltk
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

class txt_data(object):

# an example for data in ptb_train, easy for generate segmentation mask
# n o i t w a s n ' t b l a c k m o n d a y ||| 2 2 3 3 5 6 

	def __init__(self, train_fname=None, test_fname=None, dict_fname=None, dataset_fnames=[], gen_dict=False, only_dict=True, DICT_SIZE=10000, raw_file=False):	
                self.only_dict = only_dict # only_dict means using exist dict, don't add new word when parsing file, must be true now
                self.train_fname=train_fname
                self.test_fname=test_fname
                self.raw_file=raw_file
                # c_dict, dictionary for character
                # Wdict, dictionary for word
                # num2c, index in dict to character
                # num2w, index in dict to word
                # w_dim, c_dim, are count of word and character
                if only_dict:
                    f = NP.load(dict_fname+'.npz')
                    self.Cdict = f['Cdict'].tolist()
                    self.Wdict = f['Wdict'].tolist()
                    self.num2c = f['num2c'].tolist()
                    self.num2w = f['num2w'].tolist()
                    self.w_dim = f['w_dim'].tolist()
                    self.c_dim = f['c_dim'].tolist()
                else:
    		    self.Cdict = {}
		    self.Wdict = {}
		    self.num2c = {}
                    self.num2w = {}
		    self.w_dim = 1
		    self.c_dim = 1
                self.Wcnt = OrderedDict() # counting the frequency of words
                self.Ccnt = OrderedDict()
                #w_dim, c_dim start from 1, actually it is useless, just keep the 0 to special case, because argmax(all zeros) == 0 

                if gen_dict:
                    fnames = dataset_fnames
                    for i in fnames:
                        self.read_txt_gen_dict(i)
                    print 'Before Reducing Dict Size', len(self.Wcnt.keys())
                    self.Wdict, self.num2w, self.w_dim = self.reduce_dict(self.Wcnt, DICT_SIZE=DICT_SIZE)
                    self.Cdict, self.num2c, self.c_dim = self.reduce_dict(self.Ccnt, DICT_SIZE=128)
                
                self.UWgram = NP.zeros((self.w_dim))
                self.UCgram = NP.zeros((self.c_dim))

                # follow index record the position which next batch start
                self.test_data_idx = 0
                self.train_data_idx = 0
                # the len_idx record which length group should be used
                self.test_len_idx = 0
                self.train_len_idx = 0
                self.idx_spliter = [50, 100, 150, 200, 250, 300] #split the sentence length in group to speed up, the 50 contains all sentences which length smaller than 50, the batch length will be 50
		self.MAX_TIME = 300 #one sentences length no more than 300 characters 
                #read from file and split into group
		self.train_char, self.train_segs, self.train_word, self.train_idx = self.read_txt(train_fname)
		self.test_char, self.test_segs, self.test_word, self.test_idx = self.read_txt(test_fname)
                self.train_idx_list = self.split_len(self.train_idx)
                self.test_idx_list = self.split_len(self.test_idx)

		self.test_size = len(self.test_char)
		self.train_size = len(self.train_char)
                print 'Data Size ', self.w_dim
                if gen_dict:
                    print 'Writing Dcit'
                    NP.savez(dict_fname, Wdict=self.Wdict, Cdict=self.Cdict, num2c=self.num2c, num2w=self.num2w, w_dim=self.w_dim, c_dim=self.c_dim)

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
        def get_batch(self, batch_size=32, test=False, only_word=False):
                char = self.test_char if test else self.train_char
                segs = self.test_segs if test else self.train_segs
                word = self.test_word if test else self.train_word
                idx_list = self.test_idx_list if test else self.train_idx_list
                data_idx = self.test_data_idx if test else self.train_data_idx
                len_idx = self.test_len_idx if test else self.train_len_idx

                data_idx += batch_size
                #shuffle the data when pass through the whole data once
                if data_idx>=len(char):
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
                char_max_len = self.idx_spliter[len_idx]

                # extract data by index, all the extracted sentences' length smaller than local MAX_TIME now, so the batch length is MAX_TIME
		char = char[idx]
                segs = segs[idx]
                word = word[idx]
                word_max_len = 0
                for i in word:
                    if len(i)>word_max_len:
                        word_max_len = len(i)
                batch_size = len(char)
                tchar = []
                tword = []
                # data is still the index , not one-hot, i using -1 to represent all zeros, the following things compelete the sentences to batch length
                for i in xrange(batch_size):
                    tchar.append((NP.concatenate((char[i], NP.zeros((char_max_len-len(char[i])))-1), axis=0)).tolist())
                    tword.append((NP.concatenate((word[i], NP.zeros((word_max_len-len(word[i])))-1), axis=0)).tolist())


                char = NP.asarray(tchar)
                word = NP.asarray(tword)
                if only_word:
                    return word
                # -1 means all zeros in one-hot
                # fit the data when the length is not enough for batch_size
		mask = NP.zeros((batch_size, char_max_len))
                word_idx1 = NP.zeros((batch_size, word_max_len-1))
                word_idx2 = NP.zeros((batch_size, word_max_len-1))
		for b in xrange(batch_size):
			pos = 0 # positions in character seqences
			for i in xrange(len(segs[b])-1):
				t = segs[b][i] #the length of this word 
				pos += t
				if pos>=char_max_len: #useless checking ...
					break
				 #stop before last word, because there no next word for last word
				mask[b][pos-1]=1 #mask equal one means the end of one word
                                word_idx1[b][i]=b
                                word_idx2[b][i]=pos-1

                #char_label = NP.maximum(0, char) # remove -1 flag
                #word_label = NP.maximuim(0, word) # remove -1 flag
                char_label = char
                word_label = word[:, 1:]
                # to one hot representation
                #char = NP.asarray([[list_to_nparray(x, self.c_dim) for x in batch] for batch in char])

		return char, char_label, mask, word_idx1, word_idx2, word
	
        # word to character index
        def W2CI(self, word):
                result = []
                for c in word:
                    if not self.Cdict.has_key(c):
                        self.Cdict[c]=0
                    result.append(self.Cdict[c])
                return result

        def read_txt(self, fname):
                f=open(fname, 'r')
                chars = []
                words = []
                segs = []
                lens = []
                slist = f.readlines()
                new_slist = []
                if self.raw_file:
                    for s in slist:
                        s = s.lower()
                        s = s.decode('utf-8')
                        if len(s)<2 or s[:4]=='<doc' or s[:5]=='</doc': #ignore wiki head
                            continue
                        ss = nltk.sent_tokenize(s)
                        for one_sent in ss:
                            if len(one_sent)<2:
                                continue
                            new_slist.append(one_sent)
                    slist = new_slist
                for s in slist:
                    s = s.lower()
                    if self.raw_file:
                        ws = nltk.word_tokenize(s)
                    else:
                        ws = re.split('\s', s)
                    ws = filter(lambda x:x!='' and x!=' ', ws)
                    if len(ws)<2:# if only one word, there are no next word for label
                        continue
                    one_segs = [] # one row segs
                    one_words = [] # one row words
                    one_chars = [] # one row chars
                    for w in ws:
                        if not self.Wdict.has_key(w):
                            self.Wdict[w]=0
                        one_segs.append(len(w))
                        one_words.append(self.Wdict[w])
                        one_chars.extend(self.W2CI(w)) #because it is adding list to list
                    chars.append(one_chars)
                    words.append(one_words)
                    segs.append(one_segs)
                    lens.append(len(one_chars))
                f.close()
                return NP.asarray(chars), NP.asarray(segs), NP.asarray(words), NP.asarray(lens)

        def read_txt_gen_dict(self, fname):
                f=open(fname, 'r')
                slist = f.readlines()
                for s in slist:
                    s = s.lower()
                    if self.raw_file:
                        s = s.decode('utf-8')
                        if s[:4]=='<doc' or s[:5]=='</doc': #ignore wiki head
                            continue
                        ws = nltk.word_tokenize(s)
                    else:
                        ws = re.split('\s', s)

                    for w in ws:
                        self.Wcnt[w] = self.Wcnt.get(w, 0) + 1
                        if self.Wcnt[w] == 1:
                            # only check character at first time 
                            for c in w:
                                self.Ccnt[c] = self.Ccnt.get(c, 0) + 1
                f.close()

        def reduce_dict(self, Xcnt, DICT_SIZE=10000):
                Xcnt = OrderedDict((sorted(Xcnt.items(), key = lambda x:x[1], reverse=True)))
                items = Xcnt.items()
                Xdict = {}
                num2X = {}
                for i in xrange(min(DICT_SIZE, len(Xcnt.keys()))):
                    Xdict[items[i][0]] = i+1 #start from 1, keep 0 to OOV
                    num2X[i+1]=items[i][0]
                X_dim = min(DICT_SIZE, len(Xcnt.keys()))+1
                return Xdict, num2X, X_dim
              
