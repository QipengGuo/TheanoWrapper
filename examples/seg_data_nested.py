import os
import os.path
import string
from parse import *
import numpy as NP
import h5py
class Seg_Data(object):
	def __init__(self, path=""):
		self.idx = 0
                self.test_idx = 0
		self.max_len = 50
                self.storke = []
                self.test_storke = []
		self.label_st = []
		self.label_ed = []
		self.label_tag = []
                self.test_X = []
                self.test_label_st = []
                self.test_label_ed = []
                self.test_label_tag = []
		self.X = []
		self.C_vocab = 67
                
                cnt = 0 
                for root, dirs, files in os.walk("./SEG_DATA/"):
                        for d in dirs:
                                for rr, dd, ff in os.walk(os.path.join(root, d)):
                                    for f in ff:
                                            cnt+=1 
                                            #if len(self.test_X)>50:
                                            #    break
                                            #if cnt>=50:
                                            #        break
                                            self.load(os.path.join(rr, f))
                                            print 'Load ', f
           
                NP.savez('SEG_DATA.npz', test_X=self.test_X, test_label_st=self.test_label_st, test_label_ed=self.test_label_ed, test_label_tag=self.test_label_tag, X=self.X, label_st=self.label_st, label_ed=self.label_ed, label_tag=self.label_tag)
		#self.load('s001w53.dat')

        def decode(self, dec_tag, dec_st, label_seq):
                label_seq = [self.code_C(i) for i in label_seq]
                ed = len(dec_st)-1
                tag_seq = []
                dur_seq = []
                while ed>0:
                    st = dec_st[ed]-1
                    tag = dec_tag[ed]
                    dur_seq.append((max(st, 0), ed))
                    tag_seq.append(self.code_C(tag))
                    ed = st
                return tag_seq[::-1], dur_seq[::-1], label_seq

	def get_sample(self, test=False):
                if test:
                        X = self.test_X[self.test_idx]
                        storke = self.test_storke[self.test_idx]
                        label_st = self.test_label_st[self.test_idx]
                        label_ed = self.test_label_ed[self.test_idx]
                        label_tag = self.test_label_tag[self.test_idx]
                        self.test_idx+=1
                        if self.test_idx>=len(self.test_X):
                                self.test_idx = 0
                else:
		        X = self.X[self.idx]
                        storke = self.storke[self.idx]
		        label_st = self.label_st[self.idx]
		        label_ed = self.label_ed[self.idx]
		        label_tag = self.label_tag[self.idx]
                        self.idx+=1
                        if self.idx>=len(self.X):
                                self.idx = 0
		all_tag = []
		
                for k in xrange(self.C_vocab):
                    all_tag.append(k)

                match_ref = NP.zeros((len(storke), len(storke), self.C_vocab))
                for i in xrange(len(label_st)):
                    match_ref[label_st[i], label_ed[i], label_tag[i]]=1.0
                return X[:,NP.newaxis,:], storke, NP.asarray(all_tag), match_ref, label_tag

	def scmp(self, a, b):
		l = min(len(a), len(b))
		return a[:l] == b[:l]	

	def C_code(self, c):
		if c>='a' and c<='z':
			code=ord(c)-97+0
		if c>='0' and c<='9':
			code=ord(c)-48+26
		if c>='A' and c<='Z':
			code=ord(c)-65+36	
		if c=='!':
			code=63
		if c=='*':
			code=64
		if c=='&':
			code=65
		if c=='+':
			code=66
		return code

        def code_C(self, code):
                code = int(code)
                if code<26:
                    return chr(97+code)
                if code<36:
                    return chr(48+code-26)
                if code<63:
                    return chr(65+code-36)
                if code==63:
                    return '!'
                if code==64:
                    return '*'
                if code==65:
                    return '&'
                if code==66:
                    return '+'
                return None

	def load(self, fname):
		max_len =self.max_len
		C_vocab = self.C_vocab
		f=open(fname, 'r')
		slist = f.readlines()
		label = NP.zeros((max_len, ))
		Pos_X = NP.zeros((2000, ))
		Pos_Y = NP.zeros((2000, ))
		tick = 0
		pen_seg = []
		X_dim = 0
		Y_dim = 0
		label_st = []
		label_ed = []
		label_tag = []
		all_st = []
		all_ed = []
		all_tag = []
		X = []
                train_flag = True
		for s in slist:
                        if self.scmp(s, ' Group: Training'):
                                train_flag = True
                        if self.scmp(s, ' Group: Development'):
                                train_flag = False
			if self.scmp(s, '.SEGMENT LETTER '):
				t1=parse(".SEGMENT LETTER {:d} ? \"{}\"", s)
				t2=parse(".SEGMENT LETTER {:d}-{:d} ? \"{}\"", s)
				if t2 is not None:
					n1 = t2[0]
					n2 = t2[1]
					c = t2[2]
					label[n1:n2+1] = self.C_code(c)
					continue
				if t1 is not None:
					n = t1[0]
					c = t1[1]		
					label[n] = self.C_code(c)
					continue

			if self.scmp(s, '.PEN_DOWN'):
				#print s, tick
				pen_seg.append(tick)
			t = parse(".X_DIM {:d}", s)
			if t is not None:
				X_dim = t[0]
			t = parse(".Y_DIM {:d}", s)
			if t is not None:
				Y_dim = t[0]
			t = parse("{:d} {:d} {:d}", s)
			if t is not None:
				Pos_X[tick] = t[0]
				Pos_Y[tick] = t[1]
				tick+=1
		f.close()
		pen_seg.append(tick)
                if len(pen_seg)<3:
                    return
		st = 0
		for i in xrange(1, len(label)):
                        if i>=len(pen_seg):
                                break
			if label[i]!=label[i-1]:
				label_st.append(st)
				label_ed.append(i)
				label_tag.append(label[i-1])
				st = i
                if len(label_ed) < 2:
                    print 'out'
                    return 
		Pos_X = NP.asarray(Pos_X)
		Pos_Y = NP.asarray(Pos_Y)
		Pos_X = 1.0 * (Pos_X - X_dim/2.0)/X_dim
		Pos_Y = 1.0 * (Pos_Y - Y_dim/2.0)/Y_dim
		for i in xrange(tick):
			x=NP.zeros((4,))
			x[0]=Pos_X[i]
			x[1]=Pos_Y[i]
			if i>0:
				x[2]=Pos_X[i]-Pos_X[i-1]
				x[3]=Pos_Y[i]-Pos_Y[i-1]
			X.append(x)
                if label_ed[-1]!=len(pen_seg)-1:
                    print label_ed[-1], len(pen_seg)
                    return
                if train_flag:
		        self.X.append(NP.asarray(X))
                        self.storke.append(NP.asarray(pen_seg))
		        self.label_st.append(NP.asarray(label_st))
		        self.label_ed.append(NP.asarray(label_ed))
		        self.label_tag.append(NP.asarray(label_tag))
                else:
                        self.test_X.append(NP.asarray(X))
                        self.test_storke.append(NP.asarray(pen_seg))
		        self.test_label_st.append(NP.asarray(label_st))
		        self.test_label_ed.append(NP.asarray(label_ed))
		        self.test_label_tag.append(NP.asarray(label_tag))
