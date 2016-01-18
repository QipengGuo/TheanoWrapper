import os
import string
from parse import *
import numpy as NP
class Seg_Data(object):
	def __init__(self, path=""):
		self.idx = 0
		self.max_len = 50
		self.label_st = []
		self.label_ed = []
		self.label_tag = []
		self.X = []
		self.C_vocab = 67
		self.t = self.load('s001w53.dat')

	def get_sample(self):
		X = self.X[self.idx]
		label_st = self.label_st[self.idx]
		label_ed = self.label_ed[self.idx]
		label_tag = self.label_tag[self.idx]
		self.idx+=1
		all_st = []
		all_ed = []
		all_tag = []
                all_st_notag = []
                all_ed_notag = []
		if self.idx==len(self.X):
			self.idx = 0
		
                for j in xrange(1, len(X)):
                    for i in xrange(max(0, j-1-10), j):
                        for k in xrange(self.C_vocab):
                            all_st.append(i)
                            all_ed.append(j)
                            all_tag.append(k)

		for i in xrange(len(X)):
			for j in xrange(i+1, min(i+1+10, len(X))):
                                all_st_notag.append(i)
                                all_ed_notag.append(j)
                return X[:,NP.newaxis,:], NP.asarray(all_st), NP.asarray(all_ed), NP.asarray(all_st_notag), NP.asarray(all_ed_notag), NP.asarray(all_tag), label_st, label_ed, label_tag

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

	def load(self, fname):
		max_len =self.max_len
		C_vocab = self.C_vocab
		f=open(fname, 'r')
		slist = f.readlines()
		label = NP.zeros((max_len, ))
		Pos_X = NP.zeros((1000, ))
		Pos_Y = NP.zeros((1000, ))
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
		for s in slist:
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
				print s, tick
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
		st = 0
		for i in xrange(1, len(label)):
			if label[i]!=label[i-1]:
				label_st.append(pen_seg[st])
				label_ed.append(pen_seg[i])
				label_tag.append(label[i-1])
				st = i
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
		self.X.append(NP.asarray(X))
		self.label_st.append(NP.asarray(label_st))
		self.label_ed.append(NP.asarray(label_ed))
		self.label_tag.append(NP.asarray(label_tag))
