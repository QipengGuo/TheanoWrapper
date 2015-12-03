import numpy as np
def list_to_nparray(x, dim):
    y = np.zeros((dim,), dtype=np.int32)
    y[x]=1
    return y

def read_txt(filename):
	f=open(filename)
	slist=f.readlines()
	X=np.zeros((len(slist), 500, 49), dtype=np.int32)
	Y=np.zeros((len(slist), 500, 49), dtype=np.int32)
	sen_num=-1
	skip_num=0
	for s in slist:
		sen_num+=1
		num=0
		X[sen_num][num]=27
		for c in s:
			if skip_num>0:
				skip_num=skip_num-1
			else:
				inum=0
				if c=='<':
					skip_num=4
					inum=1
					code=29
				if c>='a' and c<='z':
					inum=1
					code=ord(c)-97+1
				if c>='0' and c<='9':
					inum=1
					code=ord(c)-48+39
				if c=='\'':
					inum=1
					code=30
				if c=='-':
					inum=1
					code=31
				if c=='.':
					inum=1
					code=32
				if c=='\\':
					inum=1
					code=33
				if c=='/':
					inum=1
					code=34
				if c=='&':
					inum=1
					code=35
				if c=='$':
					inum=1
					code=36
				if c==' ':
					inum=1
					code=37
				if c=='N':
					inum=1
					code=38
				if inum>0:
					num=num+inum
					X[sen_num][num]=list_to_nparray(code, 49)
					Y[sen_num][num-1]=list_to_nparray(code, 49)
				#else:
			#		print ord(c)
		X[sen_num][num+1]=list_to_nparray(28, 49)
		Y[sen_num][num]=list_to_nparray(28, 49)
	f.close()
	return X, Y