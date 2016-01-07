import numpy as NP
from apr_data import *

f = NP.load('att_rnn_test_show.npz')
label = f['label']
predict = f['predict']
apr = Amazon_data('', '', cache=True)
vocab_map = (apr.vocab_map).tolist()
re_map = [None]*(len(vocab_map.keys())+2)

for i in vocab_map.keys():
    re_map[vocab_map[i]]=i

re_map[1]='<UNK>'
re_map[0]='<EOL>'
label = NP.argmax(label, axis=-1)
#predict = NP.argmax(predict, axis=-1)
f = open('apr_predict.txt', 'w')
for i in xrange(len(label)):
    f.write('\n\n\nSamples = '+str(i)+'\n\n\n')
    for j in xrange(len(label[0])):
        predict[i][j] /= NP.sum(predict[i][j])
        t = NP.random.choice(len(predict[i][j]), p=predict[i][j])
        if label[i][j]>0:
            f.write(re_map[t]+' ')
f.close()
f = open('apr_gt.txt', 'w')
for i in xrange(len(label)):
    f.write('\n\n\nSamples = '+str(i)+'\n\n\n')
    for j in xrange(len(label[0])):
        t = label[i][j]
        if t>0:
            f.write(re_map[t]+' ')
f.close()
print 'finish'
