import numpy as np
import h5py
import scipy.misc as misc
import os

class BouncingMNIST(object):
    def __init__(self, batch_size, image_size, dataset_name, target_name, scale_range=0, run_flag='',
                 clutter_size_min = 5, clutter_size_max = 10, num_clutters = 20, face_intensity_min = 64, 
                 face_intensity_max = 255, vel=1, buff=True, with_clutters=True, clutter_set='', **kwargs):
        self.batch_size_ = batch_size
        self.image_size_ = image_size
        self.scale_range = scale_range
        self.buff = buff
        self.digit_size_ = 28
        f = h5py.File('mnist.h5')
        self.data_ = np.asarray(f[dataset_name].value.reshape(-1, 28, 28))
        self.label_ = np.asarray(f[target_name].value)
        if run_flag=='train':
            idx=np.where(self.label_<5)[0]
            self.data_=self.data_[idx]
        if run_flag=='test':
            idx=np.where(self.label_>4)[0]
            self.data_=self.data_[idx]
        f.close()
        self.dataset_size_ = 10000  # Size is relevant only for val/test sets.
        self.indices_ = np.arange(self.data_.shape[0])
        self.row_ = 0
        self.clutter_size_min_ = clutter_size_min
        self.clutter_size_max_ = clutter_size_max
        self.num_clutters_ = num_clutters
        self.face_intensity_min = face_intensity_min
        self.face_intensity_max = face_intensity_max
        self.vel_scale = vel
        np.random.shuffle(self.indices_)
        self.num_clutterPack = 100
        self.clutter_set = clutter_set
        self.clutterpack_exists=  os.path.exists('ClutterPackLarge'+clutter_set+'.hdf5')
        if not self.clutterpack_exists:
            self.InitClutterPack()
        f = h5py.File('ClutterPackLarge'+clutter_set+'.hdf5', 'r')
        self.clutterPack = f['clutterIMG'][:]
        self.buff_ptr = 0
        self.buff_size = 20
        self.buff_cap = 0
        self.buff_data = np.zeros((self.buff_size, 1, self.image_size_, self.image_size_), dtype=np.float32)
        self.buff_label = np.zeros((self.buff_size, 10))
        self.with_clutters = with_clutters
 

    def GetBatchSize(self):
        return self.batch_size_

    def GetDims(self):
        return self.frame_size_

    def GetDatasetSize(self):
        return self.dataset_size_

    def GetSeqLength(self):
        return self.seq_length_

    def Reset(self):
        pass

    def Overlap(self, a, b):
        """ Put b on top of a."""
        b = np.where(b > (np.max(b) / 4), b, 0)
        t = min(np.shape(a))
        b = b[:t, :t]
        return np.select([b == 0, b != 0], [a, b])
        #return b

    def InitClutterPack(self, num_clutterPack = None, image_size_ = None, num_clutters_ = None):
        if num_clutterPack is None :
            num_clutterPack = self.num_clutterPack
        if image_size_ is None :
            image_size_ = self.image_size_ 
        if num_clutters_ is None :
            num_clutters_ = self.num_clutters_ 
        clutterIMG = np.zeros((num_clutterPack, image_size_, image_size_))
        for i in xrange(num_clutterPack):
            clutterIMG[i] = self.GetClutter(image_size_, num_clutters_)
        f = h5py.File('ClutterPackLarge'+self.clutter_set+'.hdf5', 'w')
        f.create_dataset('clutterIMG', data=clutterIMG)
        f.close()
            
    def GetFakeClutter(self):
        if self.clutterpack_exists:
            return self.clutterPack[np.random.randint(0, len(self.clutterPack))]
    
    def GetClutter(self, image_size_ = None, num_clutters_ = None, fake = False):
        if image_size_ is None :
            image_size_ = self.image_size_
        if num_clutters_ is None :
            num_clutters_ = self.num_clutters_
        if fake and self.clutterpack_exists:
            return self.GetFakeClutter()
        clutter = np.zeros((image_size_, image_size_), dtype=np.float32)
        for i in range(num_clutters_):
            sample_index = np.random.randint(self.data_.shape[0])
            size = np.random.randint(self.clutter_size_min_, self.clutter_size_max_)
            left = np.random.randint(0, self.digit_size_ - size)
            top = np.random.randint(0, self.digit_size_ - size)
            clutter_left = np.random.randint(0, image_size_ - size)
            clutter_top = np.random.randint(0, image_size_ - size)
            single_clutter = np.zeros_like(clutter)
            single_clutter[clutter_top:clutter_top+size, clutter_left:clutter_left+size] = self.data_[np.random.randint(self.data_.shape[0]), top:top+size, left:left+size] / 255.0 * np.random.uniform(self.face_intensity_min, self.face_intensity_max)
            clutter = self.Overlap(clutter, single_clutter)
        return clutter

    def getBuff(self):
        #print 'getBuff ',
        idx = np.random.randint(0, self.buff_cap)
        return self.buff_data[idx], self.buff_label[idx]

    def setBuff(self, data, label):
        self.buff_data[self.buff_ptr]=data
        self.buff_label[self.buff_ptr]=label
        if self.buff_cap < self.buff_size:
            self.buff_cap += 1
        self.buff_ptr += 1
        self.buff_ptr = self.buff_ptr % self.buff_size

    def GetStaticBatch(self, count=1, num_digits=1):
        data = np.zeros((self.batch_size_, 1, self.image_size_, self.image_size_), dtype=np.float32)
        seg = np.zeros((self.batch_size_, 10, self.image_size_, self.image_size_), dtype=np.float32)
        label = np.zeros((self.batch_size_, 10))
        
        for j in xrange(self.batch_size_):
            if self.with_clutters:
                clutter = self.GetClutter(fake=False)
                clutter_bg = self.GetClutter(fake=False)
            for i in xrange(num_digits):
                ind = self.indices_[self.row_]
                self.row_ += 1
                if self.row_ == self.data_.shape[0]:
                    self.row_ = 0
                    np.random.shuffle(self.indices_)
                if count == 2:
                    digit_image = np.zeros((self.data_.shape[1], self.data_.shape[2]))
                    digit_image[:18, :18] = self.Overlap(digit_image[:18, :18], np.maximum.reduceat(np.maximum.reduceat(self.data_[ind], np.cast[int](np.arange(1, 28, 1.5))), np.cast[int](np.arange(1, 28, 1.5)), axis=1))
                    digit_image[10:, 10:] = self.Overlap(digit_image[10:, 10:], np.maximum.reduceat(np.maximum.reduceat(self.data_[np.random.randint(self.data_.shape[0])], np.cast[int](np.arange(0, 27, 1.5))), np.cast[int](np.arange(0, 27, 1.5)), axis=1))
                else:
                    digit_image = self.data_[ind, :, :] / 255.0 * np.random.uniform(self.face_intensity_min, self.face_intensity_max)
                bak_digit_image = digit_image 
                digit_size_ = self.digit_size_
                scale_factor = np.exp((np.random.random_sample()-0.5)*self.scale_range)
                scale_image = misc.imresize(digit_image, scale_factor)
                digit_size_ = digit_size_ * scale_factor 
                top    =  np.random.randint(0, max(1, self.image_size_-digit_size_))
                left   =  np.random.randint(0, max(1, self.image_size_-digit_size_))
                if digit_size_!=np.shape(scale_image)[0]:
                    digit_size_ = np.shape(scale_image)[0]
                bottom = top  + digit_size_
                right  = left + digit_size_
                if right>self.image_size_ or bottom>self.image_size_:
                    scale_image = bak_digit_image
                    bottom = top  + self.digit_size_
                    right  = left + self.digit_size_
                    digit_size_ = self.digit_size_
                digit_image = scale_image
                digit_image_nonzero = np.where(digit_image > (np.max(digit_image) / 4), digit_image, 0).nonzero()
                label_offset = np.array([digit_image_nonzero[0].min(), digit_image_nonzero[1].min(), digit_image_nonzero[0].max(), digit_image_nonzero[1].max()])

                data[j, 0, top:bottom, left:right] = self.Overlap(data[j, 0, top:bottom, left:right], scale_image)
                seg[j, int(self.label_[ind]), top:bottom, left:right] = (self.Overlap(
                        seg[j, int(self.label_[ind]), top:bottom, left:right], scale_image)>0).astype('int')
                label[j, int(self.label_[ind])] = 1
            if self.with_clutters:
                data[j, 0] = self.Overlap(data[j, 0], clutter)
            _EPSI = 1e-6
            for i in xrange(10):
                seg[j, i] = seg[j, i]/(np.sum(seg[j, i])+_EPSI)
            data = data / 255.0
        return data, label, seg


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
