import cv2, glob, numpy
import os

root_dir = '../../Data/sample/'

def create_directory(directory):
    '''
    Creates a new folder in the specified directory if the folder doesn't exist.

    INPUT
        directory: Folder to be created, called as "folder/".

    OUTPUT
        New folder in the current directory.
    '''
    if not os.path.exists(directory):
        os.makedirs(directory)


def scaleRadius(img, scale) :
	x=img[int(img.shape[0]/2),:,:].sum(1)
	r=(x > x.mean()/10).sum()/2
	s=scale*1.0/r
	return cv2.resize(img, (0 ,0 ), fx=s , fy=s )

def preprocess(subfolder_src, subfolder_res, scale) :
	create_directory(root_dir+subfolder_res)
	for f in glob.glob(root_dir+subfolder_src+"*.jpeg"):
	#for f in glob.glob(subfolder_src+"*.jpeg"):
		try:
			a=cv2.imread(f)
			#scale img to a given radius
			a=scaleRadius(a, scale)
			#subtract local mean color
			a=cv2.addWeighted (a, 4 ,
								cv2.GaussianBlur( a, ( 0, 0 ), scale/30), -4 ,
								128)
			#remove outer 10%
			b=numpy.zeros(a.shape)
			cv2.circle(b, (int(a.shape[1]/2) , int(a.shape[0]/2) ),
			int(scale*0.9),( 1, 1, 1 ), -1, 8, 0 )
			a=a*b+128*(1-b)
			splits = f.split('.')
			file_ext = splits[-1]
			end = len(f)-len(file_ext)-1
			#file_name = f[0:end].split('/')[-1]
			file_name = f[0:end].split('\\')[-1]

			#filepath = root_dir+subfolder_res+file_name+'_'+str(scale)+'.'+file_ext
			filepath = root_dir+subfolder_res+file_name+'.'+file_ext
			print('--write to file :', filepath)
			cv2.imwrite(filepath, a)
		except Exception as e: 
			print(e)

if __name__ == '__main__':

	# define resize scale to radius 300
	scale = 300

	# preprocess train dataset
	preprocess('data/train/','data/train-rescale/', scale)

	# preprocess test dataset
	preprocess('data/test/','data/test-rescale/', scale)

