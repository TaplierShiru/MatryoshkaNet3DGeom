from __future__ import print_function
from utils import convert_files, convert_files_parallel
import binvox_rw 
import scipy.io as sio
import numpy as np
import argparse


def convert_binvox(filename):
	""" Converts a single binvox file to Matlab. 
		The resulting mat file stores one variable 'voxel'.
	"""
	print('Converting %s ... ' % filename, end='')
	try:
		with open(filename, 'rb') as f:
			model = binvox_rw.read_as_3d_array(f)
			v = np.array(model.data, dtype='uint8')
			sio.savemat(filename[:-7]+'.vox.mat', {'voxel':v[::-1,::-1,::-1].copy()}, do_compression=True)
			pass
		pass
	except:
		print('failed.')
		return
	print('done.')
	pass


if __name__ == '__main__':

	parser = argparse.ArgumentParser('Converts .binvox files to .mat files.')
	parser.add_argument('directory', type=str, help='Directory with binvox files.', default='.')
    parser.add_argument('-p', '--parallel', action='store_true', help='Perform parallel convertaion which is fast.')
    parser.add_argument('-w', '--workers', type=int, default=10, help='Numer of process/theards in parallel convertaion.')
	parser.add_argument('-r', '--recursive', action='store_true', help='Recursively traverses the directory and converts all binvox files.')

	args = parser.parse_args()
    
    if args.parallel:
        convert_files_parallel(args.directory, '.binvox', convert_binvox, args.recursive, num_process=args.workers)
    else:
        convert_files(args.directory, '.binvox', convert_binvox, args.recursive)
