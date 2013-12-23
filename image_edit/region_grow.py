import numpy as np
import matplotlib.image as img
import numpy as np
import pycuda.autoinit
import pycuda.driver as cu
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpu
from pycuda.curandom import rand as curand
from pycuda.reduction import ReductionKernel
from pycuda.compiler import SourceModule
import math
import time

in_file_name  = "Harvard_Large.png"
out_file_name = "Harvard_RegionGrow_GPU_A.png"


# Region growing constants [min, max]
seed_threshold = [0, 0.08];
threshold      = [0, 0.27];

fullgrow_source = """
#include <stdio.h>
__global__ void fullgrow_kernel(double* d_image, int* d_region, int* d_conv, int h, int w)
{

	int j = blockDim.x * blockIdx.x + threadIdx.x;
	int i = blockDim.y * blockIdx.y + threadIdx.y;
	int index = i*w + j;

	if ((0 < i) && (i < (h - 1)) && (0 < j) && (j < (w - 1))) {
		if (d_image[index] > 0.0 && d_image[index] < .27) {

			if (d_region[index + 1] == 1 || d_region[index - 1] == 1 || d_region[index + w] == 1 || d_region[index - w] == 1) {
				d_region[index] = 1;
				d_conv[index] = 1;
			}

		}
	}
}
"""

findseeds_source = """

__global__ void findseeds_kernel(double* d_image, int* d_region, int h, int w) {

	int j = blockDim.x * blockIdx.x + threadIdx.x;
	int i = blockDim.y * blockIdx.y + threadIdx.y;
	int index = i*w + j;

	if ((0 < i) && (i < (h - 1)) && (0 < j) && (j < (w - 1))) {
		if (d_image[index] > 0.0 && d_image[index] < .08) {
			d_region[index] = 1;
		}
	}
}

"""
if __name__ == '__main__':
	
	# Compile the CUDA Kernel
	module = nvcc.SourceModule(fullgrow_source)
	# Return a handle to the compiled CUDA kernel
	fullgrow_kernel = module.get_function("fullgrow_kernel")

	# Compile findseeds kernel
	module2 = nvcc.SourceModule(findseeds_source)
	# Return a handle to the compiled kernel
	findseeds_kernel = module2.get_function("findseeds_kernel")
	
	# Read image. BW images have R=G=B so extract the R-value
	image = img.imread(in_file_name)[:,:,0]
	
	start = time.time()
	# extract height and width of image 
	height, width = np.int32(image.shape)
	N = height*width
	
	print "Processing %d x %d image" % (width, height)

	# Initialize the image region as empty
	im_region = np.zeros([height, width], dtype=np.int32)
	
	orig_im = np.array(image, dtype=np.float64)
	
	# Allocate device memory and copy host to device
	d_region = gpu.to_gpu(im_region)
	d_image = gpu.to_gpu(orig_im)


	# Block size (threads per block)
	b_size = (32, 32, 1)

	# Grid size (blocks per grid)
	g_size = (int(np.ceil(float(width)/float(b_size[0]))), int(np.ceil(float(height)/float(b_size[1]))))
	
	# find seeds, update seeds in im_region, add indices of it's neighbors to the next_front 

	#  each thread goes to a pixels in next front (which is now this front) 
	# and checks if it's already in region:
		# if yes: determines if it needs to be added to the region if yes > add it and then add it's neighbors to next front
		# each kernel: goes to  shit a pixel in next front check if needs to be added, add, add neighbors to front
		#while loop?
	# iterate grow_region on a while loop until front

	# Find pixels within seed threshold and change them to white in the grow region 
	findseeds_kernel(d_image, d_region, height, width, block = b_size, grid=g_size)

	while True:

		# Create an array of zeroes that will be updated when pixels are added to the region
		# This will act as the test for convergence
		# The while loop will only run as long as at least one element in this array has been updated 
		#d_conv = gpu.zeros(int(N), dtype=np.int32)
		d_conv_test = gpu.zeros(int(N), dtype=np.int32)

		# Copy from host to device
		#d_conv_test = gpu.to_gpu(conv_test)
		
		fullgrow_kernel(d_image, d_region, d_conv_test, np.int32(height), np.int32(width), block= b_size, grid=g_size)
		# print d_conv_test.get()

		
		if sum(d_conv_test.get()):
			break
		else:
			#d_conv_test.gpudata.free()
			continue
		d_conv_test = gpu.zeros.fill(np.int32(0))
		
		

	im_region = d_region.get()
	stop = time.time()
	print 'time: ' + str(stop- start)
	#print 'dark region: ' + str(d_region.get())
	#print 'im_region: ' + str(im_region)
	#print d_image.get()
	img.imsave(out_file_name, im_region, cmap='gray')