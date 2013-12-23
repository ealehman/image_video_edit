import numpy as np
import numpy as np
import pycuda.autoinit
import pycuda.driver as cu
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpu
from pycuda.curandom import rand as curand
from pycuda.reduction import ReductionKernel
from pycuda.compiler import SourceModule
from sys import argv
import time
import cv2
from cv2 import cv
from Cheetah.Template import Template

from mpi4py import MPI

from filters import *

# CUDA kernel source string
filter_source = """
//include <stdio.h>
//include <stdlib.h>
__global__ void filter_kernel(uchar3* in, uchar3* out)
{
	
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	int i = blockDim.y * blockIdx.y + threadIdx.y;
	int k = i*$WIDTH + j;


	
	float pout0 = 0;
	float pout1 = 0;
	float pout2 = 0;


	if (($MAX_OFF - 1 < i) && (i < ($HEIGHT - $MAX_OFF)) && ($MAX_OFF - 1 < j) && (j< ($WIDTH - $MAX_OFF))) {
		#for $a, $b, $f in $LIST

			pout0 += in[k + $a*$WIDTH + $b].x*$f;
			pout1 += in[k + $a*$WIDTH + $b].y*$f;
			pout2 += in[k + $a*$WIDTH + $b].z*$f;
		#end for
		out[k] = make_uchar3(pout0, pout1, pout2);
		

	}
}
"""

def make_frame_processor(frame_shape, frame_filter_data):
	"""
	Return a function that takes a frame and returns the filtered frame.
	"""
	size, scale, offset, F = frame_filter_data
	frame_filter = np.asarray(F, dtype=np.float32).reshape(size) / scale

	# calculate offsets based on size of filter
	min_offset= (size[0] - 1 )/2*-1
	max_offset = (size[0]- 1)/2 + 1



	# make list of offsets to apply to pixel and neighbors in kernel
	offset_list = []
	filter_list = []
	for a in xrange(min_offset, max_offset):
		for b in xrange(min_offset, max_offset):
			offset_list.append([a,b])

	# append the filter to each pair of offsets
	for f in frame_filter:
		for ff in f:
			filter_list.append(ff)
	
	comb_list = zip(offset_list, filter_list)
	#print comb_list
	# create a list consisting of (offset a, offset b, filter value)
	final_list = []
	for t in comb_list:
		t[0].append(t[1])
		final_list.append((t[0]))

	# Block size (threads per block)
	b_size = (32, 32, 1)  

	print 'frame shape: ' + str(frame_shape) 
	# Grid size (blocks per grid)
	g_size = (int(np.ceil(float(frame_shape[1])/b_size[0])), int(np.ceil(float(frame_shape[0])/b_size[1])))
	
	# initialize template and hard code variables
	template = Template(filter_source)
	template.LIST = final_list
	template.HEIGHT, template.WIDTH, _ = frame_shape
	template.MAX_OFF = max_offset - 1
	template.MIN_OFF = min_offset
	#print template
		
	# Compile the CUDA Kernel
	module = nvcc.SourceModule(template)
	# Return a handle to the compiled CUDA kernel
	filter_kernel = module.get_function("filter_kernel")
  
	def processor(frame):
		"""Applies the frame_filter 2D array to each channel of the image"""
		
		# allocate memory and transfer from host to device
		d_frame_in, d_frame_out = cu.mem_alloc(frame.nbytes), cu.mem_alloc(frame.nbytes) #, cu.mem_alloc(offset.nbytes), cu.mem_alloc(F.nbytes)
		cu.memcpy_htod(d_frame_in, frame)
		cu.memcpy_htod(d_frame_out, frame)
		
		filter_kernel(d_frame_in, d_frame_out, block=b_size, grid= g_size)

		# transfer from device to host
		cu.memcpy_dtoh(frame, d_frame_out)
		return frame
  
	# Return the function
	return processor


def create_video_stream(source, output_filename):
	'''
	Given an input video, creates an output stream that is as similar
	to it as possible.  When no codec is detected in the input, default to MPG-1.
	'''
	DEFAULT_CODEC = cv.CV_FOURCC('P','I','M','1') # MPEG-1 codec

	fps    = int(source.get(cv.CV_CAP_PROP_FPS))
	width  = int(source.get(cv.CV_CAP_PROP_FRAME_WIDTH))
	height = int(source.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
	print "Creating %ifps %i x %i video stream" % (fps, width, height)


	return cv2.VideoWriter(
		filename=output_filename,
		fourcc=int(source.get(cv.CV_CAP_PROP_FOURCC)) or DEFAULT_CODEC,
		fps=fps,
		frameSize=(width, height))


def process_video_serial(source, destination, process_frame, start_frame=0):
	'''
	Given a video stream, processes each frame in serial fashion.
	'''
	# Seek our starting frame, if any specified
	if start_frame > 0:
		source.set(cv.CV_CAP_PROP_POS_FRAMES, start_frame)
		
		# Some video containers don't support precise frame seeking;
		# if this is the case, we bail.
		assert source.get(cv.CV_CAP_PROP_POS_FRAMES) == start_frame

	while source.grab():
		_, frame = source.retrieve()

		#temp = frame
		#temp_filtered = process_frame(temp)
		#destination.write(temp_filtered)

		destination.write(process_frame(frame))

def process_video_parallel(source, destination, process_frame):
	'''
	Process the video stream in parallel.
	'''
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	size = comm.Get_size()
	device = pycuda.autoinit.device.pci_bus_id()
	node   = MPI.Get_processor_name()

	print "Rank %d/%d has GPU %s on Node %s" % (rank, size, device, node)
	
	# pad vid array with zeros
	#full_filtered = np.empty() figure out size

	frame_count = int(source.get(cv.CV_CAP_PROP_FRAME_COUNT))
	chunk = frame_count/size
	# Seek our starting frame, if any specified

	n = (frame_count+size-1)/size
	
	start_frame = chunk*rank
	source.set(cv.CV_CAP_PROP_POS_FRAMES, start_frame)

	#assert source.get(cv.CV_CAP_PROP_POS_FRAMES) == start_frame

	frame_shape = (int(source.get(cv.CV_CAP_PROP_FRAME_HEIGHT)), int(source.get(cv.CV_CAP_PROP_FRAME_WIDTH)), 3)
	filtered_chunk = np.zeros((chunk, frame_shape[0], frame_shape[1], frame_shape[2]), dtype= np.uint8)
	n_frame = 0
	
	while source.get(cv.CV_CAP_PROP_POS_FRAMES) < start_frame + chunk and source.grab():
		_, frame = source.retrieve()
		
		filtered_chunk[n_frame,:,:,:] = process_frame(frame)
		n_frame += 1

	#filtered_all = np.empty((chunk*size, frame_shape[0], frame_shape[1], frame_shape[2]), dtype= np.uint8)
	filtered_all = np.empty((n*size, frame_shape[0], frame_shape[1], frame_shape[2]), dtype= np.uint8)

	start_time = MPI.Wtime()
	comm.Gather(filtered_chunk, filtered_all, root=0)
	#filtered_all = comm.gather(filtered_chunk, root=0)
	end_time = MPI.Wtime();

	#if rank == 0:
	#	print 'MPI time: ' + str(end_time - start_time)

	if rank == 0:
		for x in xrange(frame_count):
			destination.write(filtered_all[x, :,:,:])
  

if __name__ == '__main__':
	if len(argv) != 3:
		print "Usage: python", argv[0], "[input video] [output video]"
		exit()

	# Open our source video and create an output stream
	source = cv2.VideoCapture(argv[1])
	destination = create_video_stream(source, argv[2])

	# Get the metadata on a frame
	frame_shape = (int(source.get(cv.CV_CAP_PROP_FRAME_HEIGHT)),
				 int(source.get(cv.CV_CAP_PROP_FRAME_WIDTH)),
				 3)

	# Define filters
	#identity          = np.ones((1,1), dtype=np.float32)

	#average_blur      = np.ones((3,3), dtype=np.float32) / 9
	#wide_average_blur = np.ones((5,5), dtype=np.float32) / 25
	#big_average_blur  = np.ones((7,7), dtype=np.float32) / 49
	#sharpen           = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]], dtype=np.float32)

	#sobel_horizontal  = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=np.float32)
	#sobel_vertical    = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)

	#motion_blur = np.eye((9,9), dtype=np.float32))

	# Create a frame processor based on the shape and a filter
	frame_processor = make_frame_processor(frame_shape, ave_5)

	# Apply the frame_processor to each frame of the source video
	"""
	start_time = time.time()
	process_video_serial(source, destination, frame_processor)
	end_time = time.time()


	print "Processing Time: %f" % (end_time - start_time)
	"""
	
	start_time2 = time.time()
	process_video_parallel(source, destination, frame_processor)
	end_time2 = time.time()
	print "Processing Time CUDA + MPI: %f" % (end_time2 - start_time2)
	
