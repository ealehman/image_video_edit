import numpy as np
import pycuda.autoinit
import pycuda.driver as cu
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpu
from pycuda.curandom import rand as curand
from pycuda.reduction import ReductionKernel
import matplotlib.image as img
import time

# Image files
in_file_name  = "Harvard_Large.png"
out_file_name = "Harvard_Sharpened_GPU.png"
# Sharpening constant
EPSILON    = np.float32(.005)


sharpen_source = \
"""
__global__ void sharpen_kernel(double* d_curr, float eps, double* d_next, int I, int J)
{
	
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	int i = blockDim.y * blockIdx.y + threadIdx.y;

	if ((0 < i) && (i < (I - 1)) && (0 < j) && (j< (J-1))) {
		d_next[i*J + j] = d_curr[i*J + j] + eps*(-1*d_curr[(i-1)*J + j-1] + -2*d_curr[(i-1)*J + j] + -1*d_curr[(i-1)*J + j+1] + -2*d_curr[i*J + j-1] + 12*d_curr[i*J + j] + -2*d_curr[i*J + j+1] + -1*d_curr[(i+1)*J + j-1] + -2*d_curr[(i+1)*J + j] + -1*d_curr[(i+1)*J + j+1]);
	}
}
"""

def gpu_sharpen(kernel, orig_image):
	# allocate memory for input and output
	curr_im, next_im = np.array(orig_image, dtype=np.float64), np.array(orig_image, dtype=np.float64)
	
	
	# Get image data
	height, width = np.int32(orig_image.shape)
	N = height * width
	print "Processing %d x %d image" % (width, height)

	# Allocate device memory and copy host to device
	start_transfer = time.time()
	d_curr = gpu.to_gpu(curr_im)
	d_next = gpu.to_gpu(next_im)
	stop_transfer = time.time()
  	host_to_device = stop_transfer - start_transfer
  	print "host to device tranfer time: " + str(host_to_device)

	# Block size (threads per block)
	b_size = (32, 32, 1)  
	33
	# Grid size (blocks per grid)
	g_size = (int(np.ceil(float(width)/float(b_size[0]))), int(np.ceil(float(height)/float(b_size[1])))) 
	# Initialize the GPU event trackers for timing
  	start_gpu_time = cu.Event()
  	end_gpu_time = cu.Event()
	
	start_gpu_time.record()
	
	# Compute the image's initial mean and variance
	init_mean = np.float64(gpu.sum(d_curr).get())/N

	var = ReductionKernel(dtype_out=np.float64, neutral= "0", reduce_expr= "a+b", map_expr="(x[i]-mu)*(x[i]-mu)/size", arguments="double* x, double mu, double size")
	init_variance = var(d_curr, np.float64(init_mean), np.float64(N)).get()
	
	variance = 0
	total = 0
	# while variance is less than a 20% difference from the initial variance, continue to sharpen
	while variance < 1.2 * init_variance:

		kernel(d_curr, EPSILON, d_next, height, width, block=b_size, grid=g_size)

		# Swap references to the images, next_im => curr_im
		d_curr, d_next = d_next, d_curr
		
		# calculate mean and variance
		mean = np.float64(gpu.sum(d_curr).get())/N

		variance = var(d_curr, np.float64(mean), np.float64(N)).get()
		
		print "Mean = %f,  Variance = %f" % (mean, variance)
	end_gpu_time.record()
	end_gpu_time.synchronize()
	gpu_time = start_gpu_time.time_till(end_gpu_time)*1*1e-3 

	print "GPU Time: %f" % gpu_time

	return d_curr.get()

if __name__ == '__main__':

	# Compile the CUDA Kernel
	module = nvcc.SourceModule(sharpen_source)
	# Return a handle to the compiled CUDA kernel
	sharpen_kernel = module.get_function("sharpen_kernel")

	# Read image. BW images have R=G=B so extract the R-value
	original_image = img.imread(in_file_name)[:,:,0]
	overall_start = time.time()
	
	# Print the GPU result
	image = gpu_sharpen(sharpen_kernel, original_image)
	overall_stop = time.time()
	print 'Overall time: ' + str(overall_stop - overall_start)

	# Save the current image. Clamp the values between 0.0 and 1.0
	img.imsave(out_file_name, image, cmap='gray', vmin=0, vmax=1)

