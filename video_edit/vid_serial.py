from sys import argv
import numpy as np
import time

import cv2
from cv2 import cv

from filters import *

def make_frame_processor(frame_shape, frame_filter_data):
  """
  Return a function that takes a frame and returns the filtered frame.
  """
  size, scale, offset, F = frame_filter_data
  frame_filter = np.asarray(F, dtype=np.float32).reshape(size) / scale
  #print 'size: ' + str(size)
  #print 'scale: ' + str(scale)
  #print 'offeset: ' + str(offset)
  #print 'F: ' + str(F)
  # Define a function for frame processing

  print 'frame shape: ' + str(frame_shape)

  
  def processor(frame):
    """Applies the frame_filter 2D array to each channel of the image"""
    


    return cv2.filter2D(frame, -1, frame_filter) + offset


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
    #print 'frame[0]: ' + str(frame[0])
    #print 'frame[1]: ' + str(frame[1])
    destination.write(process_frame(frame))


def process_video_parallel():
  '''
  Process the video stream in parallel.
  '''
  raise NotImplementedError


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
  frame_processor = make_frame_processor(frame_shape, sobel_horizontal_7)

  # Apply the frame_processor to each frame of the source video
  start_time = time.time()
  process_video_serial(source, destination, frame_processor)
  end_time = time.time()

  print "Processing Time: %f" % (end_time - start_time)
