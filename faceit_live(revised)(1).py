import imageio
import numpy as np
import pandas as pd
from skimage.transform import resize
import warnings
import sys
import cv2
import time
import PIL.Image as Image
import PIL.ImageFilter
import io
from io import BytesIO
import pyautogui
import os
import glob
from argparse import Namespace
import argparse
import timeit
import torch
warnings.filterwarnings("ignore")


from PIL import Image
from numpy import asarray
import dlib
import face_recognition
import face_recognition_models

############## setup ####
stream = True
media_path = 'C:/Users/Administrator/faceit_live3/media/'
model_path = 'C:/Users/Administrator/faceit_live3/model/'

parser = argparse.ArgumentParser()
parser.add_argument('--webcam_id', type = int, default = 0)
parser.add_argument('--stream_id', type = int, default = 1)
parser.add_argument('--gpu_id', type = int, default = 0)
parser.add_argument('--system', type = str, default = "win")

args = parser.parse_args()
webcam_id = args.webcam_id
gpu_id = args.gpu_id
stream_id = args.stream_id
system = args.system

webcam_height = 480
webcam_width = 640
screen_width, screen_height = pyautogui.size()
img_shape = [256, 256, 0]

if system=="linux":
    print("Linux version, importing FakeWebCam")
    import pyfakewebcam


first_order_path = 'C:/Users/Administrator/faceit_live3/first-order-model/'
sys.path.insert(0,first_order_path)
reset = True

# import methods from first-order-model
import demo
from demo import load_checkpoints, make_animation, tqdm

# prevent tqdm from outputting to console
demo.tqdm = lambda *i, **kwargs: i[0]

print("CUDA is available: ",torch.cuda.is_available())
if (torch.cuda.is_available()):
    torch.cuda.device("cuda:" + str(gpu_id))
    print("Device Name:",torch.cuda.get_device_name(gpu_id))
    print("Device Count:",torch.cuda.device_count())
    print("CUDA: ",torch.version.cuda)
    print("cuDNN",torch.backends.cudnn.version())
    print("Device",torch.cuda.current_device())


img_list = []
print("Scanning /media folder for images to use...")
for filename in os.listdir(media_path):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png") or filename.endswith(".PNG"):
        img_list.append(os.path.join(media_path, filename))
        print(os.path.join(media_path, filename))

#print(img_list, len(img_list))

############## end setup ####

def main():

	# Load the jpg file into a numpy array
	#img = face_recognition.load_image_file("C:/Users/Administrator/faceit_live3/media/1.jpg")

	# Find all the faces in the image using the default HOG-based model.
	# This method is fairly accurate, but not as accurate as the CNN model and not GPU accelerated.
	# See also: find_faces_in_picture_cnn.py


	global source_image

	#source_image =  readnextimage(0)

	image = imageio.imread("C:/Users/Administrator/faceit_live3/media/1.jpg")
	face_locations = face_recognition.face_locations(image)

	print("I found {} face(s) in this photograph.".format(len(face_locations)))

	for face_location in face_locations:
	    # Print the location of each face in this image
	    top, right, bottom, left = face_location
	    print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

	top = top-int((bottom-top)/2)
	bottom = bottom+int((bottom-top)/2)
	left = left-int((right-left)/2)
	right = right+int((right-left)/2)
	source_image = image[top:bottom, left:right]
	#source_image = Image.fromarray(source_image)
	#source_image.show()


	image = Image.fromarray(image)

	from skimage.transform import resize
	source_image = resize(source_image, (640,640))
	#source_image.show()

	# start streaming
	if system=="linux":
	    camera = pyfakewebcam.FakeWebcam(f'/dev/video{stream_id}', webcam_width, webcam_height)
	    camera.print_capabilities()
	    print(f"Fake webcam created on /dev/video{stream_id}. Use Firefox and join a Google Meeting to test.")

	# capture webcam
	video_capture = cv2.VideoCapture(webcam_id)
	time.sleep(1)
	width = video_capture.get(3)  # float
	height = video_capture.get(4) # float
	print("webcam dimensions = {} x {}".format(width,height))

	# load models
	net = load_face_model()
	generator, kp_detector = demo.load_checkpoints(config_path=f'{first_order_path}config/vox-adv-256.yaml', checkpoint_path=f'{model_path}/vox-adv-cpk.pth.tar')


	# create windows
	cv2.namedWindow('Face', cv2.WINDOW_GUI_NORMAL) # extracted face
	cv2.moveWindow('Face', int(screen_width//2)-150, 100)
	cv2.resizeWindow('Face', 256,256)

	cv2.namedWindow('DeepFake', cv2.WINDOW_GUI_NORMAL) # face transformation
	cv2.moveWindow('DeepFake', int(screen_width//2)+150, 100)
	cv2.resizeWindow('DeepFake', 256, 256)

	cv2.namedWindow('Stream', cv2.WINDOW_GUI_NORMAL) # rendered to fake webcam
	cv2.moveWindow('Stream', int(screen_width//2)-int(webcam_width//2), 400)
	cv2.resizeWindow('Stream', webcam_width,webcam_height)


	print("Press C to center Webcam, Press B/N for previous/next image in media directory, T to alter between relative and absolute transformation, Q to quit")
	x1,y1,x2,y2 = [0,0,0,0]
	relative = True
	previous = None



	while True:


	    ret, frame = video_capture.read()
	    frame = cv2.resize(frame, (640, 480))
	    frame = cv2.flip(frame,1)

	    if (previous is None or reset is True):
	        x1,y1,x2,y2 = find_face_cut(net,frame)
	        previous = cut_face_window(x1,y1,x2,y2,frame)
	        reset = False
	        #img_shape = source_image.shape
	        #cv2.resizeWindow('DeepFake', int(img_shape[1] // img_shape[0] * 256), 256)
	        #cv2.imshow('Previous',previous)


	    curr_face = cut_face_window(x1,y1,x2,y2,frame.copy())
	    # cv2.imshow('Previous',previous)
	    # cv2.imshow('Curr Face',curr_face)
	    # cv2.imshow('Source Image',source_image)

	    deep_fake = process_image(source_image,previous,curr_face,net, generator, kp_detector, relative)
	    #print("deep_fake",deep_fake.shape)

	    
	    deep_fake_array = Image.fromarray((deep_fake*255).astype(np.uint8)) # change deep_fake image from numpy array to Image
	    deep_fake_array = deep_fake_array.resize((right-left,bottom-top))
	    image.paste(deep_fake_array, (left,top,right,bottom))
	    
	    deep_fake = cv2.cvtColor(deep_fake, cv2.COLOR_RGB2BGR)
	    stream_v = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
	    
	    '''
	    rgb = cv2.resize(deep_fake,(int(source_image.shape[0] // source_image.shape[1] * 480),480))


	    # pad image 
	    x_border = int((640-(img_shape[1] // img_shape[0] * 480))//2)
	    #y_border = int((480-(img_shape[0] // img_shape[1] * 640))//2)

	    stream_v = cv2.copyMakeBorder(rgb2, 0, 0, x_border if x_border >=0 else 0, x_border if x_border >=0 else 0, cv2.BORDER_CONSTANT)
	    '''

	    #cv2.imshow('Webcam', frame)
	    cv2.imshow('Face', curr_face)
	    cv2.imshow('DeepFake', deep_fake)
	    #cv2.imshow('Previous', previous)
	    #cv2.imshow('RGB', rgb)
	    #cv2.imshow('Source Image', source_image)
	    #time.sleep(1/30.0)

	    cv2.imshow('Stream',stream_v)

	    '''
	    # stream to fakewebcam
	    if system=="linux":
	        stream_v = cv2.flip(stream_v,1)
	        stream_v = cv2.cvtColor(stream_v, cv2.COLOR_BGR2RGB)
	        stream_v = (stream_v*255).astype(np.uint8)
	        #print("output to fakecam")
	        camera.schedule_frame(stream_v)
	    '''

	    k = cv2.waitKey(1) 
	    # Hit 'q' on the keyboard to quit!
	    if k & 0xFF == ord('q'):
	        print("Quiting")
	        video_capture.release()
	        break
	    elif k==ord('c'):
	        # center
	        print("Centering the image")
	        reset = True
	    elif k==ord('b'):
	        # previous image
	        print("Loading previous image")
	        source_image = readpreviousimage()
	        reset = True
	    elif k==ord('n'):
	        # next image
	        print("Loading next image")
	        source_image = readnextimage()
	        reset = True
	    elif k==ord('t'):
	        # rotate 
	        relative = not relative
	        print("Changing transform mode")


	cv2.destroyAllWindows()
	exit()


# transform face with first-order-model
def process_image(source_image,base,current,net, generator,kp_detector,relative):
	predictions = make_animation(source_image, [base,current], generator, kp_detector, relative=relative, adapt_movement_scale=False)
	return predictions[1] 

def load_face_model():
	modelFile = f"{model_path}/res10_300x300_ssd_iter_140000.caffemodel"
	configFile = f"{model_path}deploy.prototxt.txt"
	net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
	return net

def cut_face_window(x1,y1,x2,y2,frame):
	frame = frame.copy()
	frame = frame[y1:y2,x1:x2]
	face = resize(frame, (256, 256))[..., :3]
	return face

	# find the face in webcam stream and center a 256x256 window
def find_face_cut(net,face):
	blob = cv2.dnn.blobFromImage(face, 1.0, (300, 300), [104, 117, 123], False, False)
	frameWidth = 640
	frameHeight = 480
	net.setInput(blob)
	detections = net.forward()
	bboxes = []
	face_found = False
	for i in range(detections.shape[2]):
	    #print(i)
	    confidence = detections[0, 0, i, 2]
	    if confidence > 0.9:
	        x1 = (int(detections[0, 0, i, 3] * frameWidth)//2)*2
	        y1 = (int(detections[0, 0, i, 4] * frameHeight)//2)*2
	        x2 = (int(detections[0, 0, i, 5] * frameWidth)//2)*2
	        y2 = (int(detections[0, 0, i, 6] * frameHeight)//2)*2

	        face_margin_w = int(256/2 - (abs(x1-x2)))
	        face_margin_h = int(256/2 - (abs(y1-y2)))

	        if(face_margin_w<0): face_margin_w=0
	        if(face_margin_h<0): face_margin_h=0


	        cut_x1 = int(x1 -(abs(x1-x2))*0.2)
	        cut_y1 = int(y1 -(abs(x1-x2))*0.4)

	        cut_x2 = int(x2 +(abs(x1-x2))*0.3)
	        cut_y2 = int(y2 +(abs(x1-x2))*0.4)

	        face_found = True
	        break

	if not face_found:
	    print("No face detected in video")
	    # let's just use the middle section of the image
	    cut_x1,cut_y1,cut_x2,cut_y2 = 112,192,623,704
	#(256,256)*2
	else:
	    print(f'Found face at: ({x1,y1}) ({x2},{y2} width:{(x2-x1)} height: {(y2-y1)})')
	    print(f'Cutting at: ({cut_x1,cut_y1}) ({cut_x2},{cut_y2} width:{(cut_x2-cut_x1)} height: {(cut_y2-cut_y1)})')


	return cut_x1,cut_y1,cut_x2,cut_y2

def readimage():
    global img_list,img_shape
    img = imageio.imread(img_list[pos])
    img = resize(img, (256, 256))[..., :3]
    return img

def readpreviousimage():
    global pos
    if pos<len(img_list)-1:
        pos=pos-1
    else:
        pos=0
    return readimage()

def readnextimage(position=-1):
    global pos
    if (position != -1):
        pos = position
    else:
        if pos<len(img_list)-1:
            pos=pos+1
        else:
            pos=0
    return readimage()

main()