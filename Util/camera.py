import ipywidgets
from IPython.display import display
from jetcam.utils import bgr8_to_jpeg
import numpy as np
import time

import traitlets
import threading
import atexit
import cv2


class Camera(traitlets.HasTraits):

    value = traitlets.Any()
    width = traitlets.Integer(default_value=224)
    height = traitlets.Integer(default_value=224)
    format = traitlets.Unicode(default_value='bgr8')
    running = traitlets.Bool(default_value=False)
    
    capture_device = traitlets.Integer(default_value=0)
    capture_fps = traitlets.Integer(default_value=30)
    capture_width = traitlets.Integer(default_value=640)
    capture_height = traitlets.Integer(default_value=480)
    
    def __init__(self, *args, **kwargs):
        super(Camera, self).__init__(*args, **kwargs)
        if self.format == 'bgr8':
            self.value = np.empty((self.height, self.width, 3), dtype=np.uint8)
        self._running = False
        
        try:
            self.cap = cv2.VideoCapture(self._gst_str(), cv2.CAP_GSTREAMER)

            re, image = self.cap.read()

            if not re:
                raise RuntimeError('Could not read image from camera.')
        except:
            raise RuntimeError(
                'Could not initialize camera.  Please see error trace.')

        atexit.register(self.cap.release)
        
    def _gst_str(self):
        return 'nvarguscamerasrc sensor-id=%d ! video/x-raw(memory:NVMM), width=%d, height=%d, format=(string)NV12, framerate=(fraction)%d/1 ! nvvidconv ! video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! videoconvert ! appsink drop=1' % (
                self.capture_device, self.capture_width, self.capture_height, self.capture_fps, self.width, self.height)
            
    def _read(self):
        re, image = self.cap.read()
        if re:
            return image
        else:
            raise RuntimeError('Could not read image from camera')
        
    def read(self):
        if self._running:
            raise RuntimeError('Cannot read directly while camera is running')
        self.value = self._read()
        return self.value
    
    def _capture_frames(self):
        while True:
            if not self._running:
                break
            self.value = self._read()
            
    @traitlets.observe('running')
    def _on_running(self, change):
        if change['new'] and not change['old']:
            # transition from not running -> running
            self._running = True
            self.thread = threading.Thread(target=self._capture_frames)
            self.thread.start()
        elif change['old'] and not change['new']:
            # transition from running -> not running
            self._running = False
            self.thread.join()


class CameraDisplay:
    def __init__(self, img_to_display_img_callback, lazy_camera_init: bool = False):
        self.img_to_display_img_callback = img_to_display_img_callback
        self.lazy_camera_init = lazy_camera_init
        if not self.lazy_camera_init:
            self.initialize_camera()
        else:
            self.camera = None
        self.image_widget = ipywidgets.Image(format='jpeg')
        self.image_widget.value = bgr8_to_jpeg(np.zeros((320, 320, 3), dtype=np.uint8))
        display(self.image_widget)
        
        self._processing_frame = False
        self.fps = None
    
    def initialize_camera(self):
        print('Initializing camera...')
        self.camera = Camera(width=640, height=360, capture_width=1280, capture_height=720, capture_fps=30)
        self.camera.observe(self._camera_callback, names='value')
    
    def release(self):
        if not self.camera is None:
            self.camera.running = False
            if self.camera.cap is not None:
                self.camera.cap.release()
            print("Camera released")
            return
    
    def _camera_callback(self, change):
        if not self._processing_frame:
            self._processing_frame = True
            image = change['new']
            if not self.img_to_display_img_callback is None:
                image = self.img_to_display_img_callback(image)
            self.image_widget.value = bgr8_to_jpeg(image)
            self._processing_frame = False
    
    def start(self):
        if self.camera is None:
            self.initialize_camera()
        self.camera.running = True
        self._processing_frame = False
    
    def stop(self):
        self.camera.running = False
    
    def __del__(self):
        self.release()