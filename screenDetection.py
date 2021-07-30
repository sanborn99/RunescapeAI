import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import zipfile
import cv2
from mss import mss
import pyautogui
import time
import PySimpleGUI as sg
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from numpy import array
from utils import label_map_util
import random

from utils import visualization_utils as vis_util
from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

import six
from six.moves import range
from six.moves import zip


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")


# script repurposed from sentdex's edits and TensorFlow's example script. Pretty messy as not all unnecessary
# parts of the original have been removed



# # Model preparation

# ## Variables
#
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.
#
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.



# What model to download.
MODEL_NAME = 'output_inference_graph.pb'  # change to whatever folder has the new graph
# MODEL_FILE = MODEL_NAME + '.tar.gz'   # these lines not needed as we are using our own model
# DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')  # our labels are in training/object-detection.pbkt

NUM_CLASSES = 2  # we only are using one class at the moment (mask at the time of edit)


# ## Download Model


# opener = urllib.request.URLopener()   # we don't need to download model since we have our own
# opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
# tar_file = tarfile.open(MODEL_FILE)
# for file in tar_file.getmembers():
#     file_name = os.path.basename(file.name)
#     if 'frozen_inference_graph.pb' in file_name:
#         tar_file.extract(file, os.getcwd())


# ## Load a (frozen) Tensorflow model into memory.


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine



label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)




def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)




# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'testImages'
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpeg'.format(i)) for i in range(1, 8)]  # adjust range for # of images in folder

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


sg.theme('DarkAmber')
layout = [  [sg.Text('Runescape Bot')],
            #[sg.Text('Password'), sg.InputText()],
            [sg.Checkbox('Tin Ore', default=True, key = "tin_ore")],
            [sg.Checkbox('Clay Ore', default=True, key = "clay_ore")],
            [sg.Button('Start'), sg.Button('Stop'), sg.Button('Exit')],
            [sg.Checkbox('Drop Items', default=True, key = "dropInvent")] ]

window = sg.Window('Runescape AI', layout, keep_on_top=True, location=(10,50), resizable=True, auto_size_text=True, no_titlebar=True)

monSize = pyautogui.size()

mon1width  = monSize[0] # 2560
mon1height = monSize[1] # 1080

mon2width  = 2560
mon2height = 1080

#screenshot = mss.mss().grab(window)
sct = mss()

mon2 = sct.monitors[1]
bounding_box = {'top': mon2['top'] + 0, 'left': mon2['left'] + 0, 'width': mon2width, 'height': mon2height}

class spot:
    def __init__(self, ymin, xmin, ymax, xmax, box_class_name):
        self.xmin = xmin * mon1width
        self.xmax = xmax * mon1width
        self.ymin = ymin * mon1height
        self.ymax = ymax * mon1height

        self.centerX = ((xmax * mon1width) + (xmin * mon1width)) / 2
        self.centerY = ((ymax * mon1height) + (ymin * mon1height)) / 2

        self.class_name = box_class_name


currentMillis = time.time() * 1000
prevMillis = currentMillis
timeDelta = random.randint(4,5) * 1000
teleportStartupDelay = 20000
teleportDelta = 20000


runAgent = True
harvestStatus = False
waiting = False

inventoryGuiPos = [2245, 1020]
magicBookGuiPos = [2345, 1020]
homeTeleportGuiPos = [2380, 745]


bankDepositIterator = 0
travelBool = False

bankTravelPath = [[2542, 81], [2517, 47], [2478, 34], [2452, 40], [2463, 35], [2503, 39], [2528, 58],
                  [2428, 54], [2407, 94], [2463, 37], [2438, 46], [2471, 34], [2470, 36], [2433, 48],
                  [2416, 70], [2484, 65]]

bankItemsPath = [[1532, 538], [1533, 581], [1354, 825]]

mineTravelPath = [[2443, 169], [2458, 178], [2496, 177], [2512, 172]]


def bankItems(bankPath, bankInteraction, minePath, bankDepositIterator):
    print("pong")
    if bankDepositIterator <= len(bankTravelPath):
        pyautogui.click(bankPath[bankDepositIterator][0], bankPath[bankDepositIterator][1], duration = np.random.uniform(0.2, 0.6))
    elif bankDepositIterator <= (len(bankTravelPath) + len(bankInteraction)):
        pyautogui.click(bankItemsPath[0][0], bankItemsPath[0][1], duration = np.random.uniform(0.2, 0.6), button = 'right')
        pyautogui.click(bankItemsPath[1][0], bankItemsPath[1][1] + 27, duration = np.random.uniform(0.1, 0.3))
        pyautogui.click(bankItemsPath[2][0], bankItemsPath[2][0], duration = np.random.uniform(0.1, 0.3))
    elif bankDepositIterator <= (len(bankTravelPath) + len(bankInteraction) + len(mineTravelPath)):
        pyautogui.click(bankPath[bankDepositIterator - len(bankTravelPath) - len(minePath)][0], bankPath[bankDepositIterator - len(bankTravelPath - len(minePath))][1], duration = np.random.uniform(0.2, 0.6))
    


def homeTeleport():
    pyautogui.moveTo(magicBookGuiPos[0], magicBookGuiPos[1], duration = np.random.uniform(0.2, 0.6))
    pyautogui.click()
    pyautogui.click(homeTeleportGuiPos[0], homeTeleportGuiPos[1], duration = np.random.uniform(0.2, 0.6), button = 'right')
    pyautogui.click(homeTeleportGuiPos[0], homeTeleportGuiPos[1] + 27, duration = np.random.uniform(0.1, 0.3))
    pyautogui.click(inventoryGuiPos[0], inventoryGuiPos[1], duration = np.random.uniform(0.1, 0.3))



firstInventoryPos = [2400, 760]
dropButtonDistance = 40

def dropFirstItem():
    pyautogui.click(firstInventoryPos[0], firstInventoryPos[1], button = 'right', duration = np.random.uniform(0.1, 0.2))
    pyautogui.click(firstInventoryPos[0], firstInventoryPos[1] + dropButtonDistance, duration = np.random.uniform(0.05, 0.1))


estNumHarvested = 0

# Harvest Config:
harvestConfig = {
    "tin_ore" : True,
    "clay_ore" : True
}
dropInvent = True

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:
            
            img = pyautogui.screenshot()
            frame = np.array(img)
            image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            #print(pyautogui.position())        # Cursor Position Finder
            currentMillis = time.time() * 1000
            currentPos = pyautogui.position()

            #### GUI handling ####
            event, values = window.read(timeout=0)
            if event == sg.WIN_CLOSED or event == 'Exit': # if user closes window or clicks cancel
                break
            if event == 'Start':
                print("starting harvest")
                harvestStatus = True
            if event == 'Stop':
                print("stopping harvest")
                harvestStatus = False

            if values["dropInvent"] == True:
                dropInvent = True
            elif values["dropInvent"] == False:
                dropInvent = False

            if values["clay_ore"] == True:
                harvestConfig["clay_ore"] = True
            elif values["clay_ore"] == False:
                harvestConfig["clay_ore"] = False
            if values["tin_ore"] == True:
                harvestConfig["tin_ore"] = True
            elif values["tin_ore"] == False:
                harvestConfig["tin_ore"] = False


            window.Refresh()
            ######################

            if estNumHarvested == -1:
                runAgent = False
                
                #print("finishing task")

                if((currentMillis - prevMillis) >= teleportStartupDelay + teleportDelta):
                    prevMillis = currentMillis
                    print("completing teleport")
                    waiting = False
                    runAgent = True
                    estNumHarvested += 1 
                         

                elif((currentMillis - prevMillis) >= teleportStartupDelay):
                    if waiting == False:
                        waiting = True
                        print("starting teleport")
                        homeTeleport()

            if travelBool:
                #print((len(bankTravelPath) + len(bankItemsPath) + len(mineTravelPath)))
                if((currentMillis - prevMillis) >= 30000):
                    prevMillis = currentMillis
                    print("ping")
                    if bankDepositIterator <= (len(bankTravelPath) + len(bankItemsPath) + len(mineTravelPath)):
                        print("travleing path point: ", bankDepositIterator)
                        bankItems(bankTravelPath, bankItemsPath, mineTravelPath, bankDepositIterator)
                        bankDepositIterator += 1
                    else:
                        print("arrived")
                        travelBool = False

            if runAgent:
                #ret, image_np = cap.read()


                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(frame, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                #print(np.squeeze(scores))    
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=4)

                sboxes = np.squeeze(boxes)
                sscore = np.squeeze(scores)
                sclasses = np.squeeze(classes)
                minScore = 0.5
                locations = []

                for i in range(sboxes.shape[0]):
                    if scores is None or sscore[i] > minScore:
                        #class_name = category_index[classes[0]]['name']
                        #print(six.viewkeys(category_index))
                        if sclasses[i] in six.viewkeys(category_index):
                            box_class_name = category_index[sclasses[i]]['name']
                        box = tuple(sboxes[i].tolist())
                        boxx = spot(box[0], box[1], box[2], box[3], box_class_name)
                        locations.append(boxx)

                #for i in locations:
                #    print("xmin: ", i.xmin, " xmax: ", i.xmax, " ymin: ", i.ymin," ymax: ", i.ymax,)


                if harvestStatus and (currentPos[0] > 0 and currentPos[0] < monSize[0]):
                    currentMillis = time.time() * 1000
                    if((currentMillis - prevMillis) >= timeDelta):
                        prevMillis = currentMillis

                        if len(locations) > 0:
                            for location in locations:
                                if harvestConfig[location.class_name] == True:
                                    if dropInvent == True:
                                        dropFirstItem()

                                    estNumHarvested += 1
                                    #print(location.class_name)
                                    xcord = location.centerX
                                    ycord = location.centerY

                                    xmin = location.xmin
                                    ymin = location.ymin
                                    xmax = location.xmax
                                    ymax = location.ymax

                                    #pyautogui.click(xcord, ycord, duration = np.random.uniform(0.2, 0.6))
                                    pyautogui.moveTo(xcord, ycord, duration = np.random.uniform(0.1, 0.4))
                                    pyautogui.click()
                                    break


                    #pyautogui.moveTo(xmin, ymin, duration = 0)q
                    #pyautogui.moveTo(xmax, ymin, duration = 1)
                    #pyautogui.moveTo(xmax, ymax, duration = 1)
                    #pyautogui.moveTo(xmin, ymax, duration = 1)
                    #pyautogui.moveTo(xmin, ymin, duration = 1)

                    #print("xmin: ", xmin, " xmax: ", xmax, " ymin: ", ymin," ymax: ", ymax)
                    #print("click at:    center x: ", xcord, " center y: ", ycord)


            cv2.imshow('screen', cv2.resize(np.array(image_np), (int(mon2width/2), int(mon2height/2))))
            if cv2.waitKey(25) & 0xFF == ord('p'):
                print("p pressed")
                if runAgent == True:
                    runAgent = False
                elif runAgent == False:
                    runAgent = True
            if cv2.waitKey(25) & 0xFF == ord('h'):
                homeTeleport()
            if cv2.waitKey(25) & 0xFF == ord('t'):
                dropFirstItem()
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break


window.close()