import numpy as np
import cv2
import paho.mqtt.client as mqtt
import os

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, model_from_json


######## 1 ########
#### 0) Have user place hand in bounding rectangle (so that can easily help user figure out size of hand)
#### 1) Filter hand (will be black since glove + difference with bg)
#### 2) Track hand 
#### 3) Use model of hand to track which hand gesture
#### 4) With everything, when hand gesture = fist = (stop), then record starting position when switch to different gesture

######## 2 ########
#### TUTORIAL PART ####
# Train user on different gestures and orientation, give them a score based on CNN accuracy percentage (ie.: tell them to do a fist, blah blah blah)
# Use MQTT, where Joel would just have to subscribe to a topic I listen to on my computer, and that will help me tell the user how to correct for motion
# In this case, I would be doing the iterative learning on gesture correction
#### Pseudocode: 
# if TUTORIAL_DONE = 0: send user to TUTORIAL MODE
# otherwise: post_mode, but look at 1) gesture CNN accuracy 2) IMU rotation 
# separate if-condition for TUTORIAL MODE
    # "do fist" - give rating of accuracy, freeze frame for 1 second, print "this is STOP"   -   processCount = 0
    # "do thumb" - give rating of accuracy, freeze frame for 1 second, print "this is PITCH"   -   processCount = 1
    # "do pinky" - give rating of accuracy, freeze frame for 1 second, print "this is ECHO"   -   processCount = 2
    # "do thumb/pinky" - give rating of accuracy, freeze frame for 1 second, print "this is VOLUME"   -   processCount = 3
    # "CONGRATS YOU PASSED PART 1/2!" - show for 2 seconds                                          -   processCount = 4
    # "Flick wrist to start/stop, raise/lower gesture to adjust parameter" 
    # "Raise PITCH (thumb) by 100" - check if user has correctly raised pitch to 100          
    # "CONGRATS YOU PASSED PART 2/2!" - show for 2 seconds                                           -   processCount = 5
    # "NOW SPEAK COMMAND TO BEGIN POST-PROCESSING YOUR MUSIC" - show for 2 seconds


os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"    


def on_connect(client, userdata, flags,rc):                      # function defining connection
    print("Connected with result code" +str(rc))
    client.subscribe("post")                             # subscribe to topic [CHANGE TOPIC HERE]
    client.subscribe("IMU")
    client.subscribe("virtualPad")

    
inputX = ''                                     # the input string from the IMU - x-axis
inputZ = ''                                     # the input string from the IMU - z-axis
msg = ''            
def hearMsg(client, userdata, message):
    global inputX
    global inputZ
    global msg
    tempString = str(message.topic)
    tempMsg = str(message.payload.decode("utf-8"))
    if (tempString == 'IMU'):
        if ('x' in tempMsg):
            inputX = tempMsg
        elif ('z' in tempMsg):
            inputZ = tempMsg
    elif (tempString == 'virtualPad'):
        msg = tempMsg
        


#####################
#### Definitions ####
#####################

classes = {
    0: 'fist',
    1: 'pinky',
    2: 'thumb',
    3: 'thumb_pinky'
}

classes_adjustment = {
    0: 'stop',
    1: 'volum',
    2: 'pitch',
    3: 'echo'
}

positions = {                                             
    'hand_pose': (15, 400),  # hand pose text
    'fps': (15, 20),         # fps counter
    'adjust': (15, 450),     # type of adjustment text
    'distance': (300, 450),  # distance from original
    'null_pos': (200, 200)   # used as null point for mouse control
}
 

def get_square(image,square_size):                # function to resize hand image: stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
    height,width=image.shape
    if(height>width):
      differ=height
    else:
      differ=width
    differ+=4
    mask = np.zeros((differ,differ), dtype="uint8")   
    x_pos=int((differ-width)/2)
    y_pos=int((differ-height)/2)
    mask[y_pos:y_pos+height,x_pos:x_pos+width]=image[0:height,0:width]
    mask=cv2.resize(mask,(square_size,square_size),interpolation=cv2.INTER_AREA)
    return mask 


def printFixOrientation(frameHand):
    cv2.rectangle(frameHand,(10+325,130),(10+270+375,130+60),(255, 255, 255), cv2.FILLED)
    cv2.putText(frameHand, 'FIX YOUR ORIENTATION', (10+325,180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)



def post_mode(a, b, c):
    
    #################################
    #### Subscribe to MQTT topic ####
    #################################
    
    client = mqtt.Client()                                           # create a client object
    client.on_connect = on_connect                                   # connect to broker  

    client.connect("131.179.36.215", 1883, 60)                        # connect to broker [CHANGE IP ADDRESS HERE]
    client.loop_start()
    
    
    ###############################
    #### Change CNN model here ####
    ###############################

    name = 'hand_model_8'                                 # [CHANGE NAME OF MODEL HERE]
    filename = name + ".hdf5"
    model = load_model(filename, compile=False)
    np.resize(model, 4)
    
    
    ###########################
    #### Initial Variables ####
    ###########################

    cap = cv2.VideoCapture(2)

    bound_done = 0                                  # for determining when user has completed initial step of placing hand in bound_done
    boundX = 10                                    # dimension of bounding box
    boundY = 30
    boundW = 275
    boundH = 420
    xOff = 325

    yes_hand = 0                                    # for determining whether to display hand or not
    prevPos = 0                                     # initializing variable for delta in position
    delta = 0
    adjust = 'stop'

    startFrame = 1                                  # show original frame only at beginning

    TUTORIAL_DONE = a                               # determining if user needs to go to tutorial      # [[[PARAMETER]]]
    checkMistakes = b                               # do not even try checking for mistakes if = 0      # [[[PARAMETER]]]
    countMistakes = c                               # 0 means we are in tutorial mode and do not actually take mistakes into account    # [[[PARAMETER]]]
    processCount = 100                              # count which process to perform
    achieved = 0                                    # achieved = 0 if process not yet achieved, = 1 if process achieved
    timeCount = 0                                   # count time to move on to next displays
    timeCount2 = 0                                  # count time to for other
    timeCount3 = 0                                  # couunt time for iterative learning to tutorial
    activatePart2 = 0                               # to activate the raising/lowering adjustments in tutorial mode
    numPassed = 0                                   # number passed based on time

    timeCountX = 0                                  # to detect if too long above threshold for x-axis
    timeCountY = 0                                  # to detect if too long above threshold for y-axis
    timeCountZ = 0                                  # to detect if too long above threshold for z-axis
    mistakeCount = 0                                # count number of mistakes
    determineY = 0                                  # determine whether y too much rotated (1 if too rotated)
    hand_aspect_ratio = 0.0                         # calculate aspect ratio for y-rotation
    frameOpacity = 0.99

    imageFist = cv2.imread('fist.png', cv2.IMREAD_GRAYSCALE)
    imageFist = get_square(imageFist, 150)
    imageFist = cv2.cvtColor(imageFist, cv2.COLOR_GRAY2BGR)
    imagePalm = cv2.imread('palm.png', cv2.IMREAD_GRAYSCALE)
    imagePalm = get_square(imagePalm, 150)
    imagePalm = cv2.cvtColor(imagePalm, cv2.COLOR_GRAY2BGR)
    imageBad1 = cv2.imread('bad1.png', cv2.IMREAD_GRAYSCALE)
    imageBad1 = get_square(imageBad1, 100)
    imageBad1 = cv2.cvtColor(imageBad1, cv2.COLOR_GRAY2BGR)
    imageBad2 = cv2.imread('bad2.png', cv2.IMREAD_GRAYSCALE)
    imageBad2 = get_square(imageBad2, 100)
    imageBad2 = cv2.cvtColor(imageBad2, cv2.COLOR_GRAY2BGR)
    imageBad3 = cv2.imread('bad3.png', cv2.IMREAD_GRAYSCALE)
    imageBad3 = get_square(imageBad3, 100)
    imageBad3 = cv2.cvtColor(imageBad3, cv2.COLOR_GRAY2BGR)
    imageThumb1 = cv2.imread('thumb1.png', cv2.IMREAD_GRAYSCALE)
    imageThumb1 = get_square(imageThumb1, 100)
    imageThumb1 = cv2.cvtColor(imageThumb1, cv2.COLOR_GRAY2BGR)
    imageFist1 = cv2.imread('fist1.png', cv2.IMREAD_GRAYSCALE)
    imageFist1 = get_square(imageFist1, 100)
    imageFist1 = cv2.cvtColor(imageFist1, cv2.COLOR_GRAY2BGR)
    imagePinky = cv2.imread('pinky.png', cv2.IMREAD_GRAYSCALE)
    imagePinky = get_square(imagePinky, 100)
    imagePinky = cv2.cvtColor(imagePinky, cv2.COLOR_GRAY2BGR)
    imageThPink = cv2.imread('thumb_pinky.png', cv2.IMREAD_GRAYSCALE)
    imageThPink = get_square(imageThPink, 100)
    imageThPink = cv2.cvtColor(imageThPink, cv2.COLOR_GRAY2BGR)

    #######################################################################################################################################
    #######################################################################################################################################
    #######################################################################################################################################

    global msg       # VERY IMPORTANTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT 
    global inputX
    global inputZ
        
    while(True):

        client.on_message = hearMsg                         # defines inputX and inputZ based on what it hears

        if (msg == 'stop'):
            print('end post')
            msg = ''
            client.loop_stop()
            break
        
        _, frame = cap.read()                                           # Capture frame-by-frame
        frame = cv2.medianBlur(frame,5)                                 # apply noise/high-F filter
        frame = cv2.flip(frame, 1)

        ##################################
        #### Filtering + Finding Hand ####
        ##################################

        frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)                              # Convert BGR to HSV
        frameHSV[:,:,2] += 20                                                          # raise brightness of image (cause dark room)
        lower_PINK_glove = np.array([140, 20, 20])                                      # hue, saturation, brightness                 
        upper_PINK_glove = np.array([175, 255, 255])                                                            
        frameHand = cv2.inRange(frameHSV, lower_PINK_glove, upper_PINK_glove) 
        frameHand2 = frameHand.copy() 
        
        _, contours, _ = cv2.findContours(frameHand,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_TC89_L1)       # detect contours
        numContours = range(len(contours))
        centers = []
        numCenters = 0
        frameHand = cv2.cvtColor(frameHand, cv2.COLOR_GRAY2BGR)                                    # for displaying in color
        for i in numContours:                                       # find actual contours
            if cv2.contourArea(contours[i]) < 10000:                # do not consider noise
                continue
            if cv2.contourArea(contours[i]) > 40000:                # do not consider large objects
                continue
            M = cv2.moments(contours[i])                            # calculate moments for each contour
            if M["m00"] != 0:                                       # calculate the center of each object
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0    
            centers.append((cX,cY))
            x,y,w,h = cv2.boundingRect(contours[i])
            cv2.circle(frameHand, centers[-1], 4, (24, 255, 24), -1)      # centroids shown in gray
            cv2.rectangle(frameHand,(x,y),(x+w,y+h),(24, 255, 24), 2)     # rectangle bound
          
        numCenters = len(centers)   
        if (numCenters == 1):                                       # create separate image for hand
            yes_hand = 1
            hand = frameHand2[y:y+h, x:x+w]                       
            hand = get_square(hand, 200)                            # resize + pad image to get same size image
            hand_crop_resized = np.expand_dims(cv2.resize(hand, (54, 54)), axis=0).reshape((1, 54, 54, 1))
            if (x > boundX-30 and (x+w) < (boundX+boundW)+30 and y > boundY-30 and (y+h) < (boundY+boundH)+30):
                bound_done = 1
            else:
                bound_done = 0
        else:
            yes_hand = 0 
            bound_done = 0


        ########################      
        #### Set up display ####
        ########################

        cv2.rectangle(frame,(boundX,boundY),(boundX+boundW,boundY+boundH),(255, 100, 255), 8)           # rectangle bound for initializing bound
        cv2.rectangle(frameHand,(boundX,boundY),(boundX+boundW,boundY+boundH),(255, 100, 255), 8)           # rectangle bound for initializing bound
        if (bound_done == 0):    
            if (startFrame == 1):
                cv2.imshow('frame', frame)
            if (TUTORIAL_DONE == 1):
                cv2.putText(frameHand, "KEEP FIST IN PINK BOX", (0+xOff,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 255), 2)
                cv2.putText(frameHand, "TIL               SHOWS", (0+xOff,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 255), 2)
                cv2.putText(frameHand, "    GREEN SQUARE", (0+xOff,100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


        if (startFrame == 1 and bound_done == 1):                                                       # shut down frame if right after start & hand in bound
            startFrame = 0
            cv2.destroyWindow('frame')


        display = frame.copy() 
        data_display = np.zeros_like(display, dtype=np.uint8)     # Black screen to display data
        
        if (yes_hand == 1 and bound_done == 1):

            ######################################
            #### Identify gesture using model ####
            ######################################

            prediction = model.predict(hand_crop_resized)
            predi = prediction[0].argmax()             # get index of greatest confidence
            gesture = classes[predi]                   # identify gesture
            adjust = classes_adjustment[predi]

            for i, pred in enumerate(prediction[0]):
                # Draw confidence bar for each gesture
                barx = positions['hand_pose'][0]
                bary = 60 + i*60
                bar_height = 20
                bar_length = int(400 * pred) + barx # calculate length of confidence bar

                # Make the most confidence prediction green
                if i == predi:
                    colour = (0, 255, 0)
                else:
                    colour = (0, 0, 255)

                cv2.putText(data_display, "{}: {}".format(classes[i], pred), (positions['hand_pose'][0], 30 + i*60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                cv2.rectangle(data_display, (barx, bary), (bar_length, bary - bar_height), colour, -1, 1)
                cv2.putText(data_display, "hand pose: {}".format(gesture), positions['hand_pose'], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                cv2.putText(data_display, "hand pose: {}".format(adjust), positions['adjust'], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                cv2.putText(data_display, "distance: {}".format(delta), positions['distance'], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

            cv2.imshow('data', data_display)


            ########################################################
            #### Output relative position compared to last fist ####
            ########################################################

            if (TUTORIAL_DONE == 1 or activatePart2 == 1):
                centerY = centers[0][1]
                currPos = centerY

                if (adjust == 'stop'):                   # if 'stop', do not record position changes
                    prevPos = currPos
                    delta = 0
                else:                                    # if any other gesture, record position change
                    delta = prevPos - currPos
                    cv2.putText(frameHand, adjust, (300,prevPos), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 180, 255), 2)
                    overlay = frameHand.copy()
                    cv2.line(overlay, (boundX, prevPos), (boundX+boundW, prevPos), (255, 180, 255), 5)      # line where original position is
                    cv2.rectangle(overlay, (boundX,currPos),(boundX+boundW,currPos+delta),(255, 180, 255), cv2.FILLED)     # filled rectangle for GRAPHICS
                    opacity = 0.9
                    cv2.addWeighted(overlay, opacity, frameHand, 1 - opacity, 0, frameHand)


        ############################
        #### Implement tutorial ####
        ############################

        if (TUTORIAL_DONE == 0):
            cv2.putText(frameHand, 'Tutorial Mode', (boundX+15,boundY+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            if (processCount == 100):        # Telling user preliminaries in wrist rotation
                if (timeCount < 125):
                    cv2.putText(frameHand, '1)', (10+xOff,210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 24), 2)
                    cv2.putText(frameHand, 'Face palm fwd in box', (10+xOff,240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 24), 2)
                    cv2.putText(frameHand, '(move hand until', (10+xOff,270), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 24), 2)
                    cv2.putText(frameHand, '              appears)', (10+xOff,300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 24), 2)
                    cv2.putText(frameHand, 'green square', (10+xOff,300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    frameHand[310:460, 30+xOff:180+xOff] = imagePalm
                    timeCount += 1
                elif (timeCount >= 125) and (yes_hand == 0 or bound_done == 0):
                    cv2.putText(frameHand, '1)', (10+xOff,210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 24), 2)
                    cv2.putText(frameHand, 'Face palm fwd in box', (10+xOff,240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 24), 2)
                    cv2.putText(frameHand, '(move hand until', (10+xOff,270), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 24), 2)
                    cv2.putText(frameHand, '              appears)', (10+xOff,300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 24), 2)
                    cv2.putText(frameHand, 'green square', (10+xOff,300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    frameHand[310:460, 30+xOff:180+xOff] = imagePalm
                else:
                    processCount += 1
                    timeCount = 0
            elif (processCount == 101):        # Telling user preliminaries in wrist rotation
                if (timeCount < 125):
                    cv2.putText(frameHand, '2)', (10+xOff,210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 24), 2)
                    cv2.putText(frameHand, 'Now hold a fist like this:', (10+xOff,240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 24), 2)
                    frameHand[290:440, 30+xOff:180+xOff] = imageFist
                    timeCount += 1
                elif (timeCount >= 125) and (yes_hand == 0 or bound_done == 0 or gesture != 'fist'):
                    cv2.putText(frameHand, '2)', (10+xOff,210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 24), 2)
                    cv2.putText(frameHand, 'Now hold a fist like this:', (10+xOff,240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 24), 2)
                    frameHand[290:440, 30+xOff:180+xOff] = imageFist
                else:
                    processCount += 1
                    timeCount = 0
            elif (processCount == 102):        # Telling user preliminaries in wrist rotation
                if (timeCount < 150):
                    cv2.putText(frameHand, 'This is your base gesture', (10+xOff,210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 24), 2)
                    frameHand[300:450, 30+xOff:180+xOff] = imageFist
                    timeCount += 1
                else:
                    processCount += 1
                    timeCount = 0
            elif (processCount == 103):        # Telling user preliminaries in wrist rotation
                if (timeCount < 200):
                    cv2.putText(frameHand, 'Maintain this orientation', (10+xOff,210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 24), 2)
                    cv2.putText(frameHand, 'at all times', (10+xOff,240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 24), 2)
                    cv2.putText(frameHand, '(in other words, make sure ur', (10+xOff,270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 24), 2)
                    cv2.putText(frameHand, 'fist is facing camera as such)', (10+xOff,290), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 24), 2)
                    frameHand[300:450, 30+xOff:180+xOff] = imageFist
                    timeCount += 1
                else:
                    processCount += 1
                    timeCount = 0
            elif (processCount == 104):       
                if (timeCount < 250):
                    cv2.putText(frameHand, 'Bad orientation examples:', (10+xOff,210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 24), 2)
                    cv2.putText(frameHand, '(in other words, the camera', (10+xOff,240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 24), 2)
                    cv2.putText(frameHand, 'should not be seeing these!)', (10+xOff,260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 24), 2)
                    frameHand[300:400, -30+xOff:70+xOff] = imageBad1
                    frameHand[300:400, 80+xOff:180+xOff] = imageBad2
                    frameHand[300:400, 190+xOff:290+xOff] = imageBad3
                    cv2.putText(frameHand, 'x', (300,310), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.putText(frameHand, 'x', (410,310), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.putText(frameHand, 'x', (520,310), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    timeCount += 1
                else:
                    processCount += 1
                    timeCount = 0
            elif (processCount == 105):        # Telling user preliminaries in wrist rotation
                if (timeCount < 200):
                    cv2.putText(frameHand, 'Otherwise, these (in red)', (10+xOff,210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 24), 2)
                    cv2.putText(frameHand, 'will appear', (10+xOff,240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 24), 2)
                    cv2.putText(frameHand, 'gesture mistakes: 1/3', (10+xOff,460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    printFixOrientation(frameHand)
                    timeCount += 1
                else:
                    processCount += 1
                    timeCount = 0
            elif (processCount == 106):        # Telling user preliminaries in wrist rotation
                if (timeCount < 150):
                    cv2.putText(frameHand, 'If get 3 mistakes,', (10+xOff,220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 24), 2)
                    cv2.putText(frameHand, 'u will be sent back', (10+xOff,250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 24), 2)
                    cv2.putText(frameHand, 'to tutorial', (10+xOff,280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 24), 2)
                    cv2.putText(frameHand, 'gesture mistakes: 1/3', (10+xOff,460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    printFixOrientation(frameHand)
                    timeCount += 1
                else:
                    processCount += 1
                    timeCount = 0
            elif (processCount == 107):        
                if (timeCount < 75):
                    cv2.putText(frameHand, 'NOW ONTO', (10+xOff,300), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (24, 255, 24), 3)
                    cv2.putText(frameHand, 'PART 1 OF 2', (10+xOff,340), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (24, 255, 24), 3)
                    timeCount += 1
                else:
                    processCount = -1
                    timeCount = 0
                    mistakeCount = 0
                    checkMistakes = 1
                    countMistakes = 1
            #########            
            #########            
            #########
            elif (processCount == -1):      # THUMB 
                if (achieved == 0 and (yes_hand == 0 or bound_done == 0 or gesture != 'thumb')):
                    cv2.putText(frameHand, '1)Extend thumb', (10+xOff,220), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 24), 3)
                    cv2.putText(frameHand, '  out of fist', (10+xOff,250), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 24), 3)
                    frameHand[300:400, 80+xOff:180+xOff] = imageThumb1
                else:
                    if (timeCount <= 75):
                        achieved = 1
                        cv2.putText(frameHand, 'NICE!', (10+xOff,300), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (24, 255, 24), 3)
                        cv2.putText(frameHand, 'thumb=pitch', (10+xOff,350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (24, 255, 24), 3)
                        timeCount += 1
                    else:
                        achieved = 0
                        processCount += 1
                        timeCount = 0              
            elif (processCount == 0):      # FIST 
                if (achieved == 0 and (yes_hand == 0 or bound_done == 0 or gesture != 'fist')):
                    cv2.putText(frameHand, '2)Hold a fist again', (10+xOff,220), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 24), 3)
                    frameHand[300:400, 80+xOff:180+xOff] = imageFist1
                else:
                    if (timeCount <= 75):
                        achieved = 1
                        cv2.putText(frameHand, 'NICE!', (10+xOff,300), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (24, 255, 24), 3)
                        cv2.putText(frameHand, 'fist=stop', (10+xOff,350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (24, 255, 24), 3)
                        timeCount += 1
                    else:
                        achieved = 0
                        processCount += 1
                        timeCount = 0   
            elif (processCount == 1):      # THUMB_PINKY 
                if (achieved == 0 and (yes_hand == 0 or bound_done == 0 or gesture != 'thumb_pinky')):
                    cv2.putText(frameHand, '3)Extend both', (10+xOff,220), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 24), 3)
                    cv2.putText(frameHand, '  thumb+pinky', (10+xOff,250), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 24), 3)
                    frameHand[300:400, 80+xOff:180+xOff] = imageThPink
                else:
                    if (timeCount <= 75):
                        achieved = 1
                        cv2.putText(frameHand, 'NICE!', (10+xOff,300), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (24, 255, 24), 3)
                        cv2.putText(frameHand, 'thumb+pinky=echo', (10+xOff,350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (24, 255, 24), 3)
                        timeCount += 1
                    else:
                        achieved = 0
                        processCount += 1
                        timeCount = 0 
            elif (processCount == 2):      # PINKY 
                if (achieved == 0 and (yes_hand == 0 or bound_done == 0 or gesture != 'pinky')):
                    cv2.putText(frameHand, '4)Extend just pinky', (10+xOff,220), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 24), 3)
                    frameHand[300:400, 80+xOff:180+xOff] = imagePinky
                else:
                    if (timeCount <= 75):
                        achieved = 1
                        cv2.putText(frameHand, 'NICE!', (10+xOff,300), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (24, 255, 24), 3)
                        cv2.putText(frameHand, 'pinky=volume', (10+xOff,350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (24, 255, 24), 3)
                        timeCount += 1
                    else:
                        achieved = 0
                        processCount += 1
                        timeCount = 0
            elif (processCount == 3):     # TIMED
                if (timeCount <= 250):
                    cv2.putText(frameHand, 'Remember:', (10+xOff,270), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 24), 3)
                    cv2.putText(frameHand, '1) fist = stop', (10+xOff,300), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 24), 2)
                    cv2.putText(frameHand, '2) thumb = pitch', (10+xOff,320), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 24), 2)
                    cv2.putText(frameHand, '3) pinky = volume', (10+xOff,340), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 24), 2)
                    cv2.putText(frameHand, '4) thumb+pinky = echo', (10+xOff,360), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 24), 2)
                    timeCount += 1
                else:
                    achieved = 0
                    processCount = 4
                    timeCount = 0
                    mistakeCount = 0
                    countMistakes = 1
            #########            
            #########            
            #########
            elif (processCount == 4):     # TIMED
                if (timeCount <= 100):
                    cv2.putText(frameHand, 'NOW U WILL', (10+xOff,300), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 24), 3)
                    cv2.putText(frameHand, 'BE TIMED', (10+xOff,350), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 24), 3)
                    timeCount += 1
                else:
                    processCount += 1
                    timeCount = 0
                    timeCount2 = 80
                    barxTime = 20+xOff                # create time bar to show decreasing time
                    baryTime = 430
                    barheightTime = 20
            elif (processCount == 5):       
                if (achieved == 0) and (timeCount2 > 0):
                    cv2.putText(frameHand, '1)Do volume', (10+xOff,300), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 24), 3)
                    cv2.putText(frameHand, 'Time left: {}'.format(timeCount2), (10+xOff,400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 24), 3)
                    barlengthTime = 2*timeCount2 + barxTime 
                    cv2.rectangle(frameHand, (barxTime, baryTime), (barlengthTime, baryTime - barheightTime), (200,200,24), -1, 1)       # create time bar to show decreasing time
                    if (yes_hand == 1 and bound_done == 1 and gesture == 'pinky'):
                        achieved = 1
                        numPassed += 1
                    timeCount2 -= 1
                else:
                    if (timeCount <= 75):
                        if (achieved == 1):
                            cv2.putText(frameHand, 'NICE!', (10+xOff,300), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (24, 255, 24), 3)
                            cv2.putText(frameHand, 'Pass: {}/5'.format(numPassed), (10+xOff,350), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (24, 255, 24), 3)
                            cv2.putText(frameHand, 'volume=pinky', (10+xOff,400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (24, 255, 24), 2)
                        else:
                            cv2.putText(frameHand, 'NOPE', (10+xOff,300), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
                            cv2.putText(frameHand, 'volume=pinky', (10+xOff,350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        timeCount += 1
                    else:
                        processCount += 1
                        timeCount = 0
                        timeCount2 = 65
                        achieved = 0
            elif (processCount == 6):      
                if (achieved == 0) and (timeCount2 > 0):
                    cv2.putText(frameHand, '2)Do pitch', (10+xOff,300), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 24), 3)
                    cv2.putText(frameHand, 'Time left: {}'.format(timeCount2), (10+xOff,400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 24), 3)
                    barlengthTime = 2*timeCount2 + barxTime 
                    cv2.rectangle(frameHand, (barxTime, baryTime), (barlengthTime, baryTime - barheightTime), (200,200,24), -1, 1)       # create time bar to show decreasing time
                    if (yes_hand == 1 and bound_done == 1 and gesture == 'thumb'):
                        achieved = 1
                        numPassed += 1
                    timeCount2 -= 1
                else:
                    if (timeCount <= 75):
                        if (achieved == 1):
                            cv2.putText(frameHand, 'NICE!', (10+xOff,300), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (24, 255, 24), 3)
                            cv2.putText(frameHand, 'Pass: {}/5'.format(numPassed), (10+xOff,350), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (24, 255, 24), 3)
                            cv2.putText(frameHand, 'pitch=thumb', (10+xOff,400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (24, 255, 24), 2)
                        else:
                            cv2.putText(frameHand, 'NOPE', (10+xOff,300), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
                            cv2.putText(frameHand, 'pitch=thumb', (10+xOff,350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        timeCount += 1
                    else:
                        processCount += 1
                        timeCount = 0
                        timeCount2 = 65
                        achieved = 0
            elif (processCount == 7):      
                if (achieved == 0) and (timeCount2 > 0):
                    cv2.putText(frameHand, '3)Do stop', (10+xOff,300), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 24), 3)
                    cv2.putText(frameHand, 'Time left: {}'.format(timeCount2), (10+xOff,400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 24), 3)
                    barlengthTime = 2*timeCount2 + barxTime 
                    cv2.rectangle(frameHand, (barxTime, baryTime), (barlengthTime, baryTime - barheightTime), (200,200,24), -1, 1)       # create time bar to show decreasing time
                    if (yes_hand == 1 and bound_done == 1 and gesture == 'fist'):
                        achieved = 1
                        numPassed += 1
                    timeCount2 -= 1
                else:
                    if (timeCount <= 75):
                        if (achieved == 1):
                            cv2.putText(frameHand, 'NICE!', (10+xOff,300), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (24, 255, 24), 3)
                            cv2.putText(frameHand, 'Pass: {}/5'.format(numPassed), (10+xOff,350), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (24, 255, 24), 3)
                            cv2.putText(frameHand, 'stop=fist', (10+xOff,400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (24, 255, 24), 2)
                        else:
                            cv2.putText(frameHand, 'NOPE', (10+xOff,300), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
                            cv2.putText(frameHand, 'stop=fist', (10+xOff,350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        timeCount += 1
                    else:
                        processCount += 1
                        timeCount = 0
                        timeCount2 = 50
                        achieved = 0
            elif (processCount == 8):      
                if (achieved == 0) and (timeCount2 > 0):
                    cv2.putText(frameHand, '4)Do echo', (10+xOff,300), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 24), 3)
                    cv2.putText(frameHand, 'Time left: {}'.format(timeCount2), (10+xOff,400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 24), 3)
                    barlengthTime = 2*timeCount2 + barxTime 
                    cv2.rectangle(frameHand, (barxTime, baryTime), (barlengthTime, baryTime - barheightTime), (200,200,24), -1, 1)       # create time bar to show decreasing time
                    if (yes_hand == 1 and bound_done == 1 and gesture == 'thumb_pinky'):
                        achieved = 1
                        numPassed += 1
                    timeCount2 -= 1
                else:
                    if (timeCount <= 75):
                        if (achieved == 1):
                            cv2.putText(frameHand, 'NICE!', (10+xOff,300), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (24, 255, 24), 3)
                            cv2.putText(frameHand, 'Pass: {}/5'.format(numPassed), (10+xOff,350), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (24, 255, 24), 3)
                            cv2.putText(frameHand, 'echo=thumb+pinky', (10+xOff,400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (24, 255, 24), 2)
                        else:
                            cv2.putText(frameHand, 'NOPE', (10+xOff,300), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
                            cv2.putText(frameHand, 'echo=thumb+pinky', (10+xOff,350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        timeCount += 1
                    else:
                        processCount += 1
                        timeCount = 0
                        timeCount2 = 50
                        achieved = 0
            elif (processCount == 9):      
                if (achieved == 0) and (timeCount2 > 0):
                    cv2.putText(frameHand, '5)Do pitch', (10+xOff,300), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 24), 3)
                    cv2.putText(frameHand, 'Time left: {}'.format(timeCount2), (10+xOff,400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 24), 3)
                    barlengthTime = 2*timeCount2 + barxTime 
                    cv2.rectangle(frameHand, (barxTime, baryTime), (barlengthTime, baryTime - barheightTime), (200,200,24), -1, 1)       # create time bar to show decreasing time
                    if (yes_hand == 1 and bound_done == 1 and gesture == 'thumb'):
                        achieved = 1
                        numPassed += 1
                    timeCount2 -= 1
                else:
                    if (timeCount <= 75):
                        if (achieved == 1):
                            cv2.putText(frameHand, 'NICE!', (10+xOff,300), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (24, 255, 24), 3)
                            cv2.putText(frameHand, 'Pass: {}/5'.format(numPassed), (10+xOff,350), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (24, 255, 24), 3)
                            cv2.putText(frameHand, 'pitch=thumb', (10+xOff,400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (24, 255, 24), 2)
                        else:
                            cv2.putText(frameHand, 'NOPE', (10+xOff,300), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
                            cv2.putText(frameHand, 'pitch=thumb', (10+xOff,350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        timeCount += 1
                    else:
                        processCount += 1
                        timeCount = 0
                        achieved = 0
            elif (processCount == 10):      
                if (timeCount <= 100):
                    if (numPassed >= 4):
                        achieved = 1
                        cv2.putText(frameHand, 'U PASSED', (10+xOff,300), cv2.FONT_HERSHEY_DUPLEX, 1, (24, 255, 24), 3)
                        cv2.putText(frameHand, 'PART 1 OF 2', (10+xOff,340), cv2.FONT_HERSHEY_DUPLEX, 1, (24, 255, 24), 3)
                        cv2.putText(frameHand, 'Score: {}/5'.format(numPassed), (10+xOff,390), cv2.FONT_HERSHEY_DUPLEX, 0.8, (180, 180, 24), 2)
                    else:
                        achieved = 0
                        cv2.putText(frameHand, 'U NO PASS', (10+xOff,300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        cv2.putText(frameHand, 'REPEAT!', (10+xOff,340), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        cv2.putText(frameHand, 'Score: {}/5'.format(numPassed), (10+xOff,390), cv2.FONT_HERSHEY_DUPLEX, 0.8, (180, 180, 24), 2)
                    timeCount += 1
                else:
                    if (achieved == 1):
                        processCount = 20
                        activatePart2 = 1
                    else:
                        processCount = 4
                    timeCount = 0
                    numPassed = 0
                    achieved = 0
            #########            
            #########            
            #########
            elif (processCount == 20):      # PART 2 OF 2
                if (timeCount <= 100):
                    cv2.putText(frameHand, 'ONTO', (20+xOff,300), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 24), 3)
                    cv2.putText(frameHand, 'PART 2 OF 2', (20+xOff,350), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 24), 3)
                    timeCount += 1
                else:
                    processCount += 1
                    timeCount = 0
                    activatePart2 = 1
            elif (processCount == 21):      # PART 2 - ADJUSTING PARAMETERS
                if (timeCount <= 140):
                    cv2.putText(frameHand, 'HOW TO ADJUST:', (20+xOff,200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 24), 3)
                    cv2.putText(frameHand, '1)Hold fist to start', (20+xOff,250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 24), 2)
                    frameHand[300:400, 80+xOff:180+xOff] = imageFist1
                    timeCount += 1
                else:
                    processCount += 1
                    timeCount = 0
            elif (processCount == 22):      
                if (timeCount <= 140):
                    cv2.putText(frameHand, '2)Change gesture to adjust', (20+xOff,250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 24), 2)
                    cv2.putText(frameHand, '  parameter (ie.: pitch)', (20+xOff,280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 24), 2)
                    frameHand[300:400, 80+xOff:180+xOff] = imageThumb1
                    timeCount += 1
                else:
                    processCount += 1
                    timeCount = 0
            elif (processCount == 23):      
                if (achieved == 0 and (yes_hand == 0 or bound_done == 0 or gesture != 'thumb' or delta <= 100)):
                    cv2.putText(frameHand, '3)Now raise pitch', (20+xOff,250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 24), 2)
                    cv2.putText(frameHand, '  by a decent amt', (20+xOff,280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 24), 2)
                    cv2.putText(frameHand, '  (raise hand up)', (20+xOff,310), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 24), 2)
                else:
                    if (timeCount <= 50):
                        achieved = 1
                        cv2.putText(frameHand, 'NICE!', (20+xOff,300), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (24, 255, 24), 3)
                        timeCount += 1
                    else:
                        achieved = 0
                        processCount += 1
                        timeCount = 0
            elif (processCount == 24):      # PART 2 - ADJUSTING PARAMETERS
                if (timeCount <= 140):
                    cv2.putText(frameHand, '4)Hold back fist', (20+xOff,250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 24), 2)
                    cv2.putText(frameHand, '  to stop', (20+xOff,280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 24), 2)
                    frameHand[300:400, 80+xOff:180+xOff] = imageFist1
                    timeCount += 1
                else:
                    processCount += 1
                    timeCount = 0
            elif (processCount == 25):      
                if (achieved == 0 and (yes_hand == 0 or bound_done == 0 or gesture != 'thumb_pinky' or delta > -100)):
                    cv2.putText(frameHand, 'PRACTICE:', (20+xOff,220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 24), 2)
                    cv2.putText(frameHand, 'Decrease echo', (20+xOff,250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 24), 2)
                    cv2.putText(frameHand, 'by a decent amt', (20+xOff,280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 24), 2)
                else:
                    if (timeCount <= 100):
                        achieved = 1
                        cv2.putText(frameHand, 'NICE!', (20+xOff,300), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (24, 255, 24), 3)
                        timeCount += 1
                    else:
                        achieved = 0
                        processCount += 1
                        activatePart2 = 0
                        timeCount = 0
            elif (processCount == 26):      # PASS TUTORIAL
                if (timeCount <= 150):
                    cv2.putText(frameHand, 'U PASSED', (20+xOff,300), cv2.FONT_HERSHEY_DUPLEX, 1, (24, 255, 24), 3)
                    cv2.putText(frameHand, 'TUTORIAL', (20+xOff,340), cv2.FONT_HERSHEY_DUPLEX, 1, (24, 255, 24), 3)
                    timeCount += 1
                else:
                    processCount = 0
                    achieved = 0
                    timeCount = -1       # value of -1 to signal finish tutorial 
                    mistakeCount = 0
                    countMistakes = 1
                    checkMistakes = 1
                    TUTORIAL_DONE = 1



        ######################################
        #### Iterative Learning Detection ####
        ######################################

        #### Considering (x,y) axis is the frame axises (then z would be pointing through frame)
        #### Check whether motion within threshold, only if 
        #### If outside axis threshold for set amount of time, then count that as 1 mistake, tell them to correct
        #### If number of mistakes >= 3, set TUTORIAL_DONE = 0 + set processCount = 4
        #### 1) around x-axis (nodding) - IMU - "x_too_front" "x_too_back"
        #### 2) around y-axis (wristing) - image aspect ratio 
        #### 3) around z-axis (left-right) - IMU - "z_too_right" "z_too_left"
        #### 
        #### This means in TUTORIAL, must integrate showing whether out of axis threshold
        ####
        #### When receive message, must 
        #### 1) decipher between x and z strings
        #### 2) JOEL IMU must send x_none and z_none if no errors

        if (checkMistakes == 1 and yes_hand == 1 and bound_done == 1):
            if (inputX == 'x_too_front'):                               # around x-axis (nodding) - IMU - "x_too_front" "x_too_back"
                timeCountX += 1
                if (timeCountX > 75):
                    mistakeCount += 1
                    timeCountX = 0
                #cv2.rectangle(frameHand,(5,130),(5+230+xOff,130+30),(255, 255, 255), cv2.FILLED)
                #cv2.putText(frameHand, 'BEND HAND BACK', (10,150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                printFixOrientation(frameHand)
            elif (inputX == 'x_too_back'):
                timeCountX += 1
                if (timeCountX > 75):
                    mistakeCount += 1
                    timeCountX = 0
                #cv2.rectangle(frameHand,(5+xOff,130),(10+230+xOff,130+30),(255, 255, 255), cv2.FILLED)
                #cv2.putText(frameHand, 'BEND HAND FORWARD', (10+xOff,150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                printFixOrientation(frameHand)
            else:
                timeCountX = 0

            if (inputZ == 'z_too_right'):                               # around z-axis (left-right) - IMU - "z_too_right" "z_too_left"
                timeCountZ += 1
                if (timeCountZ > 75):
                    mistakeCount += 1
                    timeCountZ = 0
                #cv2.rectangle(frameHand,(5+xOff,160),(5+230+xOff,160+30),(255, 255, 255), cv2.FILLED)
                #cv2.putText(frameHand, 'BEND HAND LEFT', (10+xOff,180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                printFixOrientation(frameHand)
            elif (inputZ == 'z_too_left'):
                timeCountZ += 1
                if (timeCountZ > 75):
                    mistakeCount += 1
                    timeCountZ = 0
                #cv2.rectangle(frameHand,(5+xOff,160),(5+230+xOff,160+30),(255, 255, 255), cv2.FILLED)
                #cv2.putText(frameHand, 'BEND HAND RIGHT', (10+xOff,180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                printFixOrientation(frameHand)
            else: 
                timeCountZ = 0


            if (yes_hand == 1 and bound_done == 1):
                hand_aspect_ratio = float(w)/float(h)                       # aspect ratio of hand image - width/height
                #cv2.putText(frameHand, 'aspect:{}'.format(hand_aspect_ratio), (10+xOff,450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 3)
                if (gesture == 'fist' and hand_aspect_ratio <= 0.69):        # custom thresholds for each gesture (WHERE DATA WILL COME FROM)
                    determineY = 1
                elif (gesture == 'pinky' and hand_aspect_ratio <= 0.60):
                    determineY = 1
                elif (gesture == 'thumb' and hand_aspect_ratio <= 0.91):
                    determineY = 1
                elif (gesture == 'thumb_pinky' and hand_aspect_ratio <= 0.76):
                    determineY = 1
                else:
                    determineY = 0
            else:
                determineY = 0

            if (determineY == 1):                                       # around y-axis (wristing) - image aspect ratio 
                timeCountY += 1
                if (timeCountY > 75):
                    mistakeCount += 1
                    timeCountY = 0
                #cv2.rectangle(frameHand,(5+xOff,190),(5+230+xOff,190+30),(255, 255, 255), cv2.FILLED)
                #cv2.putText(frameHand, 'FACE WRIST STRAIGHT', (10+xOff,210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                printFixOrientation(frameHand)
            else: 
                timeCountY = 0


            if (mistakeCount > 0):
                cv2.putText(frameHand, 'gesture mistakes: {}/3'.format(mistakeCount), (10+xOff,460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if (mistakeCount >= 3  and countMistakes == 1):                                     # if too many mistakes, send to TUTORIAL
                if (timeCount3 < 200):
                    cv2.rectangle(frameHand,(0,100),(10+700,100+400),(255, 255, 255), cv2.FILLED)
                    cv2.putText(frameHand, 'FIX YOUR ORIENTATION', (100,190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    cv2.putText(frameHand, 'MUST GO BACK TO TUTORIAL', (100,240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    frameHand[290:440, 30:180] = imageFist
                    cv2.putText(frameHand, 'GOOD', (40,300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    frameHand[300:400, -30+xOff:70+xOff] = imageBad1
                    frameHand[300:400, 80+xOff:180+xOff] = imageBad2
                    frameHand[300:400, 190+xOff:290+xOff] = imageBad3
                    cv2.putText(frameHand, 'x', (300,310), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.putText(frameHand, 'x', (410,310), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.putText(frameHand, 'x', (520,310), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    timeCount3 += 1
                else:
                    TUTORIAL_DONE = 0
                    processCount = 4
                    numPassed = 0
                    achieved = 0
                    mistakeCount = 0
                    timeCount3 = 0

        else:      # if checkMistakes == 0, so don't check mistakes at all
            mistakeCount = 0
            timeCountX = 0
            timeCountY = 0
            timeCountZ = 0
            timeCount3 = 0


        #########################
        #### Print & publish ####
        #########################

        overlay = frameHand.copy()
        cv2.addWeighted(overlay, frameOpacity, frame, 0.1, 0, frameHand)   
        cv2.imshow('frameHand', frameHand)        

        if (mistakeCount >= 3  and countMistakes == 1 and checkMistakes == 1) or (TUTORIAL_DONE == 0):
            client.publish("post", "in tutorial")                     # to let Unity know that we are in tutorial mode
        else:
            if (timeCount == -1):
                client.publish("post", "exit tutorial")
                timeCount = 0
            else:
                client.publish("post", "adjustment: " + adjust + " " + str(delta))

        if cv2.waitKey(1) & 0xFF == ord('q'):        # press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()
    return TUTORIAL_DONE




