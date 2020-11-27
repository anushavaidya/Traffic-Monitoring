import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from collections import deque
from PIL import Image
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/Data_6.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', './output.avi', 'path to output video')
flags.DEFINE_string('output_format', 'MJPG', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')


pts = [deque(maxlen=30) for _ in range(9999)]



def main(_argv):
    # Definition of the parameters
    pts = [deque(maxlen=30) for _ in range(9999)]

    i =0
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0

    #Person ID
    new_person_id=0
    arr_personid = []
    arr_new_person_id=[]
    counter = 1
    
    #Motorbike ID
    new_motorbike_id=0
    arr_motorbikeid = []
    arr_new_motorbike_id=[]
    counter = 1

    #Car ID
    new_car_id=0
    arr_carid = []
    arr_new_car_id=[]

    #Truck ID
    new_truck_id=0
    arr_truckid = []
    arr_new_truck_id=[]

    #Bus ID 
    new_bus_id=0
    arr_busid = []
    arr_new_bus_id=[]

    #Traffic Light ID 
    new_tl_id=0
    arr_tlid = []
    arr_new_tl_id=[]

    #Bicycle ID 
    new_bicycle_id=0
    arr_bicycleid = []
    arr_new_bicycle_id=[]

    
    #Parameter 
    max_cosine_distance = 0.5
    nn_budget = None
    nms_max_overlap = 1.0
    zone5_totalcounter = []
    zone5_carcounter = []
    zone5_truckcounter = []
    zone5_buscounter = []
    zone5_motorbikecounter = []
    zone5_bicyclecounter =[]
    zone4_totalcounter = []
    zone4_carcounter = []
    zone4_truckcounter = []
    zone4_buscounter = []
    zone4_motorbikecounter = []
    zone4_bicyclecounter =[]
    zone3_totalcounter = []
    zone3_carcounter = []
    zone3_truckcounter = []
    zone3_buscounter = []
    zone3_motorbikecounter = []
    zone3_bicyclecounter =[]
    zone2_totalcounter = []
    zone2_carcounter = []
    zone2_truckcounter = []
    zone2_buscounter = []
    zone2_motorbikecounter = []
    zone2_bicyclecounter =[]
    zone1_totalcounter = []
    zone1_carcounter = []
    zone1_truckcounter = []
    zone1_buscounter = []
    zone1_motorbikecounter = []
    zone1_bicyclecounter =[]
    pedzone_totalcounter = []
    

    
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_num = 0
    isClosed=1
    # while video is running
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break

        
        frame_num +=1
        #print('Frame #: ', frame_num)
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()
        zone5_currentcounter=0
        zone4_currentcounter=0
        zone3_currentcounter=0
        zone2_currentcounter=0
        zone1_currentcounter=0
        pedzone_currentcounter=0

        # run detections on tflite if flag is set
        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        #allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to customize tracker for only people)
        allowed_classes = ['person','car','truck','motorbike','bus','bicycle']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        count = len(names)
        if FLAGS.count:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        for d in detections:
            box = d.tlwh
            class_id = d.class_name
            counting_point1 = (int(box[0]+(box[2]/2)),int(box[1]+box[3]))
            if (0 <= counting_point1[0] <= ((counting_point1[1]-590)*(1/-0.17857))) and (473<=counting_point1[1]<=532):
                if (class_id == 'car') or (class_id == 'bus') or (class_id == 'motorbike') or (class_id == 'truck') or (class_id == 'bicycle'):
                    zone4_currentcounter += 1
            if (((counting_point1[1]-754.167)/-.4167) <= counting_point1[0] <= ((counting_point1[1]+59.375)/.625)) and (475<=counting_point1[1]<=500):
                if (class_id == 'car') or (class_id == 'bus') or (class_id == 'motorbike') or (class_id == 'truck') or (class_id == 'bicycle'):
                    zone3_currentcounter += 1
            if (((counting_point1[1]+59.375)/0.625)<=counting_point1[0] <= ((counting_point1[1]-72.916)/.4167)) and (475<=counting_point1[1]<=500):
                if (class_id == 'car') or (class_id == 'bus') or (class_id == 'motorbike') or (class_id == 'truck') or (class_id == 'bicycle'):
                    zone2_currentcounter +=1
            if (((counting_point1[1]-72.916)/.4167)<=counting_point1[0] <= ((counting_point1[1]-309.1)/.1515)) and (475<=counting_point1[1]<=500):
                if (class_id == 'car') or (class_id == 'bus') or (class_id == 'motorbike') or (class_id == 'truck') or (class_id == 'bicycle'):
                    zone1_currentcounter += 1
            if (450<=counting_point1[0] <= 1280) and (counting_point1[1]>=620):
                if (class_id == 'car') or (class_id == 'bus') or (class_id == 'motorbike') or (class_id == 'truck') or (class_id == 'bicycle'):
                    zone5_currentcounter += 1
            if (counting_point1[1]>=480) and (((counting_point1[1]-520)/-.0901)<=counting_point1[0]<=((counting_point1[1]-620)/-.22047)):
                if (class_id == 'person'):
                    pedzone_currentcounter += 1
                

        #pedzone
        ped_points = np.array([[0,620],[635,480],[440,480],[0,520]],np.int32).reshape((-1, 1, 2)) 
        ped_zone = cv2.polylines(frame,[ped_points],isClosed,(0,0,255),2)

        #zone 5
        z5_points = np.array([[450,950],[1280,950],[1280,620],[450,620]],np.int32).reshape((-1, 1, 2)) 
        zone5 = cv2.polylines(frame,[z5_points],isClosed,(100,50,50),2)
        
        #zone 4
        z4_points = np.array([[0,532],[325,532],[655,473],[0,473]],np.int32).reshape((-1, 1, 2)) 
        zone4 = cv2.polylines(frame,[z4_points],isClosed,(255,0,0),2)
       
        #zone 3
        z3_points = np.array([[610,500],[895,500],[855,475],[670,475]],np.int32).reshape((-1, 1, 2)) 
        zone3 = cv2.polylines(frame,[z3_points],isClosed,(255,255,0),2)
    
        #zone 2
        z2_points = np.array([[895,500],[1025,500],[965,475],[855,475]],np.int32).reshape((-1, 1, 2)) 
        zone2 = cv2.polylines(frame,[z2_points],isClosed,(0,255,0),2)
    
        #zone1 
        z1_points = np.array([[1025,500],[1260,500],[1095,475],[965,475]],np.int32).reshape((-1, 1, 2)) 
        zone1 = cv2.polylines(frame,[z1_points],isClosed,(255,0,255),2)

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
            obj_id = track.track_id
            if class_name not in allowed_classes:
                continue
        # draw bbox on screen


            bottom_left=(int(bbox[0]),int(bbox[3]))
            bottom_right=(int(bbox[2]),int(bbox[3]))
        
            counting_point = (int(bbox[0]+((bbox[2]-bbox[0])/2)),int(bbox[3]))
            #cv2.circle(frame,counting_point,5,(255,255,255),-1)

            if class_name == 'person':
                if int(obj_id) not in arr_personid:
                    new_person_id+=1
                    arr_new_person_id.append(new_person_id)
                    arr_personid.append(int(obj_id))
                                   
                for i in range(len(arr_new_person_id)):
                    a = arr_personid[i]                   
                    if obj_id == a:
                        counter = arr_new_person_id[i]
               

                color = colors[1]
                color = [i * 255 for i in color]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                cv2.putText(frame, class_name + "-" + str(int(counter)),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

            if class_name == 'motorbike':
                if int(obj_id) not in arr_motorbikeid:
                    new_motorbike_id+=1
                    arr_new_motorbike_id.append(new_motorbike_id)
                    arr_motorbikeid.append(int(obj_id))
                                   
                for i in range(len(arr_new_motorbike_id)):
                    a = arr_motorbikeid[i]                   
                    if obj_id == a:
                        counter = arr_new_motorbike_id[i]
               

                color = colors[3]
                color = [i * 255 for i in color]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                cv2.putText(frame, class_name + "-" + str(int(counter)),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)


            if class_name == 'car':
                if int(obj_id) not in arr_carid:
                    new_car_id+=1
                    arr_new_car_id.append(new_car_id)
                    arr_carid.append(int(obj_id))
                                   
                for i in range(len(arr_new_car_id)):
                    a = arr_carid[i]                   
                    if obj_id == a:
                        counter_car = arr_new_car_id[i]

               
                #print("ID {} and coordinates {}".format(counter_car,bbox))
                color = colors[15]
                color = [i * 255 for i in color]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                cv2.putText(frame, class_name + "-" + str(int(counter_car)),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

            
            if class_name == 'bus':
                

                if int(obj_id) not in arr_busid:
                    new_bus_id+=1
                    arr_new_bus_id.append(new_bus_id)
                    arr_busid.append(int(obj_id))
                                   
                for i in range(len(arr_new_bus_id)):
                    a = arr_busid[i]                   
                    if obj_id == a:
                        counter_bus = arr_new_bus_id[i]

                

                color = colors[7]
                color = [i * 255 for i in color]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                cv2.putText(frame, class_name + "-" + str(int(counter_bus)),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

            
            if class_name == 'truck':

                if int(obj_id) not in arr_truckid:
                    new_truck_id+=1
                    arr_new_truck_id.append(new_truck_id)
                    arr_truckid.append(int(obj_id))
                                   
                for i in range(len(arr_new_truck_id)):
                    a = arr_truckid[i]                   
                    if obj_id == a:
                        counter_truck = arr_new_truck_id[i]

               

                color = colors[2]
                color = [i * 255 for i in color]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                cv2.putText(frame, class_name + "-" + str(int(counter_truck)),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

            if class_name == 'bicycle':
                if int(obj_id) not in arr_bicycleid:
                    new_bicycle_id+=1
                    arr_new_bicycle_id.append(new_bicycle_id)
                    arr_bicycleid.append(int(obj_id))
                                   
                for i in range(len(arr_new_bicycle_id)):
                    a = arr_bicycleid[i]                   
                    if obj_id == a:
                        counter = arr_new_bicycle_id[i]
               

                color = colors[6]
                color = [i * 255 for i in color]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                cv2.putText(frame, class_name + "-" + str(int(counter)),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            
            if (0 <= counting_point[0] <= ((counting_point[1]-590)*(1/-0.17857))) and (473<=counting_point[1]<=532):
                if class_name == 'car':
                    zone4_carcounter.append(int(track.track_id))
                    zone4_totalcounter.append(int(track.track_id))
                elif class_name == 'bicycle':
                    zone4_bicyclecounter.append(int(track.track_id))
                    zone4_totalcounter.append(int(track.track_id))
                elif class_name == 'truck':
                    zone4_truckcounter.append(int(track.track_id))
                    zone4_totalcounter.append(int(track.track_id))
                elif class_name == 'bus':
                    zone4_buscounter.append(int(track.track_id))
                    zone4_totalcounter.append(int(track.track_id))
                elif class_name == 'motorbike':
                    zone4_motorbikecounter.append(int(track.track_id))
                    zone4_totalcounter.append(int(track.track_id))
            if (((counting_point[1]-754.167)/-.4167) <= counting_point[0] <= ((counting_point[1]+59.375)/.625)) and (475<=counting_point[1]<=500):
                if class_name == 'car':
                    zone3_carcounter.append(int(track.track_id))
                    zone3_totalcounter.append(int(track.track_id))
                elif class_name == 'bicycle':
                    zone3_bicyclecounter.append(int(track.track_id))
                    zone3_totalcounter.append(int(track.track_id))
                elif class_name == 'truck':
                    zone3_truckcounter.append(int(track.track_id))
                    zone3_totalcounter.append(int(track.track_id))
                elif class_name == 'bus':
                    zone3_buscounter.append(int(track.track_id))
                    zone3_totalcounter.append(int(track.track_id))
                elif class_name == 'motorbike':
                    zone3_motorbikecounter.append(int(track.track_id))
                    zone3_totalcounter.append(int(track.track_id))
            if (((counting_point[1]+59.375)/0.625)<=counting_point[0] <= ((counting_point[1]-72.916)/.4167)) and (475<=counting_point[1]<=500):
                if class_name == 'car':
                    zone2_carcounter.append(int(track.track_id))
                    zone2_totalcounter.append(int(track.track_id))
                elif class_name == 'bicycle':
                    zone2_bicyclecounter.append(int(track.track_id))
                    zone2_totalcounter.append(int(track.track_id))
                elif class_name == 'truck':
                    zone2_truckcounter.append(int(track.track_id))
                    zone2_totalcounter.append(int(track.track_id))
                elif class_name == 'bus':
                    zone2_buscounter.append(int(track.track_id))
                    zone2_totalcounter.append(int(track.track_id))
                elif class_name == 'motorbike':
                    zone2_motorbikecounter.append(int(track.track_id))
                    zone2_totalcounter.append(int(track.track_id))
            if (((counting_point[1]-72.916)/.4167)<=counting_point[0] <= ((counting_point[1]-309.1)/.1515)) and (475<=counting_point[1]<=500):
                if class_name == 'car':
                    zone1_carcounter.append(int(track.track_id))
                    zone1_totalcounter.append(int(track.track_id))
                elif class_name == 'bicycle':
                    zone1_bicyclecounter.append(int(track.track_id))
                    zone1_totalcounter.append(int(track.track_id))
                elif class_name == 'truck':
                    zone1_truckcounter.append(int(track.track_id))
                    zone1_totalcounter.append(int(track.track_id))
                elif class_name == 'bus':
                    zone1_buscounter.append(int(track.track_id))
                    zone1_totalcounter.append(int(track.track_id))
                elif class_name == 'motorbike':
                    zone1_motorbikecounter.append(int(track.track_id))
                    zone1_totalcounter.append(int(track.track_id))
            if (450<=counting_point[0] <= 1280) and (counting_point[1]>=620):
                if class_name == 'car':
                    zone5_carcounter.append(int(track.track_id))
                    zone5_totalcounter.append(int(track.track_id))
                elif class_name == 'bicycle':
                    zone5_bicyclecounter.append(int(track.track_id))
                    zone5_totalcounter.append(int(track.track_id))
                elif class_name == 'truck':
                    zone5_truckcounter.append(int(track.track_id))
                    zone5_totalcounter.append(int(track.track_id))
                elif class_name == 'bus':
                    zone5_buscounter.append(int(track.track_id))
                    zone5_totalcounter.append(int(track.track_id))
                elif class_name == 'motorbike':
                    zone5_motorbikecounter.append(int(track.track_id))
                    zone5_totalcounter.append(int(track.track_id))
            if (counting_point[1]>=480) and (((counting_point[1]-520)/-.0901)<=counting_point[0]<=((counting_point[1]-620)/-.22047)):
                if class_name == 'person':
                    pedzone_totalcounter.append(int(track.track_id))
                

            #mark the coordinates 
            #cv2.circle(frame,  (bottom_left), 5, (0,0,0), -1)
            #cv2.circle(frame,  (bottom_right), 5, (0,0,0), -1)


            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))


            zone5_totalcount = len(set(zone5_totalcounter))
            zone5_carcount = len(set(zone5_carcounter))
            zone5_truckcount = len(set(zone5_truckcounter))
            zone5_buscount = len(set(zone5_buscounter))
            zone5_motorbikecount = len(set(zone5_motorbikecounter))
            zone5_bicyclecount = len(set(zone5_bicyclecounter))
            zone4_totalcount = len(set(zone4_totalcounter))
            zone4_carcount = len(set(zone4_carcounter))
            zone4_truckcount = len(set(zone4_truckcounter))
            zone4_buscount = len(set(zone4_buscounter))
            zone4_motorbikecount = len(set(zone4_motorbikecounter))
            zone4_bicyclecount = len(set(zone4_bicyclecounter))
            zone3_totalcount = len(set(zone3_totalcounter))
            zone3_carcount = len(set(zone3_carcounter))
            zone3_truckcount = len(set(zone3_truckcounter))
            zone3_buscount = len(set(zone3_buscounter))
            zone3_motorbikecount = len(set(zone3_motorbikecounter))
            zone3_bicyclecount = len(set(zone3_bicyclecounter))
            zone2_totalcount = len(set(zone2_totalcounter))
            zone2_carcount = len(set(zone2_carcounter))
            zone2_truckcount = len(set(zone2_truckcounter))
            zone2_buscount = len(set(zone2_buscounter))
            zone2_motorbikecount = len(set(zone2_motorbikecounter))
            zone2_bicyclecount = len(set(zone2_bicyclecounter))
            zone1_totalcount = len(set(zone1_totalcounter))
            zone1_carcount = len(set(zone1_carcounter))
            zone1_truckcount = len(set(zone1_truckcounter))
            zone1_buscount = len(set(zone1_buscounter))
            zone1_motorbikecount = len(set(zone1_motorbikecounter))
            zone1_bicyclecount = len(set(zone1_bicyclecounter))
            pedzone_totalcount = len(set(pedzone_totalcounter))
            cv2.putText(frame, 'Zone 5 Counter:' + str(zone5_totalcount),(30, 30),0, 0.55, (100,50,50),2)
            cv2.putText(frame, 'Cars:' + str(zone5_carcount),(30, 50),0, 0.55, (100,50,50),2)
            cv2.putText(frame, 'Trucks:' + str(zone5_truckcount),(30, 70),0, 0.55, (100,50,50),2)
            cv2.putText(frame, 'Buses:' + str(zone5_buscount),(30, 90),0, 0.55, (100,50,50),2)
            cv2.putText(frame, 'Motorbikes:' + str(zone5_motorbikecount),(30, 110),0, 0.55, (100,50,50),2)
            cv2.putText(frame, 'Bicycles:' + str(zone5_bicyclecount),(30, 130),0, 0.55, (100,50,50),2)
            cv2.putText(frame, 'Current:' + str(zone5_currentcounter),(30, 150),0, 0.55, (100,50,50),2)
            cv2.putText(frame, 'Zone 4 Counter:' + str(zone4_totalcount),(200, 30),0, 0.55, (255,0,0),2)
            cv2.putText(frame, 'Cars:' + str(zone4_carcount),(200, 50),0, 0.55, (255,0,0),2)
            cv2.putText(frame, 'Trucks:' + str(zone4_truckcount),(200, 70),0, 0.55, (255,0,0),2)
            cv2.putText(frame, 'Buses:' + str(zone4_buscount),(200, 90),0, 0.55, (255,0,0),2)
            cv2.putText(frame, 'Motorbikes:' + str(zone4_motorbikecount),(200, 110),0, 0.55, (255,0,0),2)
            cv2.putText(frame, 'Bicycles:' + str(zone4_bicyclecount),(200, 130),0, 0.55, (255,0,0),2)
            cv2.putText(frame, 'Current:' + str(zone4_currentcounter),(200, 150),0, 0.55, (255,0,0),2)
            cv2.putText(frame, 'Zone 3 Counter:' + str(zone3_totalcount),(370, 30),0, 0.55, (255,255,0),2)
            cv2.putText(frame, 'Cars:' + str(zone3_carcount),(370, 50),0, 0.55, (255,255,0),2)
            cv2.putText(frame, 'Trucks:' + str(zone3_truckcount),(370, 70),0, 0.55, (255,255,0),2)
            cv2.putText(frame, 'Buses:' + str(zone3_buscount),(370, 90),0, 0.55, (255,255,0),2)
            cv2.putText(frame, 'Motorbikes:' + str(zone3_motorbikecount),(370, 110),0, 0.55, (255,255,0),2)
            cv2.putText(frame, 'Bicycles:' + str(zone3_bicyclecount),(370, 130),0, 0.55, (255,255,0),2)
            cv2.putText(frame, 'Current:' + str(zone3_currentcounter),(370, 150),0, 0.55, (255,255,0),2)
            cv2.putText(frame, 'Zone 2 Counter:' + str(zone2_totalcount),(540, 30),0, 0.55, (0,255,0),2)
            cv2.putText(frame, 'Cars:' + str(zone2_carcount),(540, 50),0, 0.55, (0,255,0),2)
            cv2.putText(frame, 'Trucks:' + str(zone2_truckcount),(540, 70),0, 0.55, (0,255,0),2)
            cv2.putText(frame, 'Buses:' + str(zone2_buscount),(540, 90),0, 0.55, (0,255,0),2)
            cv2.putText(frame, 'Motorbikes:' + str(zone2_motorbikecount),(540, 110),0, 0.55, (0,255,0),2)
            cv2.putText(frame, 'Bicycles:' + str(zone2_bicyclecount),(540, 130),0, 0.55, (0,255,0),2)
            cv2.putText(frame, 'Current:' + str(zone2_currentcounter),(540, 150),0, 0.55, (0,255,0),2)
            cv2.putText(frame, 'Zone 1 Counter:' + str(zone1_totalcount),(710, 30),0, 0.55, (255,0,255),2)
            cv2.putText(frame, 'Cars:' + str(zone1_carcount),(710, 50),0, 0.55, (255,0,255),2)
            cv2.putText(frame, 'Trucks:' + str(zone1_truckcount),(710, 70),0, 0.55, (255,0,255),2)
            cv2.putText(frame, 'Buses:' + str(zone1_buscount),(710, 90),0, 0.55, (255,0,255),2)
            cv2.putText(frame, 'Motorbikes:' + str(zone1_motorbikecount),(710, 110),0, 0.55, (255,0,255),2)
            cv2.putText(frame, 'Bicycles:' + str(zone1_bicyclecount),(710, 130),0, 0.55, (255,0,255),2)
            cv2.putText(frame, 'Current:' + str(zone1_currentcounter),(710, 150),0, 0.55, (255,0,255),2)
            cv2.putText(frame, 'Ped Zone Counter:' + str(pedzone_totalcount),(880, 30),0, 0.55, (0,0,255),2)
            cv2.putText(frame, 'Current:' + str(pedzone_currentcounter),(880, 50),0, 0.55, (0,0,255),2)
        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        #print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)
        
        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass