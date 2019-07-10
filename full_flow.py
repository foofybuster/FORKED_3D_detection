import cv2
import numpy as np
import os
from util.post_processing import gen_3D_box,draw_3D_box,draw_2D_box
from net.bbox_3D_net import bbox_3D_net
from util.process_data import get_cam_data, get_dect2D_data
from pytorch_ssd.vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from pytorch_ssd.vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from pytorch_ssd.vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from pytorch_ssd.vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from pytorch_ssd.vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from pytorch_ssd.vision.utils.misc import Timer
import cv2
import torch
import sys
from pytorch_ssd.sort.sort import *

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Construct the network
model = bbox_3D_net((224,224,3))

model.load_weights(r'model_saved/weights.h5')

# image_dir = './dataset/kitti/data_object_image_2/training/image_2/'
calib_file = './dataset/kitti/calib/000005.txt'
# box2d_dir = './dataset/kitti/label_2/'

classes = ['Car','Van','Truck','Pedestrian','Person_sitting','Cyclist','Tram']
cls_to_ind = {cls:i for i,cls in enumerate(classes)}

dims_avg = np.loadtxt(r'dataset/voc_dims.txt',delimiter=',')


# all_image = sorted(os.listdir(image_dir))
# print(all_image)
# np.random.shuffle(all_image)

def three_D(image_file):

    cam_to_img = get_cam_data(calib_file)
    fx = cam_to_img[0][0]
    u0 = cam_to_img[0][2]
    v0 = cam_to_img[1][2]

# for f in all_image:
    # image_file = image_dir + f
    # box2d_file = box2d_dir + f.replace('png', 'txt')

    img = cv2.imread(image_file)

    dect2D_data,box2d_reserved = get_dect2D_data(box2d_file,classes)
    # Pedestrian 0.00 0 -0.20 712.40 143.00 810.73 307.92 1.89 0.48 1.20 1.84 1.47 8.41 0.01
        for data in dect2D_data:
            cls = data[0]
            box_2D = np.asarray(data[1],dtype=np.float)
            xmin = box_2D[0]
            ymin = box_2D[1]
            xmax = box_2D[2]
            ymax = box_2D[3]

            patch = img[int(ymin):int(ymax), int(xmin):int(xmax)]
            patch = cv2.resize(patch, (224, 224))
            patch = patch - np.array([[[103.939, 116.779, 123.68]]])
            patch = np.expand_dims(patch, 0)

            prediction = model.predict(patch)

            # compute dims
            dims = dims_avg[cls_to_ind[cls]] + prediction[0][0]

            # Transform regressed angle
            box2d_center_x = (xmin + xmax) / 2.0
            # Transfer arctan() from (-pi/2,pi/2) to (0,pi)
            theta_ray = np.arctan(fx /(box2d_center_x - u0))
            if theta_ray<0:
                theta_ray = theta_ray+np.pi

            max_anc = np.argmax(prediction[2][0])
            anchors = prediction[1][0][max_anc]

            if anchors[1] > 0:
                angle_offset = np.arccos(anchors[0])
            else:
                angle_offset = -np.arccos(anchors[0])

            bin_num = prediction[2][0].shape[0]
            wedge = 2. * np.pi / bin_num
            theta_loc = angle_offset + max_anc * wedge

            theta = theta_loc + theta_ray
            # object's yaw angle
            yaw = np.pi/2 - theta

            points2D = gen_3D_box(yaw, dims, cam_to_img, box_2D)
            draw_3D_box(img, points2D)

        for cls,box in box2d_reserved:
            draw_2D_box(img,box)

        cv2.imshow(f, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite('output/'+ f.replace('png','jpg'), img)

def ssd():

    if len(sys.argv) < 4:
        print('Usage: python run_ssd_example.py <net type>  <model path> <label path> [video file]')
        sys.exit(0)
    net_type = sys.argv[1]
    model_path = sys.argv[2]
    label_path = sys.argv[3]

    if len(sys.argv) >= 5:
        cap = cv2.VideoCapture(sys.argv[4])  # capture from file
    else:
        cap = cv2.VideoCapture(0)   # capture from camera
        cap.set(3, 1920)
        cap.set(4, 1080)

    class_names = [name.strip() for name in open(label_path).readlines()]
    num_classes = len(class_names)


    if net_type == 'vgg16-ssd':
        net = create_vgg_ssd(len(class_names), is_test=True)
    elif net_type == 'mb1-ssd':
        net = create_mobilenetv1_ssd(len(class_names), is_test=True)
    elif net_type == 'mb1-ssd-lite':
        net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
    elif net_type == 'mb2-ssd-lite':
        net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
    elif net_type == 'sq-ssd-lite':
        net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
    else:
        print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
        sys.exit(1)
    net.load(model_path)

    if net_type == 'vgg16-ssd':
        predictor = create_vgg_ssd_predictor(net, candidate_size=200)
    elif net_type == 'mb1-ssd':
        predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
    elif net_type == 'mb1-ssd-lite':
        predictor = create_mobilenetv1_ssd_lite_predictor(net, candidate_size=200)
    elif net_type == 'mb2-ssd-lite':
        predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)
    elif net_type == 'sq-ssd-lite':
        predictor = create_squeezenet_ssd_lite_predictor(net, candidate_size=200)
    else:
        print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
        sys.exit(1)

    # traced_script_module = torch.jit.trace(net, torch.rand(1, 3, 300, 300))
    # output = traced_script_module(torch.ones(1, 3, 300, 300))
    # print(output[0 :5])
    # traced_script_module.save("model.pt")


    timer = Timer()
    mot_tracker = Sort()
    frame = 0
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    # out = cv2.VideoWriter('./everyOtherFrame-ssdWsort.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
    while True:
        ret, orig_image = cap.read()
        if orig_image is None:
            break
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        timer.start()
        if(frame%2 == 0):
            boxes, labels, probs = predictor.predict(image, 10, 0.4)
        track_bbs_ids = torch.FloatTensor(mot_tracker.update(boxes))
        print(track_bbs_ids.size(0))
        interval = timer.end()
        print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))
        for i in range(track_bbs_ids.size(0)):
            box = track_bbs_ids[i, :-1]
            # label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
            cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)

            # cv2.putText(orig_image, label,
            #             (box[0]+20, box[1]+40),
            #             cv2.FONT_HERSHEY_SIMPLEX,
            #             1,  # font scale
            #             (255, 0, 255),
            #             2)  # line type
        cv2.imshow('annotated', orig_image)
        # out.write(orig_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame += 1
    cap.release()
    cv2.destroyAllWindows()
ssd()