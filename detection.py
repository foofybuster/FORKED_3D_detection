import cv2
import numpy as np
import os
from util.post_processing import gen_3D_box,draw_3D_box,draw_2D_box
from net.bbox_3D_net import bbox_3D_net
from util.process_data import get_cam_data, get_dect2D_data

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Construct the network
model = bbox_3D_net((224,224,3))

model.load_weights(r'model_saved/weights.h5')

image_dir = './dataset/kitti/data_object_image_2/training/image_2/'
calib_file = './dataset/kitti/calib/000005.txt'
box2d_dir = './dataset/kitti/label_2/'

classes = ['Car','Van','Truck','Pedestrian','Person_sitting','Cyclist','Tram']
cls_to_ind = {cls:i for i,cls in enumerate(classes)}

dims_avg = np.loadtxt(r'dataset/voc_dims.txt',delimiter=',')


all_image = sorted(os.listdir(image_dir))
# print(all_image)
# np.random.shuffle(all_image)

cam_to_img = get_cam_data(calib_file)
fx = cam_to_img[0][0]
u0 = cam_to_img[0][2]
v0 = cam_to_img[1][2]

for f in all_image:
    image_file = image_dir + f
    box2d_file = box2d_dir + f.replace('png', 'txt')

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
