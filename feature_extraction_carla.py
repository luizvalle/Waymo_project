import os
import tensorflow as tf
import math
import numpy as np
import itertools

tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import  dataset_pb2 as open_dataset

from os import listdir
from os.path import isfile, join
import glob


# tar_name = 'training_0010'
# files_path = '/home/zg2309/data/training/{}/'.format(tar_name)

# tar_name = 'training_0001'
# files_path = '/media/yongjie/Seagate Expansion Drive/Waymo_dataset/training/{}/'.format(tar_name)

tar_name = 'trainning_car_following'
files_path = '/media/yongjie/Seagate Expansion Drive/Waymo_dataset/trainning_car_following/'
files = glob.glob(files_path + "car_following_segs_training_training_0004_segment-10664823084372323928_4360_000_4380_000.tfrecord")
print(len(files))

# For visualization: https://drive.google.com/drive/u/0/folders/1-CXDJwgd96fTHHboekfdyIxVj2lzjf1n
from shapely.geometry import Polygon, LineString

def intersects(label):
  # Starting from the upper-left corner, clock direction
  bounding_box = Polygon([
      (label.box.center_x - 0.5 * label.box.length, label.box.center_y + 0.5 * label.box.width), 
      (label.box.center_x + 0.5 * label.box.length, label.box.center_y - 0.5 * label.box.width), 
      (label.box.center_x + 0.5 * label.box.length, label.box.center_y + 0.5 * label.box.width),
      (label.box.center_x - 0.5 * label.box.length, label.box.center_y - 0.5 * label.box.width)
  ])
  
  line = LineString([(0, 0), (label.box.center_x, 0)])
  
  return bounding_box.intersects(line)

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def find_corresponding_projected_lidar_label(frame, front_car_laser_label):
    for pll_wrapper in frame.projected_lidar_labels:
        if pll_wrapper.name != FRONT:
            continue
      
    for pll in pll_wrapper.labels:
        if front_car_laser_label.id in pll.id:
            return pll
      
    return None

def show_label_on_image(camera_image, label, layout, cmap=None):
    """Show a camera image and the given camera labels."""
    ax = plt.subplot(*layout)

      # Draw the object bounding box.
    ax.add_patch(patches.Rectangle(
      xy=(label.box.center_x - 0.5 * label.box.length,
          label.box.center_y - 0.5 * label.box.width),
      width=label.box.length,
      height=label.box.width,
      linewidth=1,
      edgecolor='red',
      facecolor='none'))

  # Show the camera image.
    plt.imshow(tf.image.decode_jpeg(camera_image.image), cmap=cmap)
    plt.title(open_dataset.CameraName.Name.Name(camera_image.name))
    plt.grid(False)
    plt.axis('off')

    plt.figure(figsize=(25, 20))
    
def verify_front_car_label(frame, front_car_label):
    """
    Display the bounding box of the found front car on the image to verify if the 
    front car is captured correctly.
    """
    for index, image in enumerate(frame.images):
        front_car_pll = find_corresponding_projected_lidar_label(frame, front_car_label)
        if front_car_pll is not None:
            show_label_on_image(image, front_car_pll, [3, 3, index + 1])
        break
import random
import numpy as np

def collect_vehicle_laser_labels(frame):
    # only return label.type equal vehicle
    #TYPE_VEHICLE = 1
    return [data for data in frame.laser_labels if data.type == 1]

def get_front_car_laser_label(labels):
    """
    Find the closest bounding box which intersects with y = 0 and its center_x is positive
    """
  
    front_car_label = None
    for label in labels:
        if label.box.center_x < 0:
            continue 
      
        if intersects(label):
            if front_car_label is None or front_car_label.box.center_x > label.box.center_x:
                front_car_label = label
      
    return front_car_label


def car_acceleration(v1, v2, dt):
    return (v1 - v2) / dt

TYPE_VEHICLE = 1
FRONT = 0
FPS = 10
DT = 1.0 / FPS
VERIFY_THRESHOLD = 0.05

FRONTCAR_Y_THRESHOLD = 0.05

"""
features:
[vx, vy, vz, dx, dy, vfx, vfy, vfz, afx, afy, afz]
labels:
[ax, ay, az]
"""

def write_to_csv(filename, feats, labels):
    comb_np = np.hstack((feats, labels))
    np.savetxt(filename, comb_np, delimiter=",")

def get_vehicle_pose(frame):
    # get front pose
    front_image = frame.images[0]
    pose = [t for t in front_image.pose.transform]
    print(pose)
    return np.asarray(pose).reshape((4,4))

def get_current_car_velocity_wrt_GF(frame):
    """
    Return the speed v_x, v_y, v_z of the current car
    """
    image = frame.images[FRONT]
    return np.asarray([image.velocity.v_x, image.velocity.v_y, image.velocity.v_z])

def get_front_car_velocity_wrt_GF(front_car_label, vehicle_pose, v_cur_GF):
    v_front_VF = np.asarray([front_car_label.metadata.speed_x, front_car_label.metadata.speed_y, 0])
    return v_front_VF

def get_relative_distance(front_car_label):
    return np.asarray([front_car_label.box.center_x, front_car_label.box.center_y])

def get_current_car_accel_GF_per_frame(dt, v_cur_GF, v_cur_GF_prev):
    return car_acceleration(v_cur_GF, v_cur_GF_prev, dt) if v_cur_GF_prev is not None else np.asarray([0, 0, 0])

def get_front_car_GF_features_per_frame(dt, frame, vehicle_pose, front_car_label,
                                        v_cur_GF, v_front_GF_prev, verify=False):

    if verify and random.random() < VERIFY_THRESHOLD:
        verify_front_car_label(frame, front_car_label)

    relative_dist = get_relative_distance(front_car_label) # 2 * 1
    v_front_GF = get_front_car_velocity_wrt_GF(front_car_label, vehicle_pose, v_cur_GF) # 3 * 1
    a_front_GF = car_acceleration(v_front_GF, v_front_GF_prev, dt) if v_front_GF_prev is not None else np.asarray([0, 0, 0]) # 3 * 1

    return np.hstack((relative_dist, v_front_GF, a_front_GF)), v_front_GF


def get_essentials_per_frame(dt, frame, front_car_label, v_cur_GF_prev, v_front_GF_prev):
    vehicle_pose = get_vehicle_pose(frame)
    v_cur_GF = get_current_car_velocity_wrt_GF(frame) # 3 * 1
    front_GF_feat, v_front_GF = get_front_car_GF_features_per_frame(DT, frame, vehicle_pose, front_car_label,
                                                                  v_cur_GF, v_front_GF_prev) # 8 * 1
    a_cur_GF = get_current_car_accel_GF_per_frame(dt, v_cur_GF, v_cur_GF_prev) # 3 * 1

    return np.hstack((v_cur_GF, front_GF_feat)), a_cur_GF, v_cur_GF, v_front_GF

def get_features_and_labels(frames):
    feat_set = []
    label_set = []
    
    # init
    v_cur_GF_prev = None
    v_front_GF_prev = None

    for frame in frames:
        # Capture the front car
        v_laser_labels = collect_vehicle_laser_labels(frame)
        front_car_label = get_front_car_laser_label(v_laser_labels)

        if front_car_label is not None:
            feats, labels, v_cur_GF_prev, v_front_GF_prev = get_essentials_per_frame(DT, frame, front_car_label, v_cur_GF_prev, v_front_GF_prev)
        else:
            #if there is no front car
            v_cur_GF = get_current_car_velocity_wrt_GF(frame)
            vx, vy, vz = v_cur_GF
            feats = [vx, vy, vz, 0, 0, 0, 0, 0, 0, 0, 0]
            ax, ay, az = [0,0,0]
            
            if v_cur_GF_prev is not None:
                ax, ay, az = get_current_car_accel_GF_per_frame(DT, v_cur_GF, v_cur_GF_prev)
            labels = [ax, ay, az]
            
            v_cur_GF_prev = v_cur_GF
            v_front_GF_prev = None
            
        feat_set.append(feats)
        label_set.append(labels)
        
        
    # fix first frame acceleration [0,0,0]
    if np.sum(np.abs(label_set[0])) == 0:
        label_set[0] = label_set[1]
        
    return np.asarray(feat_set), np.asarray(label_set)




def visualization(folder_name, feats, labels, smooth_feats, smooth_labels):    
    VX = 0
    VY = 1
    VZ = 2
    DX = 3
    DY = 4
    VFX = 5
    VFY = 6
    VFZ = 7
    AFX = 8
    AFY = 9
    AFZ = 10

    AX = 0
    AY = 1

    times = [t * DT for t in range(0, len(feats))] 
 
    fig1, ax1 = plt.subplots()
    
    dxs = [f[DX] for f in feats]
    ax1.plot(times, dxs, label='origin')
    
    dxss = [f[DX] for f in smooth_feats]
    ax1.plot(times, dxss, label="smooth")
    
    ax1.set_ylabel('relative distance along x')
    ax1.set_xlabel('time')
    fig1.savefig(folder_name + 'relative_distance_x.png')
    
    fig2, ax2 = plt.subplots()
    
    dys = [f[DY] for f in feats]
    ax2.plot(times, dys, label='origin')
    
    dyss = [f[DY] for f in smooth_feats]
    ax2.plot(times, dyss, label="smooth")
    
    ax2.set_ylabel('relative distance along y')
    ax2.set_xlabel('time')
    fig2.savefig(folder_name + 'relative_distance_y.png')

    fig3, ax3 = plt.subplots()
    
    afxs = [f[AFX] for f in feats]
    ax3.plot(times, afxs, label='origin')
    
    afxss = [f[AFX] for f in smooth_feats]
    ax3.plot(times, afxss, label="smooth")
    
    ax3.set_ylabel('accel x of front car')
    ax3.set_xlabel('time')
    fig3.savefig(folder_name + 'accel_x_front_car.png')

    fig4, ax4 = plt.subplots()
    
    afys = [f[AFY] for f in feats]
    ax4.plot(times, afys, label='origin')
    
    afyss = [f[AFY] for f in smooth_feats]
    ax4.plot(times, afyss, label="smooth")
    
    ax4.set_ylabel('accel y of front car')
    ax4.set_xlabel('time')
    fig4.savefig(folder_name + 'accel_y_front_car.png')

    # For verification
    fig5, ax5 = plt.subplots()
    
    vfxs = [f[VFX] for f in feats]
    ax5.plot(times, vfxs, label='origin')
    
#     vfxss = [f[VFX] for f in smooth_feats]
#     ax5.plot(times, vfxss, label="smooth")
    
    ax5.set_ylabel('speed x of front car')
    ax5.set_xlabel('time')
    fig5.savefig(folder_name + 'speed_x_front_car.png')

    fig6, ax6 = plt.subplots()
    
    vfys = [f[VFY] for f in feats]
    ax6.plot(times, vfys, label='origin')
    
#     vfyss = [f[VFY] for f in smooth_feats]
#     ax6.plot(times, vfyss, label="smooth")
    
    ax6.set_ylabel('speed y of front car')
    ax6.set_xlabel('time')
    fig6.savefig(folder_name + 'speed_y_front_car.png')

    fig7, ax7 = plt.subplots()
    
    axs = [l[AX] for l in labels]
    ax7.plot(times, axs, label="origin")
    
    axss = [l[AX] for l in smooth_labels]
    ax7.plot(times, axss, label="smooth")
    ax7.set_ylabel('accel x of current car')
    ax7.set_xlabel('time')
    fig7.savefig(folder_name + 'accel_x_current_car.png')

    fig8, ax8 = plt.subplots()
    
    ays = [l[AY] for l in labels]
    ax8.plot(times, ays, label="origin")
    ayss = [l[AY] for l in smooth_labels]
    
    ax8.plot(times, ayss, label="smooth")
    ax8.set_ylabel('accel y of current car')
    ax8.set_xlabel('time')
    fig8.savefig(folder_name + 'accel_y_current_car.png')

    fig9, ax9 = plt.subplots()
    
    vxs = [f[VX] for f in feats]
    ax9.plot(times, vxs, label='origin')
    
#     vxss = [f[VX] for f in smooth_feats]
#     ax9.plot(times, vxss, label="smooth")
    
    ax9.set_ylabel('speed x of current car')
    ax9.set_xlabel('time')
    fig9.savefig(folder_name + 'speed_x_current_car.png')

    fig10, ax10 = plt.subplots()
    
    vys = [f[VY] for f in feats]
    ax10.plot(times, vys, label='origin')
    
#     vyss = [f[VY] for f in smooth_feats]
#     ax10.plot(times, vyss, label="smooth")
    
    ax10.set_ylabel('speed y of current car')
    ax10.set_xlabel('time')
    fig10.savefig(folder_name + 'speed_y_current_car.png')
    
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)
    plt.close(fig5)
    plt.close(fig6)
    plt.close(fig7)
    plt.close(fig8)
    plt.close(fig9)
    plt.close(fig10)

import cv2

FRONT = 0 # front view
TYPE_VEHICLE = 1
TYPE_PEDESTRIAN = 2
TYPE_SIGN = 3
TYPE_CYCLIST = 4

def video_generation(folder, frames):
    imgs = []

    for frame in frames:
        image = frame.images[FRONT]
        img = tf.image.decode_jpeg(image.image)
        imgs.append(img.numpy())
    
    img = imgs[0]
    height, width, _ = img.shape
    size = (width, height)
    out_video = folder + 'video.mp4'
    fps = 10
    
    out = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*'XVID'), fps, size)
    
    for img in imgs:
        gbr = img[...,::-1].copy()
        out.write(gbr)
    out.release()

def add_label_to_camera_image(camera_image, camera_labels):
    
    # convert rgb array to opencv's bgr format
    im_arr_bgr = cv2.cvtColor(camera_image, cv2.COLOR_RGB2BGR)
    for label in camera_labels:
        # Draw the object bounding box.
        lt = label.type
        #bgr
        if lt == TYPE_VEHICLE:
            #blue
            ec = (0,0,255)
        elif lt == TYPE_PEDESTRIAN:
            ec = (0,255,0)
        elif lt == TYPE_SIGN:
            ec = (255,255,255)
        elif lt == TYPE_CYCLIST:
            ec = (255,0,0)
        
        pts1 = (int(label.box.center_x - 0.5 * label.box.length), int(label.box.center_y - 0.5 * label.box.width))
        pts2 = (int(label.box.center_x + 0.5 * label.box.length), int(label.box.center_y + 0.5 * label.box.width))

        cv2.rectangle(im_arr_bgr, pts1, pts2, ec, 1)
    
    im_arr = cv2.cvtColor(im_arr_bgr, cv2.COLOR_BGR2RGB)
    return im_arr
    
def video_generation_with_label(folder, frames):
    imgs = []

    for frame in frames:
        image = frame.images[FRONT]
        img = tf.image.decode_jpeg(image.image).numpy()
        labels = frame.projected_lidar_labels[FRONT].labels
        if len(labels) > 0:
            img = add_label_to_camera_image(img, labels)
        imgs.append(img)
    
    img = imgs[0]
    height, width, _ = img.shape
    size = (width, height)
    out_video = folder + 'video_label.mp4'
    fps = 10
    
    out = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*'XVID'), fps, size)
    
    for img in imgs:
        gbr = img[...,::-1].copy()
        out.write(gbr)
    out.release()

import shutil
RESULT_PATH = '/media/yongjie/Seagate Expansion Drive/Waymo_dataset/Team1-master/Team1-master/result/'

if os.path.exists(RESULT_PATH):
    shutil.rmtree(RESULT_PATH)
os.mkdir(RESULT_PATH)

for i in range(len(files)):
    file = files[i]
    dataset = tf.data.TFRecordDataset(file, compression_type='')
    
    # Load frames from dataset
    frames = []
    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        frames.append(frame)
        
    
#     print(collect_vehicle_laser_labels(frames[0])[0])
#     break
    print("filename:", file, "Num of frames:", len(frames))
        
    file_name = file.split('/')[-1].split('.')[0]
    save_folder = RESULT_PATH + file_name + '/'
    os.mkdir(save_folder)
    
    feats, labels = get_features_and_labels(frames)

    write_to_csv(save_folder + 'data.csv', feats, labels)
    
    # smooth only acceleration!
    box_pts = 5
    box = np.ones(box_pts)/box_pts
    
    smooth_feats = np.array(feats).copy()
    
#    DX = 3
#     DY = 4
#     AFX = 8
#     AFY = 9
#     AFZ = 10
    smooth_feats_idx = [3,4,8,9,10]
    
    for i in smooth_feats_idx:
        smooth_feats[:,i] = np.convolve(smooth_feats[:,i], box, mode='same')
    
    smooth_labels = np.array(labels).copy()
    _, label_num = smooth_labels.shape
    
    for i in range(label_num):
        smooth_labels[:,i] = np.convolve(smooth_labels[:,i], box, mode='same')
    
    write_to_csv(save_folder + 'data_smooth.csv', smooth_feats, smooth_labels)

    visualization(save_folder, feats, labels, smooth_feats, smooth_labels)
    
    video_generation(save_folder, frames)
    video_generation_with_label(save_folder, frames)
