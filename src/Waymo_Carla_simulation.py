import os
import math
import numpy as np
import tensorflow as tf
import itertools
import carla
import random
import shutil
tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import  dataset_pb2 as open_dataset

from os import listdir
from os.path import isfile, join
import glob

start_point = [-8,-85.1]
DT =0.1
global self_vehicle

def collect_vehicle_laser_labels(frame):
    # only return label.type equal vehicle
    #TYPE_VEHICLE = 1
    return [data for data in frame.laser_labels if data.type == 1]

def get_vehicle_pose(frame):
    # get front pose
    front_image = frame.images[0]
    pose = [t for t in front_image.pose.transform]
    return np.asarray(pose).reshape((4,4))

def get_self_angle(self_vehicle_pose):
    return(math.atan2(self_vehicle_pose[1][0],self_vehicle_pose[0][0])/math.pi*180)

def get_relative_distance(front_car_label, vehicle_pose):
    relative_dist = np.asarray([front_car_label.box.center_x, front_car_label.box.center_y, 0])
    _relative_dist = np.hstack((relative_dist, [0]))
    return np.matmul(vehicle_pose, _relative_dist)[:2]

def move_self_vehicle(frame,world,if_first,self_vehicle_sita):
    global self_vehicle
    blueprint_library = world.get_blueprint_library()
    vehicle_bp_back = random.choice(blueprint_library.filter('vehicle.lincoln.mkz2017'))
    vehicle_bp_back.set_attribute('role_name', 'hero')
    wmap = world.get_map()

    if if_first==True:
        transform_self = carla.Transform(carla.Location(x=start_point[0], y=start_point[1], z= 0.3), carla.Rotation(yaw=90))
        self_vehicle = world.spawn_actor(vehicle_bp_back, transform_self)
        self_x = start_point[0]
        self_y = start_point[1]

    else:
        sita = self_vehicle_sita/180*math.pi
        self_x = self_vehicle.get_location().x
        self_y = self_vehicle.get_location().y
        self_vx = frame.images[0].velocity.v_x
        self_vy = frame.images[0].velocity.v_y
        vx = self_vy* math.sin(sita)+ self_vx* math.cos(sita)
        vy = self_vy* math.cos(sita)- self_vx* math.sin(sita)
        self_x = self_x+vy*DT
        self_y = self_y+vx*DT
        height_b = wmap.get_waypoint(carla.Location(x=self_x, y=self_y),project_to_road=False).transform.location.z
        self_vehicle.set_transform(carla.Transform(carla.Location(x=self_x, y=self_y, z =height_b), carla.Rotation(yaw=90+self_vehicle_sita-get_self_angle(get_vehicle_pose(frame)))))
    
    return self_x,self_y

def move_surrounding_vehicles(self_vehicle_sita,frame, world,vehicles_frame,vehicles_rolename_world,self_x,self_y):
    wmap = world.get_map()
    vehicles_world = world.get_actors()
    vehicles_world = vehicles_world.filter('vehicle.*')
    self_vehicle_pose = get_vehicle_pose(frame)
    self_vehicle_angle = get_self_angle(self_vehicle_pose)
    sita = self_vehicle_sita/180*math.pi
    blueprint_library = world.get_blueprint_library()
    vehicle_surrounding = random.choice(blueprint_library.filter('vehicle.mini.cooperst'))

    for vehicle in vehicles_frame:
        x_relavent,y_relavent = get_relative_distance(vehicle, self_vehicle_pose)
        angle = 90+self_vehicle_sita-self_vehicle_angle-vehicle.box.heading/math.pi*180
        x_g = self_x + y_relavent* math.cos(sita)- x_relavent* math.sin(sita)
        y_g = self_y + y_relavent* math.sin(sita)+ x_relavent* math.cos(sita)
        try:
            h_g = wmap.get_waypoint(carla.Location(x=x_g, y=y_g),project_to_road=False).transform.location.z
        except:
            h_g = 0
        if vehicle.id not in vehicles_rolename_world:
            vehicle_surrounding.set_attribute('role_name', vehicle.id)
            transform_self = carla.Transform(carla.Location(x= x_g, y= y_g, z=0.3), carla.Rotation(yaw=angle))
            actor = world.try_spawn_actor(vehicle_surrounding, transform_self)
        else:
            for vehicle_carla in vehicles_world:
                if vehicle_carla.attributes.get('role_name') == vehicle.id:
                    vehicle_carla.set_transform(carla.Transform(carla.Location(x=x_g, y=y_g, z=h_g), carla.Rotation(yaw=angle)))
                    continue

def start_simulation(frames,world):
    global self_vehicle
    i = 1
    self_vehicle_pose = get_vehicle_pose(frames[0])
    self_vehicle_sita = get_self_angle(self_vehicle_pose)
    print('road_direction',self_vehicle_sita)
    if_first = True
    for frame in frames:

        i = i+1
        print('frame:',i)
        vehicles_world = world.get_actors()
        vehicles_world = vehicles_world.filter('vehicle.*')
        vehicles_frame = collect_vehicle_laser_labels(frame)
        

        ### vehicles on this frame
        vehicles_frame_id = []
        vehicles_frame_id.append('hero')
        for vehicle in vehicles_frame:
            vehicles_frame_id.append(vehicle.id)

        ### destroy vehicles in carla
        for vehicle in vehicles_world:
            if vehicle.attributes.get('role_name') not in vehicles_frame_id:
                vehicle.destroy()

        ### vehicles in carla
        vehicles_rolename_world = []
        for vehicle in vehicles_world:
            vehicles_rolename_world.append(vehicle.attributes.get('role_name'))

        self_x,self_y = move_self_vehicle(frame,world,if_first,self_vehicle_sita)
        move_surrounding_vehicles(self_vehicle_sita,frame, world, vehicles_frame, vehicles_rolename_world,self_x,self_y)
        
        if if_first:
            blueprint_library = world.get_blueprint_library()
            blueprint = blueprint_library.find('sensor.camera.rgb')
            blueprint.set_attribute('image_size_x', '1920')
            blueprint.set_attribute('image_size_y', '1080')
            blueprint.set_attribute('fov', '50')
            transform_sensor = carla.Transform(carla.Location(x=0.7, z=1.9))
            sensor = world.spawn_actor(blueprint, transform_sensor, attach_to=self_vehicle)
            RESULT_PATH_png = '/media/yongjie/Seagate Expansion Drive/Waymo_test_coordinate/images/'
            sensor.listen(lambda image: image.save_to_disk(RESULT_PATH_png+'%06d.png' % image.frame))
            if os.path.exists(RESULT_PATH_png):
                shutil.rmtree(RESULT_PATH_png)
            os.makedirs(RESULT_PATH_png)
        
        world.tick()
        if_first = False
        # vehicles_frame = collect_vehicle_laser_labels(frame)
        # for vehicle in vehicles_frame:
        #     print(i,vehicle.id)


file_path = '../datasets/training_00003'
file_path = file_path + 'car_following_segs_training_training_0006_segment-12012663867578114640_820_000_840_000.tfrecord'
dataset = tf.data.TFRecordDataset(file_path, compression_type='')

def main():
    frames = []
    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        frames.append(frame)

    print("Num of frames:", len(frames))

    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)
    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = DT
    world.apply_settings(settings)
    debug = world.debug
    start_simulation(frames,world)

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
