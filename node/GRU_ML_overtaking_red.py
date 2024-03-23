#!/usr/bin/env python3

"""
Copyright 2022 Jiancheng Zhang

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from queue import Queue
import numpy as np

import rospy
import rospkg
import tensorflow
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import String

import tensorflow as tf
from tensorflow.keras.models import load_model





"""
This node subscribes red car's LiDAR and carState, publish driving command for red car
"""

# how many timestamps will be used?
# larger input means the model will use more inputs data, but the performance will not necessarily be better!
# 虽然增加输入数据的数量（即考虑更多的时间戳）可以提供更多的历史信息给模型，但这并不总是能带来性能上的提升。实际上，太多的输入数据可能会导致模型处理变得复杂，增加计算负担，且有可能引入噪声，反而降低预测性能。因此，选择合适的size_of_input是一项需要考虑数据特性、模型能力和实际应用需求的任务。



size_of_input = 6
LiDAR_raw_array = Queue(maxsize=size_of_input)
# LiDAR_raw_array = Queue(maxsize=size_of_input)创建了一个先进先出（FIFO）队列，用于存储LiDAR传感器捕获的原始数据。队列的最大大小被设置为size_of_input，即6，这意味着队列将只保存最新的6个时间点的LiDAR数据。当新的数据点加入队列时，如果队列已满，则最旧的数据点将被自动移除。
car_state_array = Queue(maxsize=size_of_input)
# car_state_array = Queue(maxsize=size_of_input)同样创建了一个队列，用于存储车辆状态数据，如速度和转向角度等。这个队列也遵循先进先出的原则，其最大大小同样为6。通过维护这样一个队列，可以确保模型总是使用最近的车辆状态数据进行预测。


def LiDAR_callback(data):
    global LiDAR_raw_array

    # Note the ros message is not related to programming language, same type of message can be read in both Python and C
    LiDAR_raw = data.ranges

    LiDAR_raw = np.asarray(LiDAR_raw)
    LiDAR_raw = np.reshape(LiDAR_raw, (1, -1))

    # First In First Out
    if LiDAR_raw_array.full():
        LiDAR_raw_array.get()
    LiDAR_raw_array.put(LiDAR_raw)
    # 队列实时维护了6(size_of_input)个长度为1081的雷达数据



# 实时更新并维护一个包含六个时间点的车辆状态信息的队列。每个时间点的信息包括车辆的X轴速度（Velocity_X）和车辆的转向角度（Steering_angle）
def carState_callback(data):
    global car_state_array

    car_state = np.asarray(data.data.split(","), dtype=float)

    speed_steering = []
    # car_state[3] is Velocity_X
    speed_steering.append(car_state[3])
    # car_state[5] is Steering_angle
    speed_steering.append(car_state[5])
    speed_steering = np.asarray(speed_steering)
    speed_steering = np.reshape(speed_steering, (1, -1))

    if car_state_array.full():
        car_state_array.get()
    car_state_array.put(speed_steering)


if __name__ == '__main__':

    rospy.init_node("ML_overtake_red", anonymous=True)
    """
    Australia 5 7 8 9 10 11 12 13 14
    Shanghai 9 10 12 14
    14_5 14_3 12_3 10_4 10_3

    14 = none -> Australia
    14_1 = 14 -> Shanghai
    14_2 = 14_1 -> Gulf data
    14_5 = 14_2 -> all data
    14_3 = 14_2 -> malaysian data

    12 = none -> Australia
    12_1 = 12 -> Shanghai
    12_2 = 12_1 -> Gulf data
    12_3 = 12_2 -> malaysian data

    10 = none -> Australia
    10_1 = 10 -> Shanghai
    10_2 = 10_1 -> Gulf data
    10_3 = 10_2 -> malaysian data
    10_4 = 10_3 -> Australia+malaysian
    """

    # # 记录节点开始运行的时间
    # start_time = rospy.Time.now()


    rospack = rospkg.RosPack()
    # overtaking_model = load_model(rospack.get_path("f1tenth_simulator_two_agents")+'/overtaking_models/shanghai_GRUN_First_edition')
    overtaking_model = load_model(rospack.get_path("f1tenth_simulator_two_agents")+'/overtaking_models/12_GRU_V2')


    overtaking_model.summary()
    overtaking_model.compile(loss="mean_squared_error", optimizer="adam", metrics=['mean_absolute_error'])
    # 编译模型，设置损失函数为平均绝对误差（MAE），优化器为Adam，这是常见的配置，适用于回归任务。

    # # 定义一个回调函数来在节点关闭时执行
    # def on_shutdown():
    #     end_time = rospy.Time.now()
    #     total_time = end_time - start_time
    #     rospy.loginfo("Node shutdown. Total runtime: {:.2f} seconds".format(total_time.to_sec()))
    # # 注册回调函数，以便在节点关闭时执行
    # rospy.on_shutdown(on_shutdown)


    for i, layer in enumerate(overtaking_model.layers):
        print(i, layer)
        try:
            print(" ", layer.activation)
        except AttributeError:
            print(" no activation attribute")

    carState_topic = rospy.get_param("~carState_topic_red")
    rospy.Subscriber(carState_topic, String, carState_callback)
    # 获取车辆状态信息

    scan_topic_red = rospy.get_param("~scan_topic_red")
    rospy.Subscriber(scan_topic_red, LaserScan, LiDAR_callback)
    # 获取LiDAR扫描数据

    overtaking_drive_topic = rospy.get_param("~overtaking_drive_topic")
    drive_pub_red = rospy.Publisher(overtaking_drive_topic, AckermannDriveStamped, queue_size=10)
    # 创建了一个发布者，用于发布车辆的控制命令。这些命令包括速度和转向角，使用AckermannDriveStamped消息类型。
    # 在发布和接收消息的过程中，可能会出现消息的发送速度快于接收端处理速度的情况。为了不立即丢失这些额外的消息，ROS提供了一个队列机制，暂时存储这些消息直到接收者准备好处理它们。queue_size=10 定义了这个队列可以存储的消息数量。

    # the LiDAR_raw_array need at least one LiDAR data in it
    rospy.wait_for_message(scan_topic_red, LaserScan)

    while not rospy.is_shutdown():

        inputA = np.asarray(list(LiDAR_raw_array.queue))
        inputB = np.asarray(list(car_state_array.queue))
        # 将LiDAR数据和车辆状态数据从各自的队列中提取出来，并转换为NumPy数组。这样做是为了方便后续的数据处理和模型预测。
        # 这是不能改变的,因为ROS的两个节点是需要分别读取雷达数据和车辆状态数据的,没有办法一起读取.
        # print(inputA.shape)
        # print(inputB.shape)

        # for new model 新添加部分
        # 将inputA和inputB沿最后一个维度（特征维度）合并
        # 不应该在这里！！！！！！！！而是在if条件下面
        # 因为在这里合并的话 第一个维度可能不匹配，因为没有if length_min == size_of_input:的保证，所以可能合并失败

        
        # combined_input = np.concatenate((inputA.reshape(size_of_input, -1), inputB.reshape(size_of_input, -1)), axis=1) 
        # 这是错的代码

        # combined_input = np.concatenate((inputA, inputB), axis=2)
        # 这是对的代码，但不应该出现在这里



        

        # inputA: (size_of_input, 1, 1081) means there are size_of_input number of instances, each instance is 1*1081 inputA结构的第一个元素是size_of_input 传进来就是(6,1,1081)
        # inputB: (size_of_input, 1, 2) means there are size_of_input number of instances, each instance is 1*2
        lengthA = inputA.shape[0]
        lengthB = inputB.shape[0]
        length_min = min(lengthA, lengthB)
        # 这是确保inputA和B都包含了若干个实例,理论上应该都为6

        # After both LiDAR_raw_array and car_state_array reached size_of_input, we can start feed them to the model
        # 如果如预期所示,两个序列都有六个实例被传入,length_min==6,那么说明可以开始执行

        # 到现在为止 都是通用的 必要的 没有需要改变的部分

        if length_min == size_of_input:
            # !!!!!!!!!!!!!!------------------------------需要调整的是模型输入数据部分---------------------------------------------
            



            # 假设inputA和inputB已经是正确形状的NumPy数组
            # inputA的形状应为(size_of_input, 1081)，每个实例包含1081个雷达测量值
            # inputB的形状应为(size_of_input, 2)，每个实例包含1个速度值和1个转向角度值


            # we want there are just one instance, each instance is size_of_input*1081 or 2
            # 每次处理一个批次,每个批次有6个数据,以这六个数据 推理得到下时刻预期的 车辆速度和转向角.三
            # rospy.loginfo(inputA.shape)
            # rospy.loginfo(inputB.shape)
            # inputA: (1, size_of_input, 1081)
            # inputB: (1, size_of_input, 2)
            combined_input = np.concatenate((inputA, inputB), axis=2)

            # combined_input_pad = tensorflow.keras.preprocessing.sequence.pad_sequences(combined_input, maxlen=7, padding='post', value=-100, dtype='float32')

            # 重塑combined_input以符合模型预期的输入形状
            # 假设模型预期的输入形状为(1, size_of_input, 特征数量)
            # 这里的特征数量是1081个雷达测量值加上2个车辆状态值
            model_input = np.reshape(combined_input, (1, size_of_input, 1083))




            # if you want to see more info, change verbose to 1
            # 使用模型进行预测s
            command = overtaking_model.predict(model_input, verbose=0)



            # 每次模型推理确实是基于过去六个时间点的实例（数据），来推导出下一个时间点的车辆预期速度和转向角度

            # rospy.loginfo(command)
            ack_msg = AckermannDriveStamped()
            ack_msg.header.stamp = rospy.Time.now()
            ack_msg.drive.steering_angle = command[0, -1, 1] * 0.24
            ack_msg.drive.speed = command[0, -1, 0] * 17
            drive_pub_red.publish(ack_msg)

            # pass #修改