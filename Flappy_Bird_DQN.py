#!/usr/bin/env python
from __future__ import print_function

import tensorflow as tf
import cv2
import sys

sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque

GAME = 'bird'
ACTIONS = 2
GAMMA = 0.99
OBSERVE = 10000.
EXPLORE = 3000000.
FINAL_EPSILON = 0.0001
INITIAL_EPSILON = 0.1
REPLAY_MEMORY = 50000
BATCH = 32
FRAME_PER_ACTION = 1


def weight_variable(shape):
    '''产生截尾正态分布3D矩阵用于初始化卷积核'''
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    '''初始化阀值'''
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    '''计算卷积：
    x是输入的tensor
    W是过滤器也就是卷积核
    stried为卷积移动步长
    padding = ‘SAME’表示卷积后的结果与输入结果的尺寸一样
    '''
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

def max_pool_2x2(x):
    '''最大池化函数，x是输入的tensor，ksize表示池化窗口的尺寸，strides表示池化窗口的移动步长'''
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

def createNetwork():
    ''' 定义深度神经网络的参数和偏置
        1.其中bias就是平常所说的阀值
        2.激活函数relu，与西瓜书的sigmoid类似
        3.池化是为了降低维数，并提取主要特征
        4.在理解CNN时借助标量来理解，只是现在标量换成了张量，其他都是一一对应的
        5.在训练过程中需要不断的跟新的参数为所有的W和B
        6.一次训练从样本池中选取32batch同时训练，累计32个batch的输出误差进行优化
        7.CNN的权共享是指每个batch都用相同的W和b训练
        8.如果输入是32个batch将创建地位一样的32个神经网络'''

    #定义第一个卷积核和偏置量
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])
    #定义第二个卷积核与偏置量
    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])
    # 定义第三个卷积核与偏置量
    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    #定义第一个全连接层与偏置量
    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])
    # 定义第二个全连接层与偏置量
    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    #输入
    s = tf.placeholder("float", [None, 80, 80, 4])

    # 隐藏层
    #将矩阵中每行的非最大值置零
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    #对结果进行池化操作
    h_pool1 = max_pool_2x2(h_conv1)
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # 输出层
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2
    #s输入的图片，readout输出层，h_fc1全连接层一
    return s, readout, h_fc1

def trainNetwork(s, readout, h_fc1, sess):
    ''' 定义损失函数
        1.None表示这个占位符的数值对应的轴维数未至，计算过程中传入多大就是多大
        2.在这个程序中，选择的batch是32，因此None的位置将是32
        3.a代表每个batch采取的动作，每一个batch的动作要么是[0,1],要么是[1,0]
        4.每个batch经过神经网络也会产生两个输出，第一个位置代表采取不采取动作产生的奖励，第二个位置表示条约将会得到的奖励
        5.readout_action表示32个batch的累计奖励
        6.y表示实际奖励'''

    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.multiply(readout, a),axis=1)

    #cost节点需要三个输入，a，y，readout_acton，而readout_action需要s作为输入
    #因此train_step需要三个输入，a,y,s，计算32组数据损失函数的平均值，作为优化目标调用Adam优化算法进行优化
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    # 开启游戏模拟器，会打开一个模拟器的窗口，实时显示游戏的信息
    game_state = game.GameState()
    # 创建双端队列用于存放replay memory
    D = deque()
    # 获取游戏的初始状态，设置动作为不执行跳跃，并将初始状态修改成80*80*4大小
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    # frame_step函数输入的2x1数组之和必须等于1，第一个数为1表示do nothing，第二个数为一表示执行跳跃的动作
    #x_t,表示输出的图片，r_0表示奖励，terminal表示状态是成功还是失败，如果terminal是失败则没有s_t+1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    #将图片换成80x80尺寸
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    #将颜色值表示成黑白两色
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    #第一次输入采用四张同样的图片
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    # 用于加载或保存网络参数
    saver = tf.train.Saver()
    #tensorflow初始化命令，以使所有变量可以用
    sess.run(tf.initialize_all_variables())

    #载入之前训练好的神经网络
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    # 开始训练
    epsilon = INITIAL_EPSILON
    t = 0
    while "flappy bird" != "angry bird":
        # 使用epsilon贪心策略选择一个动作

        #输入图片s_t计算输出
        readout_t = readout.eval(feed_dict={s: [s_t]})[0]#为什么要[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0
        # FRAME_PER_ACTION什么意思？
        if t % FRAME_PER_ACTION == 0:
            # 执行一个随机动作
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[random.randrange(ACTIONS)] = 1
            # 由神经网络计算的Q(s,a)值选择对应的动作
            else:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
        else:
            a_t[0] = 1  # 不执行跳跃动作

        # 随游戏的进行，不断降低epsilon，减少随机动作
        #分为三个阶段：
        #   1观察期：不对神经网络更新只是采用
        # Ⅱ探索期：当进入探索期时，epislon会开始减小
        #Ⅲ训练期：当epsilon 降到一定程度时，进入训练期，这是epsilon将不再减小
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # 执行选择的动作，并获得下一状态及回报，如果状态是死亡，那么x_t1将会是初始界面
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)),cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        # 将状态转移过程存储到D中，用于更新参数时采样，如果内存溢出了，就丢掉队列中的第一个数据
        D.append((s_t, a_t, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # 过了观察期，进行神经网络参数的更新
        if t > OBSERVE:
            # 从D中随机采样32个batch，用于参数更新，这32个batch并不连续，而是随机抽取
            minibatch = random.sample(D, BATCH)

            # 分别将当前状态、采取的动作、获得的回报、下一状态分组存放
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            # 计算Q(s,a)的新值
            y_batch = []
            readout_j1_batch = readout.eval(feed_dict={s: s_j1_batch})
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                # 如果游戏结束，则只有反馈值
                if terminal:
                    y_batch.append(r_batch[i])
                #如果没有结束则用神经网络更新奖励，而不用模拟器更新，因为随着迭代的进行，神经网络和模拟器达到的效果将会一样
                else:
                    y_batch.append(r_batch[i] +GAMMA * np.max(readout_j1_batch[i]))

            # 使用梯度下降更新网络参数
            train_step.run(feed_dict={
                y: y_batch,
                a: a_batch,
                s: s_j_batch}
            )

        # 状态发生改变，用于下次循环
        s_t = s_t1
        t += 1

        # 每进行100000次迭代，保留一下网络参数
        if t % 100000 == 0:
            saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step=t)

        # 打印游戏信息
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state, \
              "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
              "/ Q_MAX %e" % np.max(readout_t))


def playGame():
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = createNetwork()
    trainNetwork(s, readout, h_fc1, sess)


if __name__ == "__main__":
    #playGame()
    reader = tf.train.NewCheckpointReader('saved_networks/bird-dqn-2400000')

    #获取所有变量的名称以及维度信息，这个程序中没有设置变量名称，所以要先获取一下系统取的变量是什么
    all_variables = reader.get_variable_to_shape_map()
    #获取你想要的张量
    w1 = reader.get_tensor("Variable_6/Adam_1")
    print(w1)