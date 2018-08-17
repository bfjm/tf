#coding:utf-8
#设损失函数loss=(w+1)^2,令w初值是常数5,反向传播就是就最优w,即求最小loss对应的w值
import tensorflow as tf

LEARNING_RATE_BASE = 0.1#最初学习率
LEARNING_RATE_DECAY = 0.99#学习衰减率
LEARNING_RATE_STEP = 1#喂入多少轮BATE_SIZE后，更新学习率，一般设为：总样本数/BATCH_SIZE
#运行了几轮的BATCH-SIZE的计数器，初值给0，设为不训练
global_step = tf.Variable(0,trainable=False)
#定义指数下降学习率
learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,LEARNING_RATE_STEP,LEARNING_RATE_DECAY,staircase=True)



#定义待优化参数w初值赋5
w = tf.Variable(tf.constant(5,dtype=tf.float32))
#定义待优化函数loss
loss = tf.square(w+1)
#定义反向传播方法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
#生成会话，训练40轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(40):
        sess.run(train_step)
        w_val = sess.run(w)
        loss_val = sess.run(loss)
        print("after %s steps: w is %f,loss is %f."%(i,w_val,loss_val))
