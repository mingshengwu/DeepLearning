import tensorflow as tf
import random
from mpi4py import MPI
from matplotlib import pyplot as plt
import time
from sys import path
from os import getcwd
from cifar10 import read_data_sets
from SGD_optimizer import SGD_optimizer
from Opserver import Opserver


NUM_CLASSES = 10
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Image processing for training the network. Note the many random
# distortions applied to the image.

# Randomly crop a [height, width] section of the image.
#distorted_image = tf.random_crop(reshaped_image, [total_size, height, height,3])
# test_data = tf.random_crop(test_data, [total_size, height, height,3])
# # Randomly flip the image horizontally.
# distorted_image = tf.image.random_flip_left_right(distorted_image)

# # Because these operations are not commutative, consider randomizing
# # the order their operation.
# distorted_image = tf.image.random_brightness(distorted_image,
#                                            max_delta=63)
# distorted_image = tf.image.random_contrast(distorted_image,
#                                          lower=0.2, upper=1.8)

# # # Subtract off the mean and divide by the variance of the pixels.
# train_data = tf.image.per_image_whitening(distorted_image)



def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  var = tf.get_variable(name, shape, initializer=initializer)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  var = _variable_on_cpu(name, shape,
                         tf.truncated_normal_initializer(stddev=stddev))
  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


# tf Graph input
x = tf.placeholder(tf.float32,[None,32,32,3])
y = tf.placeholder(tf.float32,[None,10])


with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[5, 5, 3, 64],
                                         stddev=1e-4, wd=0.0)
    #conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope.name)

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')
# conv2
with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[5, 5, 64, 64],
                                         stddev=1e-4, wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope.name)

    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
    # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

# local3
with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool2, [-1, 8*8*64])
    #print(reshape.get_shape)
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
# local4
with tf.variable_scope('local4') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)

# softmax, i.e. softmax(WX + b)
with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                          stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    pred = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    
# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
image=tf.placeholder(tf.float32,[32,32,3])
distorted_image = tf.random_crop(image, [24, 24, 3])
distorted_image=tf.image.random_flip_left_right(image)
distorted_image = tf.image.random_brightness(distorted_image,
                                               max_delta=63)
distorted_image = tf.image.random_contrast(distorted_image,
                                             lower=0.2, upper=1.8)
float_image=tf.image.per_image_whitening(distorted_image)
# Initializing the variables
# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
if rank==0:
    cifar10 = read_data_sets("")

config = tf.ConfigProto(device_count={"CPU": 1, "GPU": 0},
                            inter_op_parallelism_threads=1,
                            intra_op_parallelism_threads=1)

sess=tf.Session(config=config)
sess.run(init)

if rank==0:
    tx,ty = cifar10.train.images,cifar10.train.labels
    testx,testy= cifar10.test.images,cifar10.train.labels


def get_batch(batch_size):
    ind=random.sample(range(50000),batch_size)
    sample_x=tx[ind]
    sample_y=ty[ind]
    batch_y=[]
    batch_x=[]
    for x,y in zip(sample_x,sample_y):
        batch_x.append(sess.run(float_image, {image:x}))
        batch_y.append(y)
    return batch_x,batch_y

if rank == 0:
    trainer=SGD_optimizer(cost,[],sess,comm,size,rank)
    total=0
    #bx,by=get_batch(128)
    for i in range(100000):
        starttime=time.time()
        bx,by=get_batch(128)
        endtime=time.time()
        #print "batch Time", endtime-starttime
        #print len(bx)
        starttime=time.time()
        trainer.update(bx,by,x,y)
        d,f=trainer.minimize(0.1)
        endtime=time.time()
        total=total+endtime-starttime
        if not (i+1)%400==0:
            print "Epoch:",i,"Time:", total, "Loss:",f, "Gradient Norm:", d
        else:
            s=time.time()
            test=sess.run(accuracy,{x:cifar10.test.images,y:cifar10.test.labels})
            e=time.time()
            print "Time Used Testing:",e-s,"Accuracy:", test
    comm.send("k", dest=1, tag=11)
#elif rank == 1:
#    dataq=[]
#    while True:
#        while not comm.Iprobe(source=0, tag=11):
#            dataq.append(get_batch(128))
#        rdata = comm.recv(source=0, tag=11)
#        if rdata=="b":
#            comm.send(dataq.pop(), dest=0, tag=11)
#        else:
#            break
else:
    opserver=Opserver(cost,[],sess,comm,size,rank,0,x,y,None)
    opserver.run()













