__author__ = 'billywu'

import tensorflow as tf
import numpy as np
import time
from mpi4py import MPI


class Opserver:
    def __init__(self, cost,feed,sess,comm,size,rank,root,x,y,keep_prob):
        self.cost=cost
        self.feed=feed
        self.sess=sess
        self.rank=rank
        self.comm=comm
        self.root=root
        self.size=size
        v=[]
        self.assign_placeholders=[]
        assign_op=[]
        for t in tf.trainable_variables():
            v.append(sess.run(t))
            self.assign_placeholders.append(tf.placeholder(shape=v[-1].shape,dtype="float32"))
            assign_op.append(t.assign(self.assign_placeholders[-1]))
        self.assign=tf.group(*assign_op)
        self.gradient=tf.gradients(cost,tf.trainable_variables())
        comm.bcast([],root=root)
        self.var=np.load('var.npy')
        self.old_grad=None
        self.x=x
        self.y=y
        self.keep_prob=keep_prob

    def update_var(self,var=None):
        feed={}
        for t,v in zip(self.assign_placeholders,var):
            feed[t]=v
        self.sess.run(self.assign,feed)

    def run(self):
        while (True):
            data=self.comm.bcast('None',root=self.root)
            if data=="G":
                s=time.time()
                g=np.array(self.sess.run(self.gradient,self.feed))
                #print time.time()-s
                ss=time.time()
                for gr in g:
                    y = self.comm.reduce(gr, op=MPI.SUM,root=self.root)
                e=time.time()
                #print "Gradient Server Compute",e-ss
            elif data=="C":
                c=self.sess.run(self.cost,self.feed)
                y = self.comm.reduce(c, op=MPI.SUM,root=self.root)
                e=time.time()
            elif data=="K":
                break
            elif data=="U":
                fdata=self.comm.scatter([],root=self.root)
                data_x,data_y,kp=fdata
                if kp:
                    self.feed={self.x:data_x,self.y:data_y,self.keep_prob:1.0}
                else:
                    self.feed={self.x:data_x,self.y:data_y}
            elif data[0]=="W":
                s=time.time()
                fdata=self.comm.bcast([],root=self.root)
                self.update_var(fdata)
                e=time.time()
                #print "Update Time", e-s
        print "Core,", self.rank, "Finish"

