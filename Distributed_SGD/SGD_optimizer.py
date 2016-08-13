__author__ = 'billywu'


import tensorflow as tf
import numpy as np
import math
import time
from mpi4py import MPI

class SGD_optimizer:
    def __init__(self,cost,feed,sess,comm,size,rank):
        self.cost=cost
        self.sess=sess
        self.gradientEval=0
        self.functionEval=0
        self.innerEval=0
        self.rank=rank
        self.comm=comm
        self.size=size
        v=[]
        self.assign_placeholders=[]
        assign_op=[]
        for t in tf.trainable_variables():
            v.append(sess.run(t))
            self.assign_placeholders.append(tf.placeholder(shape=v[-1].shape,dtype="float32"))
            assign_op.append(t.assign(self.assign_placeholders[-1]))
        self.assign=tf.group(*assign_op)
        self.var=np.array(v)
        # self.var=np.load('var.npy')
        np.save('var.npy',self.var)
        comm.bcast('Init',root=rank)
        self.gradient=tf.gradients(self.cost,tf.trainable_variables())

    def update(self,data_x,data_y,x,y,keep_prob=None):
        start=time.time()
        feed=[]
        s=len(data_x)/(self.size-1)
        if keep_prob!=None:
            kp=True
        else:
            kp=False
        for i in range(self.size):
            feed.append((data_x[i*s:(i+1)*s],data_y[i*s:(i+1)*s],kp))
        self.comm.bcast("U",root=self.rank)
        data=self.comm.scatter(feed,root=self.rank)
        data_x,data_y,kp=data
        if kp:
            self.feed={x:data_x,y:data_y,keep_prob:1.0}
        else:
            self.feed={x:data_x,y:data_y}
        #print "Update Batch:", time.time()-start



    def update_var(self,var=None):
        s=time.time()
        if var==None:
            var=self.var
        self.comm.bcast("W")
        self.comm.bcast(var,root=self.rank)
        feed={}
        for t,v in zip(self.assign_placeholders,var):
            feed[t]=v
        self.sess.run(self.assign,feed)
        #print "Update Var:", time.time()-s


    def getGradient(self,var):
        self.gradientEval=self.gradientEval+1
        s=time.time()
        self.update_var(var)
        self.comm.bcast("G",root=self.rank)
        data=np.array(self.sess.run(self.gradient,self.feed))
        #s=time.time()
        ret=[]
        for gr in data:
            y = self.comm.reduce(gr, op=MPI.SUM,root=self.rank)
            ret.append(y/self.size)
        ret=np.array(ret)
        e=time.time()
        #print "Gradient Time:",e-s
        return ret




    def kill(self):
        self.comm.scatter("K",root=self.rank)

    def getFunction(self,var):
        self.functionEval=self.functionEval+1
        s=time.time()
        self.update_var(var)
        self.comm.bcast("C",root=self.rank)
        data=self.sess.run(self.cost,self.feed)
        y = self.comm.reduce(data, op=MPI.SUM,root=self.rank)
        e=time.time()
        #print "Function Time", e-s
        return y/self.size

    def var_self_inner(self,var_v1,useFlatten=False):
        s=time.time()
        self.innerEval=self.innerEval+1
        ret=0
        for m in var_v1:
            v=np.ravel(m)
            ret=ret+np.inner(v,v)
        e=time.time()
        #print "Inner product:", e-s
        return ret

    def minimize(self,lr):
        g=self.getGradient(self.var)
        d=math.sqrt(self.var_self_inner(g))
        if d>1000:
            g=g/d
        self.var=self.var-g*lr
        f=self.getFunction(self.var)
        return d,f