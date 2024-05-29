#import numpy as np
import cupy as np


def softmax(y):
    y = y - np.max(y, axis=1, keepdims=True)
    return np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)

def cross_entropy(prob, t):
    return -np.mean(np.log(prob[np.arange(prob.shape[0]), t] + 1e-7))

def softmax_cross_entropy(y, t):
    return cross_entropy(softmax(y), t)


class Layer(object):
    def __init__(self, lr=0.001, momentum=0.9, weight_decay_rate=5e-4):
        self.params = {}
        self.grads = {}
        self.v = None
        self.momentum = momentum
        self.lr = lr
        self.weight_decay_rate = weight_decay_rate

    def update(self):
        if self.v == None:
            self.v = {}
            for k in self.params.keys():
                self.v[k] = np.zeros(shape = self.params[k].shape, dtype = self.params[k].dtype)

        for k in self.params.keys():
            self.v[k] = self.v[k] * self.momentum - self.lr * self.grads[k]
            self.params[k] = (1 - self.lr * self.weight_decay_rate) * self.params[k] + self.v[k]

    def zerograd(self):
        for k in self.params.keys():
            self.grads[k] = np.zeros(shape = self.params[k].shape, dtype = self.params[k].dtype)


class ReLULayer(Layer):
    def __init__(self):
        super(ReLULayer, self).__init__()

    def forward(self, x):
        out = np.maximum(x, 0)
        self.mask = np.sign(out)
        return out

    def backward(self, dout):
        #print('ReLUback')
        #print(np.shape(self.mask))
        #print(np.shape(dout))
        return self.mask * dout


class LinearLayer(Layer):
    def __init__(self, input_dim, output_dim):
        super(LinearLayer, self).__init__()
        self.params['W'] = np.random.normal(scale=np.sqrt(1.0/input_dim), size=(input_dim, output_dim)).astype(np.float32)
        self.params['b'] = np.zeros(shape = (1, output_dim), dtype=np.float32)

    def forward(self, x):
        self.x = x
        #print(np.shape(x))
        #print(np.shape(self.params['W']))
        #print(np.shape(self.params['b']))
        return np.dot(x, self.params['W']) + self.params['b']

    def backward(self, dout):
        self.grads['W'] = np.dot(self.x.T, dout)
        self.grads['b'] = np.sum(dout, axis=0, keepdims=True)
        return np.dot(dout, self.params['W'].T)


class FlattenLayer(Layer):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):
        self.origshape = x.shape
        #print('flatten')
        #print(np.shape(x))
        return np.reshape(x, (x.shape[0], x.size // x.shape[0]))


    def backward(self, dout):
        return np.reshape(dout, self.origshape)


class Convsize3Layer(Layer):
    def __init__(self, input_dim, output_dim):
        super(Convsize3Layer, self).__init__()
        self.params['W'] = np.random.normal(scale=np.sqrt(1.0/input_dim/9), size=(output_dim, input_dim, 3, 3)).astype(np.float32)
        self.params['b'] = np.zeros(shape = (1, output_dim, 1, 1), dtype=np.float32)

    def _im2col(self, data):
        bs, fsize, w, h = data.shape
        #print(np.shape(data))
        #print(w)
        #print(h)
        col = np.zeros((bs, w, h, fsize, 3, 3), dtype=data.dtype)
        data = np.pad(data, [(0, 0), (0, 0), (1, 1), (1, 1)], 'constant')

        for fx in range(3):
            for fy in range(3):
                col[:, :, :, :, fx, fy] = np.transpose(data[:,:,fx:fx+w,fy:fy+h], (0, 2, 3, 1))

        return col


    def _conv(self, data, filt):
        bs, fsize, w, h = data.shape
        mult = np.dot(np.reshape(self._im2col(data), (bs, w, h, fsize*3*3)), np.reshape(filt, (-1, fsize*3*3)).T)
        return np.transpose(mult, (0, 3, 1, 2))

    def forward(self, x):
        self.x = x
        #print('conv')
        return self._conv(x, self.params['W']) + self.params['b']

    def backward(self, dout):
        col = self._im2col(self.x)
        cs = col.shape
        self.grads['W'] = np.reshape(np.dot(np.reshape(np.transpose(dout,(1,0,2,3)),(dout.shape[1],-1)),np.reshape(col,(cs[0]*cs[1]*cs[2],cs[3]*cs[4]*cs[5]))), self.params['W'].shape)
        self.grads['b'] = np.sum(dout, axis=(0,2,3), keepdims=True)
        return self._conv(dout, np.transpose(self.params['W'][:,:,::-1,::-1],(1,0,2,3)))


class Convsize3Layer_nopadding(Layer):
    def __init__(self, input_dim, output_dim):
        super(Convsize3Layer_nopadding, self).__init__()
        self.params['W'] = np.random.normal(scale=np.sqrt(1.0 / input_dim / 9), size=(output_dim, input_dim, 3, 3)).astype(np.float32)
        self.params['b'] = np.zeros(shape=(1, output_dim, 1, 1), dtype=np.float32)

    def _im2col(self, data):
        bs, fsize, w, h = data.shape
        out_h = h - 2  
        out_w = w - 2  
        col = np.zeros((bs, out_w, out_h, fsize, 3, 3), dtype=data.dtype)
        #print('con3')
        #print(np.shape(data))
        #print(w)
        #print(h)
        for fx in range(3):
            for fy in range(3):
                col[:, :, :, :, fx, fy] = np.transpose(data[:, :, fx:fx+out_w, fy:fy+out_h], (0, 2, 3, 1))

        return col

    def _conv(self, data, filt):
        bs, fsize, w, h = data.shape
        out_h = h - 2
        out_w = w - 2
        mult = np.dot(np.reshape(self._im2col(data), (bs, out_w, out_h, fsize * 3 * 3)), np.reshape(filt, (-1, fsize * 3 * 3)).T)
        return np.transpose(mult, (0, 3, 1, 2))

    def forward(self, x):
        self.x = x
        return self._conv(x, self.params['W']) + self.params['b']

    def _backward(self, dout):
        col = self._im2col(self.x)
        cs = col.shape
        self.grads['W'] = np.reshape(np.dot(np.reshape(np.transpose(dout, (1, 0, 2, 3)), (dout.shape[1], -1)), np.reshape(col, (cs[0] * cs[1] * cs[2], cs[3] * cs[4] * cs[5]))), self.params['W'].shape)
        self.grads['b'] = np.sum(dout, axis=(0, 2, 3), keepdims=True)

        
        rot_W = np.rot90(np.rot90(self.params['W']))
        dout_padded = np.pad(dout, ((0, 0), (0, 0), (1, 1), (1, 1)))
        #print('con3p_back')
        #print(np.shape(dout))
        #print(np.shape(dout_padded))
        #print(np.shape(rot_W))
        dx = self._conv(dout_padded, rot_W)
        #print(np.shape(dx))

        return dx

    def backward(self, dout):
        col = self._im2col(self.x)
        cs = col.shape
        self.grads['W'] = np.reshape(np.dot(np.reshape(np.transpose(dout, (1, 0, 2, 3)), (dout.shape[1], -1)), np.reshape(col, (cs[0] * cs[1] * cs[2], cs[3] * cs[4] * cs[5]))), self.params['W'].shape)
        self.grads['b'] = np.sum(dout, axis=(0, 2, 3), keepdims=True)

        
        rot_W = np.rot90(np.rot90(self.params['W'], k=2, axes=(2, 3)))

        
        pad_h = self.params['W'].shape[2] - 1
        pad_w = self.params['W'].shape[3] - 1

        
        dout_padded = np.pad(dout, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')

        
        dx = self._conv(dout_padded, rot_W)

        
        crop_h = (dx.shape[2] - self.x.shape[2]) // 2
        crop_w = (dx.shape[3] - self.x.shape[3]) // 2
        dx = dx[:, :, crop_h:-crop_h or None, crop_w:-crop_w or None]
        #print('con3p_back')
        #print(np.shape(dout))
        #print(np.shape(dout_padded))
        #print(np.shape(rot_W))
        dx = self._conv(dout_padded, rot_W)
        return dx






class Convsize2Layer_nopadding(Layer):
    def __init__(self, input_dim, output_dim):
        super(Convsize2Layer_nopadding, self).__init__()
        self.params['W'] = np.random.normal(scale=np.sqrt(1.0 / input_dim / 4), size=(output_dim, input_dim, 2, 2)).astype(np.float32)
        self.params['b'] = np.zeros(shape=(1, output_dim, 1, 1), dtype=np.float32)

    def _im2col(self, data):
        bs, fsize, w, h = data.shape
        out_h = h - 1  
        out_w = w - 1  
        col = np.zeros((bs, out_w, out_h, fsize, 2, 2), dtype=data.dtype)
        #print('con2')
        #print(np.shape(data))
        #print(w)
        #print(h)
        for fx in range(2):
            for fy in range(2):
                col[:, :, :, :, fx, fy] = np.transpose(data[:,:,fx:fx+out_w,fy:fy+out_h], (0, 2, 3, 1))

        return col

    def _conv(self, data, filt):
        bs, fsize, w, h = data.shape
        out_h = h - 1
        out_w = w - 1
        mult = np.dot(np.reshape(self._im2col(data), (bs, out_w, out_h, fsize * 2 * 2)), np.reshape(filt, (-1, fsize * 2 * 2)).T)
        return np.transpose(mult, (0, 3, 1, 2))

    def forward(self, x):
        self.x = x
        return self._conv(x, self.params['W']) + self.params['b']

    def backward(self, dout):
        col = self._im2col(self.x)
        cs = col.shape
        self.grads['W'] = np.reshape(np.dot(np.reshape(np.transpose(dout, (1, 0, 2, 3)), (dout.shape[1], -1)), np.reshape(col, (cs[0] * cs[1] * cs[2], cs[3] * cs[4] * cs[5]))), self.params['W'].shape)
        self.grads['b'] = np.sum(dout, axis=(0, 2, 3), keepdims=True)

        
        rot_W = np.rot90(np.rot90(self.params['W']))
        dout_padded = np.pad(dout, ((0, 0), (0, 0), (1, 1), (1, 1)))
        #print('con2p_back')
        #print(np.shape(dout))
        #print(np.shape(dout_padded))
        #print(np.shape(rot_W))
        dx = self._conv(dout_padded, rot_W)
        #print(np.shape(dx))

        return dx



class PoolingLayer(Layer):
    def __init__(self, ksize):
        super(PoolingLayer,self).__init__()
        self.ksize = ksize

    def _im2col(self, data):
        k = self.ksize
        bs, fsize, w, h = data.shape

        #assert w%k ==0 and h%k == 0, 'input image size should be multiple of kernel size'
        ow, oh = w//k, h//k
        col = np.zeros((bs,fsize,ow,oh,k,k),dtype=data.dtype)

        for fx in range(k):
            for fy in range(k):
                col[:,:,:,:,fx,fy] = data[:,:,fx:fx+ow*k:k,fy:fy+oh*k:k]
        return col

    def _col2im(self, col):
        k = self.ksize
        bs,fsize,ow,oh=col.shape[0],col.shape[1],col.shape[2],col.shape[3]
        data = np.zeros((bs,fsize,ow*k,oh*k),dtype=col.dtype)
        for fx in range(k):
            for fy in range(k):
                data[:,:,fx:fx+ow*k:k,fy:fy+oh*k:k] = col[:,:,:,:,fx,fy]
        return data


class AvgPoolingLayer(PoolingLayer):
    def __init__(self, ksize):
        super(AvgPoolingLayer, self).__init__(ksize)

    def forward(self, x):
        #print('avgpooling')
        #print(np.shape(x))
        return np.mean(self._im2col(x), axis=(4,5))

    def backward(self, dout):
        #print('avgback')
        #print(np.shape(dout))
        return np.repeat(np.repeat(dout, self.ksize, axis=2), self.ksize, axis=3) / (self.ksize ** 2)


class MaxPoolingLayer(PoolingLayer):
    def __init__(self, ksize):
        super(MaxPoolingLayer, self).__init__(ksize)

    def forward(self, x):
        col = self._im2col(x)
        col = np.reshape(col, (col.shape[0], col.shape[1], col.shape[2], col.shape[3], -1))
        self.maxinds = np.argmax(col, axis=-1).flatten()
        #print('maxpooling')
        return np.max(col, axis=-1)

    def backward(self, dout):
        #print('maxback1')
        #print(np.shape(dout))
        bs, fsize, ow, oh = dout.shape
        mask = np.zeros((self.maxinds.size, self.ksize**2), dtype=dout.dtype)
        mask[np.arange(mask.shape[0]), self.maxinds] = 1
        mask = self._col2im(np.reshape(mask, (bs, fsize, ow, oh, self.ksize, self.ksize)))
        #print(1112)
        #print('maxback2')
        #print(np.shape(mask))

        return mask * np.repeat(np.repeat(dout, self.ksize, axis=2), self.ksize, axis=3)


class Sequential:
    def __init__(self, layers = []):
        self.layers = layers

    def addlayer(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        for l in self.layers:
            #print('sequential')
            #print(np.shape(x))
            x = l.forward(x)
        return x

    def backward(self, dout):
        for l in reversed(self.layers):
            dout = l.backward(dout)
        return dout

    def update(self):
        for l in self.layers:
            l.update()

    def zerograd(self):
        for l in self.layers:
            l.zerograd()


class Classifier:
    def __init__(self, model):
        self.model = model

    def predict(self, x, y):
        h = self.model.forward(x)
        pred = np.argmax(h, axis=1)
        acc = 1.0 * np.where(pred == y)[0].size / h.shape[0]
        loss = softmax_cross_entropy(h, y)
        return loss, acc

    def update(self, x, y):
        self.model.zerograd()
        #print('classifier')
        #print(np.shape(x))
        h = self.model.forward(x)
        pred = np.argmax(h, axis=1)
        acc = 1.0 * np.where(pred == y)[0].size / h.shape[0]
        prob = softmax(h)
        loss = cross_entropy(prob, y)

        dout = prob
        dout[np.arange(dout.shape[0]), y] -= 1

        self.model.backward(dout / dout.shape[0])
        self.model.update()

        return loss, acc
