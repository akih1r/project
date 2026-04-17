import sys, os
sys.path.append(os.pardir)
import numpy as np
from collections import OrderedDict
# Affine と numerical_gradient を追加
# ML_model.py の冒頭
from method import Dropout, Relu, softmax, cross_entropy_error, Convolution, Pooling, SoftmaxWithLoss, Affine,im2col, col2im, BatchNormalization
import pickle
print(softmax(np.array([2,3])))




#  input_dim: 入力データの（チャンネル、高さ、幅）
#  conv_param:　畳み込み層のハイパーパラメータ
#  hidden_size:　
#  output_size:　出力層のニューロンの数



class CNN:
    def __init__(self, input_dim=(1, 128, 32), 
                 conv_param={'filter_num':30, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_2={'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
                 hidden_size=100, output_size=35):
        
        
       
        fn1 = conv_param['filter_num']
        fs1 = conv_param['filter_size']
        p1 = conv_param['pad']
        s1 = conv_param['stride']
        
        
        fn2 = conv_param_2['filter_num']
        fs2 = conv_param_2['filter_size']
        p2 = conv_param_2['pad']
        s2 = conv_param_2['stride']

        final_h = 32
        final_w = 8
        pool_output_size = int(fn2 * final_h * final_w)
        



        # 重みの初期化
        self.params = {}
        
        #畳み込み層用
        self.params['W1'] = np.sqrt(2.0 / (input_dim[0] * fs1 * fs1)) * \
                            np.random.randn(fn1, input_dim[0], fs1, fs1)  #(FN, C, FH, FW)
        self.params['b1'] = np.zeros(fn1)
        self.params['gamma1'] = np.ones(fn1) #He用
        self.params['beta1'] = np.zeros(fn1)
        
        
        # W2: (64, 32, 3, 3)
        self.params['W2'] = np.sqrt(2.0 / (fn1 * fs2 * fs2)) * np.random.randn(fn2, fn1, fs2, fs2)
        self.params['b2'] = np.zeros(fn2)
        self.params['gamma2'], self.params['beta2'] = np.ones(fn2), np.zeros(fn2)
        
        #Affin layer1用
        self.params['W3'] = np.sqrt(2.0 / pool_output_size) * \
                            np.random.randn(pool_output_size, hidden_size)
        self.params['b3'] = np.zeros(hidden_size)
        self.params['gamma3'] = np.ones(hidden_size)
        self.params['beta3'] = np.zeros(hidden_size)
        
        #Affin layer2用
        self.params['W4'] = np.sqrt(2.0 / hidden_size) * \
                            np.random.randn(hidden_size, output_size)
        self.params['b4'] = np.zeros(output_size)




        # レイヤの生成
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           conv_param['stride'], conv_param['pad'])
        self.layers['BN1'] = BatchNormalization(self.params['gamma1'], self.params['beta1'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        
        
        self.layers['Conv2'] = Convolution(self.params['W2'], self.params['b2'], s2, p2)
        self.layers['BN2'] = BatchNormalization(self.params['gamma2'], self.params['beta2'])
        self.layers['Relu2'] = Relu()
        self.layers['Pool2'] = Pooling(pool_h=2, pool_w=2, stride=2)
        
        
        self.layers['Affine1'] = Affine(self.params['W3'], self.params['b3'])
        self.layers['BN3'] = BatchNormalization(self.params['gamma3'], self.params['beta3'])
        
        
        self.layers['Relu3'] = Relu()
        self.layers['Dropout1'] = Dropout(dropout_ratio=0.5)
        self.layers['Affine2'] = Affine(self.params['W4'], self.params['b4'])

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x, train_flg=False):
        for key, layer in self.layers.items():
            if "BN" in key or "Dropout" in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t, train_flg=False):
        y = self.predict(x, train_flg)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : 
            t = np.argmax(t, axis=1)
        
        acc = 0.0
        
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx, train_flg=False)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt) 
        
        return acc / x.shape[0]

    def numerical_gradient(self, x, t):
        loss_w = lambda w: self.loss(x, t, train_flg=True)

        grads = {}
        for idx in (1, 2, 3):
            grads['W' + str(idx)] = numerical_gradient(loss_w, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_w, self.params['b' + str(idx)])
            
            if 'gamma' + str(idx) in self.params:
                grads['gamma' + str(idx)] = numerical_gradient(loss_w, self.params['gamma' + str(idx)])
                grads['beta' + str(idx)] = numerical_gradient(loss_w, self.params['beta' + str(idx)])

        return grads

    def gradient(self, x, t):
        
        # forward
        self.loss(x, t, train_flg=True)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        # すべての層に対して、.dW .dbをつくる
        for layer in layers:
            dout = layer.backward(dout)


        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['gamma1'], grads['beta1'] = self.layers['BN1'].dgamma, self.layers['BN1'].dbeta
        
        grads['W2'], grads['b2'] = self.layers['Conv2'].dW, self.layers['Conv2'].db
        grads['gamma2'], grads['beta2'] = self.layers['BN2'].dgamma, self.layers['BN2'].dbeta
        
        grads['W3'], grads['b3'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['gamma3'], grads['beta3'] = self.layers['BN3'].dgamma, self.layers['BN3'].dbeta
        
        grads['W4'], grads['b4'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads
    

    def save_params(self, file_name="params.pkl"):
        params = {}
        
        # 1. 学習した重み(W, b, gamma, beta)を保存
        for key, val in self.params.items():
            params[key] = val
            
        # 2. BatchNormalizationの平均と分散を保存
        params['bn_mean'] = {}
        params['bn_var'] = {}
        
        for key, layer in self.layers.items():
            if "BN" in key: # レイヤ名にBNが含まれていれば
                params['bn_mean'][key] = layer.running_mean
                params['bn_var'][key] = layer.running_var

        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
            
        for key, val in params.items():
            if key not in ['bn_mean', 'bn_var']:
                self.params[key] = val


        # Conv1
        self.layers['Conv1'].W = self.params['W1']
        self.layers['Conv1'].b = self.params['b1']
        self.layers['BN1'].gamma = self.params['gamma1']
        self.layers['BN1'].beta = self.params['beta1']

        # Conv2
        self.layers['Conv2'].W = self.params['W2']
        self.layers['Conv2'].b = self.params['b2']
        self.layers['BN2'].gamma = self.params['gamma2']
        self.layers['BN2'].beta = self.params['beta2']
        
        # Affine1
        self.layers['Affine1'].W = self.params['W3']
        self.layers['Affine1'].b = self.params['b3']
        self.layers['BN3'].gamma = self.params['gamma3']
        self.layers['BN3'].beta = self.params['beta3']

        # Affine2
        self.layers['Affine2'].W = self.params['W4']
        self.layers['Affine2'].b = self.params['b4']
        
        
        if 'bn_mean' in params and 'bn_var' in params:
            for key, layer in self.layers.items():
                if "BN" in key:
                    layer.running_mean = params['bn_mean'][key]
                    layer.running_var = params['bn_var'][key]