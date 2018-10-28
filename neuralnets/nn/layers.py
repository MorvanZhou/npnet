import numpy as np
import neuralnets as nn


class Layer:
    def __init__(self, w_shape, activation, w_initializer, b_initializer, use_bias):
        self.order = None
        self.name = None
        self.w = np.empty(w_shape, dtype=np.float32)
        if use_bias:
            shape = [1]*len(w_shape)
            shape[-1] = w_shape[-1]     # only have bias on the last dimension
            self.b = np.empty(shape, dtype=np.float32)
        self.use_bias = use_bias

        if activation is None:
            self._a = nn.act.Linear()
        elif isinstance(activation, nn.act.Activation):
            self._a = activation
        else:
            raise TypeError

        if w_initializer is None:
            nn.init.RandomUniform().initialize(self.w)
        elif isinstance(w_initializer, nn.init.BaseInitializer):
            w_initializer.initialize(self.w)
        else:
            raise TypeError

        if use_bias:
            if b_initializer is None:
                nn.init.Constant(0.1).initialize(self.b)
            elif isinstance(b_initializer, nn.init.BaseInitializer):
                b_initializer.initialize(self.b)
            else:
                raise TypeError

        self._wx_b = None
        self._activated = None
        self._x = None

    def forward(self, x):
        raise NotImplementedError

    def backward(self, delta):
        raise NotImplementedError

    def _process_input(self, x):
        if isinstance(x, np.ndarray):
            self.order = 0  # use layer input's information to set layer order
            _x = x.astype(np.float32)
        elif isinstance(x, nn.Variable):
            # x is Variable, extract _x value from x.data
            self.order = x.info["new_layer_order"]
            _x = x.data
        else:
            raise ValueError
        return _x

    def __call__(self, x):
        return self.forward(x)


class Dense(Layer):
    def __init__(self,
                 n_in,
                 n_out,
                 activation=None,
                 w_initializer=None,
                 b_initializer=None,
                 use_bias=True,
                 ):
        super().__init__(
            w_shape=(n_in, n_out),
            activation=activation,
            w_initializer=w_initializer,
            b_initializer=b_initializer,
            use_bias=use_bias)

        self._n_in = n_in
        self._n_out = n_out

    def forward(self, x):
        self._x = self._process_input(x)
        self._wx_b = self._x.dot(self.w)
        if self.use_bias:
            self._wx_b += self.b

        self._activated = self._a(self._wx_b)   # if act is None, act will be Linear
        out = nn.Variable(self._activated)
        out.info["new_layer_order"] = self.order + 1
        return out

    def backward(self, delta):
        delta = delta * self._a.derivative(self._wx_b)
        grads = {"gw": self._x.T.dot(delta)}
        if self.use_bias:
            grads["gb"] = np.sum(delta, axis=0, keepdims=True)
        delta_ = delta.dot(self.w.T)
        return delta_, grads


class SlowConv2d(Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format='channels_last',
                 activation=None,
                 w_initializer=None,
                 b_initializer=None,
                 use_bias=True,
                 ):
        if isinstance(kernel_size, (tuple, list)):
            self.kernel_size = tuple(kernel_size)
        elif isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            raise TypeError

        if isinstance(strides, (tuple, list)):
            self.strides = tuple(strides)
        elif isinstance(strides, int):
            self.strides = (strides, strides)
        else:
            raise TypeError

        super().__init__(
            w_shape=(in_channels,) + self.kernel_size + (out_channels,),
            activation=activation,
            w_initializer=w_initializer,
            b_initializer=b_initializer,
            use_bias=use_bias)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding.lower()
        assert padding in ("valid", "same"), ValueError

        self.data_format = data_format.lower()
        assert self.data_format in ("channels_last", "channels_first"), ValueError
        self._padded = None
        self._pad_top_idx = None
        self._pad_bottom_idx = None
        self._pad_left_idx = None
        self._pad_right_idx = None

    def forward(self, x):
        self._x = self._process_input(x)
        if self.data_format == "channels_first":
            # [batch, channel, height, width] => [batch, height, width, channel]
            self._x = np.transpose(self._x, (0, 2, 3, 1))
        self._padded, tmp_out = self.padded_and_tmp_out(self._x)

        # convolution
        for n_flt in range(self.w.shape[-1]):   # for each filter
            flt = self.w[:, :, :, n_flt:n_flt+1]        # [in_channels, height, width, 1]
            for i in range(0, tmp_out.shape[1]):
                for j in range(0, tmp_out.shape[2]):
                    i_ = i*self.strides[0]
                    j_ = j*self.strides[1]
                    tmp_out[:, i, j, n_flt] = np.sum(
                        flt*self._padded[
                            :,
                            i_:i_+self.kernel_size[0],
                            j_:j_+self.kernel_size[1],
                            :], axis=(1, 2, 3))
        self._wx_b = tmp_out
        if self.use_bias:   # tied biases
            self._wx_b += self.b

        self._activated = self._a(self._wx_b)
        out = nn.Variable(self._activated)
        out.info["new_layer_order"] = self.order + 1
        return out

    def backward(self, delta):
        delta = delta * self._a.derivative(self._wx_b)

        # grads of w
        dw = np.empty_like(self.w)
        for n_delta in range(self.w.shape[-1]):   # for each filter
            dlt = delta[:, :, :, n_delta:n_delta+1]        # [batch, height, width]
            for i in range(0, dw.shape[1]):
                for j in range(0, dw.shape[2]):
                    i_ = i * self.strides[0]
                    j_ = j * self.strides[1]
                    dw[:, i, j, n_delta] = np.sum(
                        dlt*self._padded[
                                :,
                                i_:i_+dlt.shape[1],
                                j_:j_+dlt.shape[2],
                                :], axis=(1, 2, 3))

        grads = {"gw": dw}
        if self.use_bias:   # tied biases
            grads["gb"] = np.sum(delta, axis=(0, 1, 2), keepdims=True)

        delta_ = np.zeros_like(self._x)
        tmp_delta_ = self._padded.copy()
        tmp_delta_[:, self._pad_top_idx:] = delta
        delta_[:, ]     # Todo: not complete

        return delta_, grads

    def padded_and_tmp_out(self, img):
        # according to: http://machinelearninguru.com/computer_vision/basics/convolution/convolution_layer.html
        batch, h, w = img.shape[:3]
        (fh, fw), (sh, sw) = self.kernel_size, self.strides

        if self.padding == "same":
            out_h = int(np.ceil(h / sh))
            out_w = int(np.ceil(w / sw))
            ph = int(np.max([0, (out_h - 1) * sh + fh - h]))
            pw = int(np.max([0, (out_w - 1) * sw + fw - w]))
            pt, pl = int(np.floor(ph / 2)), int(np.floor(pw / 2))
            pb, pr = ph - pt, pw - pl
            padded_img = np.zeros((batch, ph, pw, img.shape[3]), dtype=np.float32)
            padded_img[:, pt:-pb, pl:-pr, :] = img
            out_img = np.empty(img.shape[:3] + (self.out_channels,), dtype=np.float32)
        else:   # valid padding
            out_h = int(np.ceil((h-fh+1)/sh))
            out_w = int(np.ceil((w-fw+1)/sw))
            pt, pb, pl, pr = 0, img.shape[1], 0, img.shape[2]
            padded_img = img
            out_img = np.empty((batch, out_h, out_w, self.out_channels), dtype=np.float32)
        self._pad_top_idx = pt
        self._pad_bottom_idx = pb
        self._pad_left_idx = pl
        self._pad_right_idx = pr
        return padded_img, out_img



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from skimage import io, color

    img = io.imread('../../image.jpg')  # Load the image
    img = color.rgb2gray(img)[None, :, :, None]  # Convert the image to grayscale (1 channel)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    l = SlowConv2d(1, 2, kernel_size=3, strides=5, padding="same", activation=nn.act.relu, data_format="channels_last",use_bias=True)
    l.w[0, :, :, 0] = kernel
    out = l.forward(img)
    # plt.imshow(np.squeeze(out.data), cmap=plt.cm.gray)
    # plt.show()
    l.backward(out.data)
