import numpy as np

from Q import shift_2_POT
'''
filter [filter_h, filter_w, ch_in, ch_out] 
to [filter_h*filter_w*ch_in, ch_out]
'''
def filter2col(filter):
    # filter_ = np.transpose(filter, axes=(3,2,0,1)) #[R, S, C, K] -> [K, C, R, S]
    k_, c_, r_, s_ = filter.shape
    return np.transpose(filter.reshape(k_, c_*r_*s_), axes=(1,0))

'''
relayout images data to convolution tiles.
flat a RSC tile to a row
    |              |
    |              |
-------------------------
    | image data   |
    |              |
    | H            |
    |      W       |
-------------------------
    |              |
    |              |
'''
def im2col(img_data, filter_shape, output_shape, pad, stride, dtype=None):
    if dtype is None:
        dtype = img_data.dtype
    N, C, H, W = img_data.shape
    K, C_, R, S = filter_shape
    P, Q = output_shape
    padding_left, padding_top, padding_right, padding_bottom = pad
    u, v = stride
    img_matrix = np.zeros((N*P*Q, R*S*C), dtype=dtype)
    step_col = R*S
    step_row = P*Q
    for p in range(P): # traversequantized_depthwise_conv2d_relu each row of img_matrix
        for q in range(Q): # traverse each coloumn of img_matrix
            # left_top x of tile on the image
            i = q * v - padding_left
            # left_top y of tile on the image
            j = p * u - padding_top
            col = 0 
            row = Q*p + q
            for r in range(R): # tranverse each row of tile
                for s in range(S): # traverse each coloumn of tile
                    img_x = s + i
                    img_y = r + j
                    if 0 <= img_x < W and 0 <= img_y < H:
                        img_matrix[row::step_row, col::step_col] = img_data[:, :, img_y, img_x]
                    col += 1
    return img_matrix

'''
input_data: [N, H, W, C] format
'''
def max_pooling2d(input_data, pool_size=(2, 2), strides=(2, 2), padding='VALID'):
    N, C, H, W = input_data.shape
    u, v = strides # strides in vertical and horizontal direction
    assert u==v, "only support same strides in vertical and horizontal direction"
    R, S = pool_size
    padding_left = padding_right = padding_top = padding_bottom = 0
    if padding == 'SAME':
        pass
    elif padding == 'VALID':
        P = int(np.floor((H-R)/u) + 1)
        Q = int(np.floor((W-S)/v) + 1)
    pad = (padding_left, padding_top, padding_right, padding_bottom)
    im_matrix = im2col(input_data, (-1, C, R,S), (P,Q), pad, (u,v)) # [Q*P*N, R*S*C]
    im_matrix = im_matrix.reshape(Q*P*N*C, R*S) # [Q*P*N*C, 1]
    return np.amax(im_matrix, axis=1).reshape(N, C, P, Q)

'''
input: [N, H, W, C]
filter: [R, S, C, K]
output: [N, P, Q, K]
'''
def conv2d(input, filter, strides=[1,1], padding='SAME', bias=None):
    N, C, H, W = input.shape
    K, C_, R, S = filter.shape
    assert C==C_, "input's channel and filter's channel must be same."
    u, v = strides # strides in vertical and horizontal direction
    assert u==v, "only support same strides in vertical and horizontal direction"
    padding_left = padding_right = padding_top = padding_bottom = 0
    if padding == 'SAME':
        P = int(np.floor((H-1)/u+1))
        Q = int(np.floor((W-1)/v+1))
        padding_needed_rows = (P-1) * u + R - H
        padding_needed_cols = (Q-1) * v + S - W
        padding_top = 0 if padding_needed_rows < 0 else int(padding_needed_rows // 2)
        padding_bottom = 0 if padding_needed_rows < 0 else int(padding_needed_rows - padding_top)
        padding_left = 0 if padding_needed_cols < 0 else int(padding_needed_cols // 2)
        padding_right = 0 if padding_needed_cols < 0 else int(padding_needed_rows - padding_left)
        print(padding_top, padding_bottom, padding_left, padding_right)
    elif padding == 'VALID':
        P = int(np.floor((H-R)/u) + 1)
        Q = int(np.floor((W-S)/v) + 1)
    pad = (padding_left, padding_top, padding_right, padding_bottom)
    im_matrix = im2col(input, filter.shape, (P,Q), pad, (u,v))
    filter_matrix = filter2col(filter)
    feature_maps = np.matmul(im_matrix, filter_matrix, dtype=np.float32)
    feature_maps = feature_maps.reshape(N, P, Q, K)
    if bias is not None:
        feature_maps += bias
    feature_maps = feature_maps.transpose((0, 3, 1, 2))
    return feature_maps


def conv2d_pytorch(input, filter, strides=[1,1], padding='SAME', bias=None):
    N, C, H, W = input.shape
    K, C_, R, S = filter.shape
    assert C==C_, "input's channel and filter's channel must be same."
    u, v = strides # strides in vertical and horizontal direction
    assert u==v, "only support same strides in vertical and horizontal direction"
    padding_left = padding_right = padding_top = padding_bottom = 0
    if padding == 'SAME':
        P = int(np.floor((H-1)/u+1))
        Q = int(np.floor((W-1)/v+1))
        padding_needed_rows = (P-1) * u + R - H
        padding_needed_cols = (Q-1) * v + S - W
        padding_bottom = 0 if padding_needed_rows < 0 else int(padding_needed_rows // 2)
        padding_top = 0 if padding_needed_rows < 0 else int(padding_needed_rows - padding_bottom)
        padding_right = 0 if padding_needed_cols < 0 else int(padding_needed_cols // 2)
        padding_left = 0 if padding_needed_cols < 0 else int(padding_needed_rows - padding_right)
    elif padding == 'VALID':
        P = int(np.floor((H-R)/u) + 1)
        Q = int(np.floor((W-S)/v) + 1)
    pad = (padding_left, padding_top, padding_right, padding_bottom)
    im_matrix = im2col(input, filter.shape, (P,Q), pad, (u,v))
    filter_matrix = filter2col(filter)
    feature_maps = np.matmul(im_matrix, filter_matrix, dtype=np.float32)
    feature_maps = feature_maps.reshape(N, P, Q, K)
    if bias is not None:
        feature_maps += bias
    feature_maps = feature_maps.transpose((0, 3, 1, 2))
    return feature_maps


def separable_conv2d(input_data, depthwise_filter, pointwise_filter, strides=[1, 1], padding='SAME'):
    """
    input: N H W C
    depthwise_filter: HH * WW * C * 1
    pointwise_filter: 1 * 1 * C * F
    """
    N, C, H, W = input_data.shape
    dF, dC, HH, WW = depthwise_filter.shape
    F, pC, pH, pW = pointwise_filter.shape
    assert C == dC == pC, "input channel, depthwise_filter input channel, pointwise_filter input channel must be same"
    assert dF == 1, "depthwise_filter output channel must be one"
    assert pH == pW == 1, "pointwise_filter kernel size must be one"
    assert strides[0] == strides[1], "strides must have same size"
    stride = strides[0]

    padding_left = padding_right = padding_top = padding_bottom = 0

    if padding == 'SAME':
        P = int(np.floor((H - 1) / stride + 1))
        Q = int(np.floor((W - 1) / stride + 1))
        padding_needed_rows = (P - 1) * stride + HH - H
        padding_needed_cols = (Q - 1) * stride + WW - W
        padding_top = 0 if padding_needed_rows < 0 else int(padding_needed_rows // 2)
        padding_bottom = 0 if padding_needed_rows < 0 else int(padding_needed_rows - padding_top)
        padding_left = 0 if padding_needed_cols < 0 else int(padding_needed_cols // 2)
        padding_right = 0 if padding_needed_cols < 0 else int(padding_needed_rows - padding_left)
    elif padding == 'VALID':
        # P = int(np.floor((H - HH) / stride) + 1)
        # Q = int(np.floor((W - WW) / stride) + 1)
        pass
    pad = (padding_left, padding_top, padding_right, padding_bottom)
    input_pad = np.pad(input_data, ((0, 0), (pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), "constant",
                       constant_values=((0, 0), (0, 0), (0, 0), (0, 0)))
    Hp = 1 + int((H + (pad[1] + pad[3]) - HH) / stride)
    Wp = 1 + int((W + (pad[0] + pad[2]) - WW) / stride)

    out = np.zeros((N, Hp, Wp, F))
    for iX in range(N):
        for iF in range(F):
            for iH in range(Hp):
                for iW in range(Wp):
                    out[iX, iH, iW, iF] = np.sum(
                        input_pad[iX, (iH * stride):(iH * stride + HH), (iW * stride):(iW * stride + WW), :]
                        * depthwise_filter[:, :, :, 0] * pointwise_filter[0, 0, :, iF])

    return out.transpose((0, 3, 1, 2))


def depthwise_conv2d(input_data, depthwise_filter, strides=[1, 1], padding='SAME', bias=None):
    """
        input_data: N H W C
        depthwise_filter: HH * WW * C * multiplifier，multiplifier represents channel multiplifer, but not support here
    """
    N, C, H, W = input_data.shape
    _, dC, HH, WW = depthwise_filter.shape
    assert C == dC, "input channel and depthwise_filter input channel must be same"
    assert strides[0] == strides[1], "strides must have same size"

    stride = strides[0]
    padding_left = padding_right = padding_top = padding_bottom = 0

    if padding == 'SAME':
        P = int(np.floor((H - 1) / stride + 1))
        Q = int(np.floor((W - 1) / stride + 1))
        padding_needed_rows = (P - 1) * stride + HH - H
        padding_needed_cols = (Q - 1) * stride + WW - W
        padding_top = 0 if padding_needed_rows < 0 else int(padding_needed_rows // 2)
        padding_bottom = 0 if padding_needed_rows < 0 else int(padding_needed_rows - padding_top)
        padding_left = 0 if padding_needed_cols < 0 else int(padding_needed_cols // 2)
        padding_right = 0 if padding_needed_cols < 0 else int(padding_needed_rows - padding_left)
    elif padding == 'VALID':
        # P = int(np.floor((H - HH) / stride) + 1)
        # Q = int(np.floor((W - WW) / stride) + 1)
        pass
    pad = (padding_left, padding_top, padding_right, padding_bottom)
    input_pad = np.pad(input_data, ((0, 0), (pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), "constant",
                       constant_values=((0, 0), (0, 0), (0, 0), (0, 0)))
    Hp = 1 + int((H + (pad[1] + pad[3]) - HH) / stride)
    Wp = 1 + int((W + (pad[0] + pad[2]) - WW) / stride)

    out = np.zeros((N, Hp, Wp, C))
    for iX in range(N):
        for iC in range(C):
            for iH in range(Hp):
                for iW in range(Wp):
                    out[iX, iH, iW, iC] = np.sum(
                        input_pad[iX, (iH * stride):(iH * stride + HH), (iW * stride):(iW * stride + WW), iC]
                        * depthwise_filter[:, :, iC, 0])

    if bias is not None:
        print("shape of out:{}*************shape of bias:{}".format(out.shape, bias.shape))
        out += bias

    return out.transpose((0, 3, 1, 2))


def depthwise_conv2d_pytorch(input_data, depthwise_filter, strides=[1, 1], padding='SAME', bias=None):
    """
        input_data: N H W C
        depthwise_filter: HH * WW * C * multiplifier，multiplifier represents channel multiplifer, but not support here
    """
    N, C, H, W = input_data.shape
    _, dC, HH, WW = depthwise_filter.shape
    assert C == dC, "input channel and depthwise_filter input channel must be same"
    assert strides[0] == strides[1], "strides must have same size"

    stride = strides[0]
    padding_left = padding_right = padding_top = padding_bottom = 0

    if padding == 'SAME':
        P = int(np.floor((H - 1) / stride + 1))
        Q = int(np.floor((W - 1) / stride + 1))
        padding_needed_rows = (P - 1) * stride + HH - H
        padding_needed_cols = (Q - 1) * stride + WW - W
        padding_bottom = 0 if padding_needed_rows < 0 else int(padding_needed_rows // 2)
        padding_top = 0 if padding_needed_rows < 0 else int(padding_needed_rows - padding_bottom)
        padding_right = 0 if padding_needed_cols < 0 else int(padding_needed_cols // 2)
        padding_left = 0 if padding_needed_cols < 0 else int(padding_needed_rows - padding_right)
    elif padding == 'VALID':
        # P = int(np.floor((H - HH) / stride) + 1)
        # Q = int(np.floor((W - WW) / stride) + 1)
        pass
    pad = (padding_left, padding_top, padding_right, padding_bottom)
    input_pad = np.pad(input_data, ((0, 0), (pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), "constant",
                       constant_values=((0, 0), (0, 0), (0, 0), (0, 0)))
    Hp = 1 + int((H + (pad[1] + pad[3]) - HH) / stride)
    Wp = 1 + int((W + (pad[0] + pad[2]) - WW) / stride)

    out = np.zeros((N, Hp, Wp, C))
    for iX in range(N):
        for iC in range(C):
            for iH in range(Hp):
                for iW in range(Wp):
                    out[iX, iH, iW, iC] = np.sum(
                        input_pad[iX, (iH * stride):(iH * stride + HH), (iW * stride):(iW * stride + WW), iC]
                        * depthwise_filter[:, :, iC, 0])

    if bias is not None:
        print("shape of out:{}*************shape of bias:{}".format(out.shape, bias.shape))
        out += bias

    return out.transpose((0, 3, 1, 2))


def quantized_depthwise_conv2d(input_data, depthwise_filter, strides=[1, 1], padding='SAME', bias=None):
    N, C, H, W = input_data.shape
    _, dC, HH, WW = depthwise_filter.shape
    assert C == dC, "input channel and depthwise_filter input channel must be same"
    assert strides[0] == strides[1], "strides must have same size"

    stride = strides[0]
    padding_left = padding_right = padding_top = padding_bottom = 0

    if padding == 'SAME':
        P = int(np.floor((H - 1) / stride + 1))
        Q = int(np.floor((W - 1) / stride + 1))
        padding_needed_rows = (P - 1) * stride + HH - H
        padding_needed_cols = (Q - 1) * stride + WW - W
        padding_top = 0 if padding_needed_rows < 0 else int(padding_needed_rows // 2)
        padding_bottom = 0 if padding_needed_rows < 0 else int(padding_needed_rows - padding_top)
        padding_left = 0 if padding_needed_cols < 0 else int(padding_needed_cols // 2)
        padding_right = 0 if padding_needed_cols < 0 else int(padding_needed_rows - padding_left)
    elif padding == 'VALID':
        # P = int(np.floor((H - HH) / stride) + 1)
        # Q = int(np.floor((W - WW) / stride) + 1)
        pass
    pad = (padding_left, padding_top, padding_right, padding_bottom)
    input_pad = np.pad(input_data, ((0, 0), (pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), "constant",
                   constant_values=((0, 0), (0, 0), (0, 0), (0, 0)))
    Hp = 1 + int((H + (pad[1] + pad[3]) - HH) / stride)
    Wp = 1 + int((W + (pad[0] + pad[2]) - WW) / stride)

    out = np.zeros((N, Hp, Wp, C), dtype=np.int32)
    for iX in range(N):
        for iC in range(C):
            for iH in range(Hp):
                for iW in range(Wp):
                    out[iX, iH, iW, iC] = np.sum(
                        input_pad[iX, (iH * stride):(iH * stride + HH), (iW * stride):(iW * stride + WW), iC].astype(
                            np.uint8)
                        * depthwise_filter[:, :, iC, 0].astype(np.int8))

    if bias is not None:
        out += bias
    return out.transpose((0, 3, 1, 2))


def quantized_depthwise_conv2d_pytorch(input_data, depthwise_filter, strides=[1, 1], padding='SAME', bias=None):
    N, C, H, W = input_data.shape
    _, dC, HH, WW = depthwise_filter.shape
    assert C == dC, "input channel and depthwise_filter input channel must be same"
    assert strides[0] == strides[1], "strides must have same size"

    stride = strides[0]
    padding_left = padding_right = padding_top = padding_bottom = 0

    if padding == 'SAME':
        P = int(np.floor((H - 1) / stride + 1))
        Q = int(np.floor((W - 1) / stride + 1))
        padding_needed_rows = (P - 1) * stride + HH - H
        padding_needed_cols = (Q - 1) * stride + WW - W
        padding_bottom = 0 if padding_needed_rows < 0 else int(padding_needed_rows // 2)
        padding_top = 0 if padding_needed_rows < 0 else int(padding_needed_rows - padding_bottom)
        padding_right = 0 if padding_needed_cols < 0 else int(padding_needed_cols // 2)
        padding_left = 0 if padding_needed_cols < 0 else int(padding_needed_rows - padding_right)
    elif padding == 'VALID':
        # P = int(np.floor((H - HH) / stride) + 1)
        # Q = int(np.floor((W - WW) / stride) + 1)
        pass
    pad = (padding_left, padding_top, padding_right, padding_bottom)
    input_pad = np.pad(input_data, ((0, 0), (pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), "constant",
                   constant_values=((0, 0), (0, 0), (0, 0), (0, 0)))
    Hp = 1 + int((H + (pad[1] + pad[3]) - HH) / stride)
    Wp = 1 + int((W + (pad[0] + pad[2]) - WW) / stride)

    out = np.zeros((N, Hp, Wp, C), dtype=np.int32)
    for iX in range(N):
        for iC in range(C):
            for iH in range(Hp):
                for iW in range(Wp):
                    out[iX, iH, iW, iC] = np.sum(
                        input_pad[iX, (iH * stride):(iH * stride + HH), (iW * stride):(iW * stride + WW), iC].astype(
                            np.uint8)
                        * depthwise_filter[:, :, iC, 0].astype(np.int8))

    if bias is not None:
        out += bias
    return out.transpose((0, 3, 1, 2))


def quantized_separable_conv2d(input_data, depthwise_filter, pointwise_filter, strides=[1, 1], padding='SAME'):
    """
    input: N H W C
    depthwise_filter: HH * WW * C * 1
    pointwise_filter: 1 * 1 * C * F
    """
    N, C, H, W = input_data.shape
    dF, dC, HH, WW = depthwise_filter.shape
    pH, pW, pC, F = pointwise_filter.shape
    assert C == dC == pC, "input channel, depthwise_filter input channel, pointwise_filter input channel must be same"
    assert dF == 1, "depthwise_filter output channel must be one"
    assert pH == pW == 1, "pointwise_filter kernel size must be one"
    assert strides[0] == strides[1], "strides must have same size"
    stride = strides[0]

    padding_left = padding_right = padding_top = padding_bottom = 0

    if padding == 'SAME':
        P = int(np.floor((H - 1) / stride + 1))
        Q = int(np.floor((W - 1) / stride + 1))
        padding_needed_rows = (P - 1) * stride + HH - H
        padding_needed_cols = (Q - 1) * stride + WW - W
        padding_top = 0 if padding_needed_rows < 0 else int(padding_needed_rows // 2)
        padding_bottom = 0 if padding_needed_rows < 0 else int(padding_needed_rows - padding_top)
        padding_left = 0 if padding_needed_cols < 0 else int(padding_needed_cols // 2)
        padding_right = 0 if padding_needed_cols < 0 else int(padding_needed_rows - padding_left)
    elif padding == 'VALID':
        # P = int(np.floor((H - HH) / stride) + 1)
        # Q = int(np.floor((W - WW) / stride) + 1)
        pass
    pad = (padding_left, padding_top, padding_right, padding_bottom)
    input_pad = np.pad(input_data, ((0, 0), (pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), "constant",
                       constant_values=((0, 0), (0, 0), (0, 0), (0, 0)))
    Hp = 1 + int((H + (pad[1] + pad[3]) - HH) / stride)
    Wp = 1 + int((W + (pad[0] + pad[2]) - WW) / stride)

    out = np.zeros((N, Hp, Wp, F))
    for iX in range(N):
        for iF in range(F):
            for iH in range(Hp):
                for iW in range(Wp):
                    out[iX, iH, iW, iF] = np.sum(
                        input_pad[iX, (iH * stride):(iH * stride + HH), (iW * stride):(iW * stride + WW), :].astype(np.uint8)
                        * depthwise_filter[:, :, :, 0].astype(np.int8)
                        * pointwise_filter[0, 0, :, iF]).astype(np.int8)

    return out.transpose((0, 3, 1, 2))


'''
ReLU activation function
return max(0, x)
'''
def relu(input):
    input_ = np.copy(input)
    input_[np.where(input_<0.0)] = 0
    return input_

'''
ReLU6 activation function
return min(max(0, x), 6)
'''
def relu6(input):
    input_ = np.copy(input)
    input_[np.where(input_<0.0)] = 0
    input_[np.where(input_ > 6.0)] = 6.0
    return input_

'''
input_data: [N, units]
softmax function
'''
def softmax(input_data):
    exponential = np.exp(input_data)
    return exponential / np.sum(exponential, axis=1, keepdims=True)

'''
batch flatten for NHWC input_data
return flatten data in [N, H*W*C] shape
Note that, We copy the input_data and reshape the copied data
so that the input keep itself, but slow.
'''
def flatten(input_data):
    N, *_ = input_data.shape
    return input_data.copy().reshape((N, -1))

'''
batch flatten fast
'''
def flatten_fast(input_data):
    return input_data.reshape((input_data.shape[0], -1))


def quantized_conv2d_pytorch(input_data, filter_data, strides=[1,1], padding='SAME', bias=None):
    N, C, H, W = input_data.shape
    K, C_, R, S = filter_data.shape
    assert C==C_, "input's channel and filter's channel must be same."
    u, v = strides # strides in vertical and horizontal direction
    assert u==v, "only support same strides in vertical and horizontal direction"
    padding_left = padding_right = padding_top = padding_bottom = 0
    if padding == 'SAME':
        P = int(np.floor((H-1)/u+1))
        Q = int(np.floor((W-1)/v+1))
        padding_needed_rows = (P-1) * u + R - H
        padding_needed_cols = (Q-1) * v + S - W
        padding_bottom = 0 if padding_needed_rows < 0 else int(padding_needed_rows // 2)
        padding_top = 0 if padding_needed_rows < 0 else int(padding_needed_rows - padding_bottom)
        padding_right = 0 if padding_needed_cols < 0 else int(padding_needed_cols // 2)
        padding_left = 0 if padding_needed_cols < 0 else int(padding_needed_rows - padding_right)
    elif padding == 'VALID':
        P = int(np.floor((H-R)/u) + 1)
        Q = int(np.floor((W-S)/v) + 1)
    pad = (padding_left, padding_top, padding_right, padding_bottom)
    im_matrix = im2col(input_data, filter_data.shape, (P,Q), pad, (u,v), np.uint8) # u8
    filter_matrix = filter2col(filter_data) # s8
    feature_maps_int32 = np.matmul(im_matrix.astype(np.int32), filter_matrix.astype(np.int32), dtype=np.int32)
    feature_maps_int32 = feature_maps_int32.reshape(N, P, Q, K)
    if bias is not None:
        feature_maps_int32 += bias
    return feature_maps_int32.transpose((0, 3, 1, 2))


def quantized_conv2d(input_data, filter_data, strides=[1,1], padding='SAME', bias=None):
    N, C, H, W = input_data.shape
    K, C_, R, S = filter_data.shape
    assert C==C_, "input's channel and filter's channel must be same."
    u, v = strides # strides in vertical and horizontal direction
    assert u==v, "only support same strides in vertical and horizontal direction"
    padding_left = padding_right = padding_top = padding_bottom = 0
    if padding == 'SAME':
        P = int(np.floor((H-1)/u+1))
        Q = int(np.floor((W-1)/v+1))
        padding_needed_rows = (P-1) * u + R - H
        padding_needed_cols = (Q-1) * v + S - W
        padding_top = 0 if padding_needed_rows < 0 else int(padding_needed_rows // 2)
        padding_bottom = 0 if padding_needed_rows < 0 else int(padding_needed_rows - padding_top)
        padding_left = 0 if padding_needed_cols < 0 else int(padding_needed_cols // 2)
        padding_right = 0 if padding_needed_cols < 0 else int(padding_needed_rows - padding_left)
    elif padding == 'VALID':
        P = int(np.floor((H-R)/u) + 1)
        Q = int(np.floor((W-S)/v) + 1)
    pad = (padding_left, padding_top, padding_right, padding_bottom)
    im_matrix = im2col(input_data, filter_data.shape, (P,Q), pad, (u,v), np.uint8) # u8
    filter_matrix = filter2col(filter_data) # s8
    feature_maps_int32 = np.matmul(im_matrix.astype(np.int32), filter_matrix.astype(np.int32), dtype=np.int32)
    feature_maps_int32 = feature_maps_int32.reshape(N, P, Q, K)
    if bias is not None:
        feature_maps_int32 += bias
    return feature_maps_int32.transpose((0, 3, 1, 2)) 


def quantized_conv2d_relu(input_data, filter_data, deq_factor, q_a_factor, strides=[1,1], padding='SAME', bias=None):
    x_s32 = quantized_conv2d(input_data, filter_data, strides, padding, bias)
    x_float64 = q_a_factor * deq_factor * relu(x_s32)
    np.clip(x_float64, 0, 255, out=x_float64)
    x_uint8 = np.uint8(np.rint(x_float64))
    return x_uint8


def quantized_depthwise_conv2d_relu6(input_data, filter_data, deq_factor, q_a_factor, strides=[1, 1],
                                            padding='SAME', bias=None):
    x_s32 = quantized_depthwise_conv2d(input_data, filter_data, strides, padding, bias)
    x_float64 = q_a_factor * relu6(deq_factor * x_s32)
    np.clip(x_float64, 0, 255, out=x_float64)
    x_uint8 = np.uint8(np.rint(x_float64))
    return x_uint8


def quantized_depthwise_conv2d_relu(input_data, filter_data, deq_factor, q_a_factor, strides=[1, 1],
                                            padding='SAME', bias=None):
    x_s32 = quantized_depthwise_conv2d(input_data, filter_data, strides, padding, bias)
    x_float64 = q_a_factor * deq_factor * relu(x_s32)
    np.clip(x_float64, 0, 255, out=x_float64)
    x_uint8 = np.uint8(np.rint(x_float64))
    return x_uint8


# def quantized_depthwise_conv2d_without_relu(input_data, filter_data, deq_factor, q_a_factor, strides=[1, 1],
#                                             padding='SAME', bias=None):
#     x_s32 = quantized_depthwise_conv2d(input_data, filter_data, strides, padding, bias)
#     x_float64 = q_a_factor * deq_factor * x_s32
#     x_float64 += 128
#     np.clip(x_float64, 0, 255, out=x_float64)
#     x_uint8 = np.uint8(np.rint(x_float64))
#     return x_uint8


def quantized_separable_conv2d_relu(input_data, depthwise_filter, pointwise_filter, deq_factor, q_a_factor,
                                    strides=[1, 1], padding='SAME'):
    x_s32 = quantized_separable_conv2d(input_data, depthwise_filter, pointwise_filter, strides, padding)
    x_float64 = q_a_factor * deq_factor * relu(x_s32)
    np.clip(x_float64, 0, 255, out=x_float64)
    x_uint8 = np.uint8(np.rint(x_float64))
    return x_uint8

'''
only integer and bit shift operations in this conv and relu function
'''
def quantized_conv2d_relu_only_integer(input_data, filter_data, deq_factor, q_a_factor, strides=[1,1], padding='SAME', bias=None):
    x_s32 = quantized_conv2d(input_data, filter_data, strides, padding, bias)
    scale = q_a_factor * deq_factor
    num_right_shift, m_int32 = shift_2_POT(scale)
    x_int64 = m_int32.astype(np.int64) * relu(x_s32)
    np.right_shift(x_int64, 31, out=x_int64) # 误差较大，更好的办法是取整前根据小数点后面第一位数进行四舍五入
    np.left_shift(x_int64, num_right_shift, out=x_int64) # 同上
    np.clip(x_int64, 0, 255, out=x_int64)
    x_uint8 = np.uint8(np.rint(x_int64))
    return x_uint8

def quantized_add(input_data_from_close_conv, deq_factor_close, input_data_from_far_conv, deq_factor_far):
    x_from_close = deq_factor_close * input_data_from_close_conv
    x_from_far = deq_factor_far * input_data_from_far_conv
    return x_from_close, x_from_far

def quantized_add_relu(q_a_factor, input_data_from_close_conv, deq_factor_close, input_data_from_far_conv, deq_factor_far):
    x_from_close, x_from_far = quantized_add(input_data_from_close_conv, deq_factor_close, input_data_from_far_conv, deq_factor_far)
    x_clipped = np.clip(q_a_factor*x_from_close + q_a_factor*x_from_far, -128, 127)
    x_int8 = np.rint(x_clipped).astype(np.int8)
    return relu(x_int8).astype(np.uint8)

def quantized_add_relu_deep_1(q_a_factor, input_data_from_conv, deq_factor_from_conv, input_data_from_close_conv, deq_factor_close, input_data_from_far_conv, deq_factor_far):
    x_from_far = quantized_add_relu(q_a_factor, input_data_from_close_conv, deq_factor_close, input_data_from_far_conv, deq_factor_far)
    x_from_close_conv = q_a_factor * deq_factor_from_conv * input_data_from_conv
    x_int16 = np.int16(x_from_far) + np.int16(np.rint(np.clip(x_from_close_conv, -128, 127)))
    return np.uint8(np.clip(x_int16, 0, 127)) # clip as ReLU

def quantized_dense(input_data, weight_data, bias=None):
    x_int32 = np.matmul(input_data.astype(np.int32), weight_data.astype(np.int32))
    if bias is not None:
        x_int32 += bias
    return x_int32

def quantized_dense_relu(input_data, weight_data, deq_factor, q_a_factor, bias=None):
    x_s32 = quantized_dense(input_data, weight_data, bias=bias)
    x_float64 = q_a_factor * deq_factor * relu(x_s32)
    np.clip(x_float64, 0, 255, out=x_float64)
    return np.uint8(np.rint(x_float64))

'''
Normalizes a tensor x by mean and variance, 
and applies (optionally) a scale to it, as well as an offset
'''
def batch_normalization(x, mean, variance, scale, offset):
    assert x.shape[3] == mean.shape[0], 'in_channel != mean length'
    assert x.shape[3] == variance.shape[0], 'in_channel != var length'
    assert x.shape[3] == scale.shape[0], 'in_chaneel != scale length'
    assert x.shape[3] == offset.shape[0], 'in_channel != offset length'
    return scale/(np.sqrt(np.float64(variance))+1e-5) * (x-mean) + offset

def bias_add(input_data, bias):
    num_ch_in = input_data.shape[1] # n c h w format
    num_bias = bias.shape[0]
    assert num_bias == num_ch_in, 'values in bias != input channel'
    return (input_data.transpose((0,2,3,1)) + bias).transpose((0,3,1,2))

if __name__ == '__main__':
    pass