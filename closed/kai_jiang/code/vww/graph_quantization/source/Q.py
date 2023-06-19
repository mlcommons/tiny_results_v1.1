from dataclasses import dataclass
import numpy as np

# '''
# 给出权重的绝对值的最大值 range_w,
# 计算权重的量化因子
# '''
# def weights_quantizing_factor(range_w, format='int8'):
#     assert range_w>0, 'range of weights must be positive'
#     if format=='int8':
#         return 127.0/np.float64(range_w)
#     if format=='int7':
#         return 63.0/np.float64(range_w)
#     else:
#         raise NotImplementedError

'''
dequantizing factor of activation is computed from 
the layer's weights' quantizing factor Qw
and the layer's input quantizing factor Qa
'''
def dequantizing_factor(q_w_factor, q_a_factor):
    return 1.0 / (np.float64(q_w_factor) * np.float64(q_a_factor))


'''
separable_convolution dequantizing factor of activation is computed from 
the layer's depthwise weights' quantizing factor Qw
and the layer's input quantizing factor Qa
and the layer's pointwise weights' quantizing factor Qw2
'''
def separable_conv2d_dequantizing_factor(q_w_factor, q_w_factor2, q_a_factor):
    return 1.0 / (np.float64(q_w_factor) * np.float64(q_w_factor2) * np.float64(q_a_factor))


'''
quantizing weights to int8 format in [-127, 127]
'''
def quantizing_weights(weights_fp32, q_w):
    return np.int8(np.clip(np.rint(q_w * weights_fp32), -127, 127))

def quantizing_activation(activation_fp32, q_a):
    return np.uint8(np.clip(np.rint(q_a * activation_fp32), 0, 255))

def quantizing_bias(bias_fp32, q_w_factor, q_a_factor):
    if isinstance(q_w_factor, list):
        return np.int32(np.clip(np.rint(q_a_factor * np.array(q_w_factor) * bias_fp32), -2**31, 2**31-1))
    else:
        return np.int32(np.clip(np.rint(q_a_factor * q_w_factor * bias_fp32), -2**31, 2**31-1))

def quantizing_data(data, q_w, format):
    if isinstance(q_w, list):
        q_w = np.array(q_w)
        assert q_w.shape[0] == data.shape[0]  # number of q_w should be equal to channel of data
        if format == 'int8': # -> [-128, 127] + 128
            return np.array([np.int8(np.clip(np.rint(w * d), -127, 127)) for w, d in zip(q_w, data)])
        elif format == 'uint8': # -> [0, 255]
            return np.array([np.int8(np.clip(np.rint(w * d), 0, 255)) for w, d in zip(q_w, data)])
        else:
            raise NotImplementedError
    else:
        if format == 'int8': # -> [-128, 127] + 128
            return np.int8(np.clip(np.rint(q_w * data), -127, 127))
        elif format == 'uint8': # -> [0, 255]
            return np.uint8(np.clip(np.rint(q_w * data), 0, 255))
        else:
            raise NotImplementedError


def get_quantizing_factor(range_a, format):

    # assert range_a>0, 'range of data must be positive'
    if isinstance(range_a, list):
        if format == 'int8': # -> [-128, 127] + 128
            return [127.0/np.float64(ra) for ra in range_a]
        elif format == 'uint8': # -> [0, 255]
            return [255.0/np.float64(ra) for ra in range_a]
        elif format == 'int7':
            return [63.0/np.float64(ra) for ra in range_a]
        else:
            raise NotImplementedError
    else:
        if format == 'int8': # -> [-128, 127] + 128
            return 127.0/np.float64(range_a)
        elif format == 'uint8': # -> [0, 255]
            return 255.0/np.float64(range_a)
        elif format == 'int7':
            return 63.0/np.float64(range_a)
        else:
            raise NotImplementedError
    

'''
quantizing activation to uint8 format in [0, 255]
'''
def quantizing_activation2(activation, range_a, q_a, has_negative=False):
    assert range_a>0, 'range of activations must be positive'
    # if activation has negative values, we first relax the (negative) saturation range
    r_left = -128*np.float64(range_a)/127.0 if has_negative else 0.0
    # saturate activation in range [r_left, range_a]
    clipped_activation = np.clip(activation, r_left, range_a)
    # quantizing to integer (round to nearest integer)
    activation_int = np.rint(q_a * clipped_activation)
    # shift (if has negative), clip [0, 255] and convert to uint8
    if has_negative:
        activation_int += 128
    activation_uint8 = np.uint8(np.clip(activation_int, 0, 255))
    return activation_uint8


def fused_quantization(input_int32, quant_factor_input, quant_factor_w, output_quant_factor, activation_func=None):
    """
        a_u8 = || Q_a * D * g(x_32)||
    """
    deq_factor = 1.0/(np.float64(quant_factor_input)*np.float64(quant_factor_w))
    if activation_func:
        # uint8
        activation = output_quant_factor * deq_factor * activation_func(input_int32)
        return np.rint(np.clip(activation, 0, 255)).astype(np.uint8)
    else:
        # int8
        activation = output_quant_factor * deq_factor * input_int32
        return np.rint(np.clip(activation, -127, 127)).astype(np.int8)


def fused_quantization2(input_int32, quant_factor_input, quant_factor_w, output_quant_factor, activation_func=None):
    """
    a_u8 = || Q_a * g(D * x_32)||
    """
    deq_factor = 1.0/(np.float64(quant_factor_input)*np.float64(quant_factor_w))
    if activation_func:
        activation = output_quant_factor * activation_func(input_int32 * deq_factor)
    else:
        activation = output_quant_factor * input_int32 * deq_factor
    return np.rint(np.clip(activation, 0, 255)).astype(np.uint8)


def fused_quantization3(input_int32, quant_factor_input, quant_factor_w, activation_func=None):
    """
    x_f32 = g(D * x_32)
    """
    deq_factor = 1.0/(np.float64(quant_factor_input)*np.float64(quant_factor_w))
    if activation_func:
        activation_s32 = activation_func(input_int32 * deq_factor)
    else:
        activation_s32 = input_int32 * deq_factor
    return activation_s32


def fused_skip_quantization(input1_int32, input2_int32, 
quant_factor_w1, quant_factor_a1, 
quant_factor_w2, quant_factor_a2, 
quant_factor_a3):
    clipped1 = np.clip(quant_factor_a3 /(quant_factor_w1 * quant_factor_a1) * input1_int32, -2**15, 2**15-1)
    clipped2 = np.clip(quant_factor_a3 /(quant_factor_w2 * quant_factor_a2) * input2_int32, -2**15, 2**15-1)
    return np.uint8(np.clip(np.int16(np.rint(clipped1)) + np.int16(np.rint(clipped2)), 0, 127))

def dequantizing_activation(activation_u8, deq_factor_a, has_negative=False):
    activation_float = np.float64(activation_u8)
    if has_negative:
        activation_float = activation_float - 128
    return np.float64(deq_factor_a) * np.float64(activation_float)

def shift_2_POT(factor_fp64):
    assert factor_fp64>0, 'factor must be positive'
    assert factor_fp64==1.0, "factor is 1, not shift" 
    num_right_shift = 0
    if factor_fp64 < 1.0:
        while factor_fp64<0.5:
            factor_fp64 *= 2.0
            num_right_shift -= 1
    elif factor_fp64 > 1.0:
        while factor_fp64 > 1.0:
            factor_fp64 /= 2.0
            num_right_shift += 1
    # now, factor_fp64 is in [0.5, 1.0)
    # convert it to int32 format
    return num_right_shift, np.int32(2**31*factor_fp64)

'''
modify the values of bias in convolution layer,
because we shift input by k(128) before conv.
note that, shape of weights is [filter_h, filter_w, ch_in, ch_out]
'''
def modify_bias(bias, weights, q_factor_a, shift_k=128):
    return bias - shift_k / q_factor_a * np.sum(weights, axis=(0,1,2))


'''
scale weights of convolution layer,
becuase we fuse conv+BN into convolution
'''
def scale_weights_for_BN(weights, gamma, variance, layout='HWCF'):
    if layout == 'HWCF':
        # [K] x [R, S, C, K] or [C] x [R, S, C, 1](depthwise weights)
        # [F] x [H, W, C, F] or [F] x [H, W, C, 1](depthwise weights)
        assert (gamma.shape[0] == weights.shape[3]) or (gamma.shape[0] == weights.shape[2]), "weighs shape wrong"
        
    elif layout == 'FCHW':
        # [F] x [F, C, H, W] or [F] x [1, C, H, W](depthwise weights)
        assert (gamma.shape[0] == weights.shape[0]) or (gamma.shape[0] == weights.shape[1]), "weighs shape wrong"
        weights = np.transpose(weights, [2, 3, 1, 0])
    
    if weights.shape[3] == gamma.shape[0]:
            # conv_2d weight
            out = gamma/np.sqrt(np.float64(variance)) * weights
    else:
        # depthwise_conv2d_weight
        weights = np.squeeze(weights, axis=-1)
        out = gamma/np.sqrt(np.float64(variance)) * weights
        out = np.expand_dims(out, axis=-1)
    
    if layout == 'FCHW':
        out = np.transpose(out, [3, 2, 0, 1])
    return out



def modify_bias_for_BN(bias, gamma, beta, mean, variance):
    # [k] * [K] - [K] * [K] + [1 or K]
    scale = gamma/np.sqrt(np.float64(variance))
    if bias:
        out = scale * bias - scale * mean + beta
    else:
        out = 0 - scale * mean + beta
    return out

'''
modify the values of bias in convolution layer,
because we fuse conv+BN into convolution
'''
