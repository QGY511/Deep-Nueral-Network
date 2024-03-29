import numpy as np
from nndl.layers import *
import pdb

""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""

def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  pad = conv_param['pad']
  stride = conv_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the forward pass of a convolutional neural network.
  #   Store the output as 'out'.
  #   Hint: to pad the array, you can use the function np.pad.
  # ================================================================ #
    
  N, _, H, W = x.shape
  F, _, H_height, W_width = w.shape

  H_out = int(1 + (H + 2 * pad - H_height) / stride)
  W_out = int(1 + (W + 2 * pad - W_width) / stride)
  out = np.zeros((N, F, H_out, W_out))
    
  xpad= np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant')
    
  for n in range(N): 
     for f in range(F):  
        for i in range(0, H_out):
           for j in range(0, W_out):
            vertical_start = i * stride
            vertical_end = vertical_start + H_height
            horizontal_start = j * stride
            horizontal_end = horizontal_start + W_width
            x_slice = xpad[n, :, vertical_start:vertical_end, horizontal_start:horizontal_end]
            out[n, f, i, j] = np.sum(x_slice * w[f, :, :, :]) + b[f]
    
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None

  N, F, out_height, out_width = dout.shape
  x, w, b, conv_param = cache
  
  stride, pad = [conv_param['stride'], conv_param['pad']]
  xpad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')
  num_filts, _, f_height, f_width = w.shape

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the backward pass of a convolutional neural network.
  #   Calculate the gradients: dx, dw, and db.
  # ================================================================ #

  dx = np.zeros_like(x)
  dw = np.zeros_like(w)
  db = np.zeros_like(b)
  dxpad = np.zeros_like(xpad)

  for n in range(N):
    for f in range(F):
       for i in range(out_height):
          for j in range(out_width):
            vert_start = i * stride
            vert_end = vert_start + f_height
            horiz_start = j * stride
            horiz_end = horiz_start + f_width
            dw[f] += xpad[n, :, vert_start:vert_end, horiz_start:horiz_end] * dout[n, f, i, j]
            dxpad[n, :, vert_start:vert_end, horiz_start:horiz_end] += w[f] * dout[n, f, i, j]
    
    if pad > 0:
        dx = dxpad[:, :, pad:-pad, pad:-pad]
    else:
        dx = dxpad

  db = np.sum(dout, axis= (0, 2, 3))

  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  
  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling forward pass.
  # ================================================================ #
  N, C, H, W = x.shape
  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']
  stride = pool_param['stride']

  H_out = (H - pool_height) // stride + 1
  W_out = (W - pool_width) // stride + 1
    
  out = np.zeros((N, C, H_out, W_out))

  for n in range(N):
     for c in range(C):
        for i in range(H_out):
           for j in range(W_out):
              h_1 = i * stride
              h_2 = h_1 + pool_height
              w_1 = j * stride
              w_2 = w_1 + pool_width
              out [n, c, i, j] = np.max(x[n, c, h_1:h_2, w_1:w_2])


  
  cache = (x, pool_param)

  return out, cache

def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  x, pool_param = cache
  pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling backward pass.
  # ================================================================ #
  N, C, H, W = x.shape
  H_out = (H - pool_height) // stride + 1
  W_out = (W - pool_width) // stride + 1
  dx = np.zeros_like(x)

  for n in range(N):
     for c in range(C):
        for i in range(H_out):
           for j in range(W_out):
            h_1 = i * stride
            h_2 = h_1 + pool_height
            w_1 = j * stride
            w_2 = w_1 + pool_width

            mask = (x[n, c, h_1:h_2, w_1:w_2] == np.max(x[n, c, h_1:h_2, w_1:w_2]))

            dx[n, c, h_1:h_2, w_1:w_2] =  dx[n, c, h_1:h_2, w_1:w_2]  + dout [n, c, i, j] * mask

  return dx

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """

  out, cache = None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm forward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  
  N, C, H, W = x.shape
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  running_mean = bn_param.get('running_mean', np.zeros(C, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(C, dtype=x.dtype))

  if mode == 'train':

    x_reshaped = x.transpose(0, 2, 3, 1).reshape(-1, C)
        
    minibatch_mean = np.mean(x_reshaped, axis=0)
    minibatch_var = np.var(x_reshaped, axis=0)
        
    minibatch_normalized = (x_reshaped - minibatch_mean) / np.sqrt(minibatch_var + eps)
        
    out = gamma * minibatch_normalized + beta
    out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        
    running_mean = momentum * running_mean + (1 - momentum) * minibatch_mean
    running_var = momentum * running_var + (1 - momentum) * minibatch_var
        
    cache = (x, minibatch_normalized, minibatch_mean, minibatch_var, gamma, eps)

  elif mode == 'test':
    x_reshaped = x.transpose(0, 2, 3, 1).reshape(-1, C)

    minibatch_normalized = (x_reshaped - running_mean) / np.sqrt(running_var + eps)

    out = gamma * minibatch_normalized + beta

    out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)

  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)


  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm backward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  
  x, minibatch_normalized, minibatch_mean, minibatch_var, gamma, eps = cache
  N, C, H, W = dout.shape
    

  dout_reshaped = dout.transpose(0, 2, 3, 1).reshape(-1, C) 
  x_reshaped = x.transpose(0, 2, 3, 1).reshape(-1, C) 

  dbeta = np.sum(dout, axis=(0, 2, 3))
  dgamma = np.sum(dout_reshaped * minibatch_normalized.reshape(-1, C), axis=0)
    
  numerator = x_reshaped - minibatch_mean
  denominator = minibatch_var + eps

  dx_normalized = dout_reshaped * gamma
  d_1 = dx_normalized / np.sqrt(denominator)
    
  dx_var = np.sum(dx_normalized * numerator * -0.5 * np.power(denominator, -1.5), axis=0)
  d_2 = dx_var * 2.0 * numerator / (N * H * W)
    
  dx_mean = np.sum(dx_normalized * -1.0 / np.sqrt(denominator), axis=0) + dx_var * np.mean(-2.0 * numerator, axis=0)
  d_3 = dx_mean /  (N * H * W)
    
  dx = d_1 + d_2 + d_3
  dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2) 

  return dx, dgamma, dbeta