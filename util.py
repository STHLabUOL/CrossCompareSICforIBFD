import numpy as np
import scipy.io as sio
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

def load_signal_data(path, noSignals=10, norm_x='var', norm_y='var', seq_len=2048, filter_len=32, trim_y=True, with_noise=False):


    matlabVariables = sio.loadmat(path+'/fileID0.mat')

    isHammersteinData = 'sig_s' in matlabVariables

    if isHammersteinData:
        signalLen = len(matlabVariables['sig_s'])
    else:
        signalLen = len(matlabVariables['sig_z'])
    signalLen = seq_len #NK: makes sense to set this for now
    x_train = np.zeros(shape=(noSignals,signalLen))+1j*np.zeros(shape=(noSignals,signalLen))
    y_train = np.zeros(shape=(noSignals,signalLen))+1j*np.zeros(shape=(noSignals,signalLen))
    noise = np.zeros(shape=(noSignals,signalLen))+1j*np.zeros(shape=(noSignals,signalLen))
    for p in range(noSignals):  
        matlabVariables = sio.loadmat(path+'/fileID'+str(p)+'.mat')
        if isHammersteinData:
            x_train[p,:] = tf.squeeze(matlabVariables['sig_s'])[:seq_len]
            y_train[p,:] = tf.squeeze(matlabVariables['sig_yH'])[:seq_len]
            if 'sig_x' in matlabVariables:
                noise[p,:] = tf.squeeze(matlabVariables['sig_yH'])[:seq_len] - tf.squeeze(matlabVariables['sig_x'])[:seq_len]
        else:
            x_train[p,:] = tf.squeeze(matlabVariables['sig_z'])[:seq_len]
            y_train[p,:] = tf.squeeze(matlabVariables['sig_yW'])[:seq_len]
            if 'sig_x' in matlabVariables:
                noise[p,:] = tf.squeeze(matlabVariables['sig_yH'])[:seq_len] - tf.squeeze(matlabVariables['sig_x'])[:seq_len]

    sigma_x, sigma_y = np.sqrt(np.var(x_train)), np.sqrt(np.var(y_train)) 
    max_x, max_y = np.max(np.abs(x_train)), np.max(np.abs(y_train))


    # TODO: this is so unelegant - please improve:

    if norm_x != 'none':
        if norm_x == 'var':
            x_train = x_train/sigma_x
        elif norm_x == 'max':
            x_train = x_train/max_x
        else:
            raise ValueError('norm_x must be "var" or "max"')
    
    if norm_y != 'none':
        if norm_y == 'var':
            y_train = y_train/sigma_y
        elif norm_y == 'max':
            y_train = y_train/max_y
        else:
            raise ValueError('norm_y must be "var" or "max"')


    x_train = np.expand_dims(x_train, axis=2) # 1 signal, 1-channel
    y_train = np.expand_dims(y_train, axis=2)


    #filterLen = 256  # 150 scheint auch fast zu reichen; 350 bringt nichts dazu für die Abraham Daten
    sub_sequence_shift = seq_len-filter_len+1
    #filter_len = seq_len-sub_sequence_shift+1 # trailing zeros auf GRU Einschwingphase vorstellen
    sub_sequences_no = signalLen//sub_sequence_shift   # etwas mehr als tatsächlich -> nur 1 Optimizer step


    x_train_batched = tf.signal.frame(x_train, seq_len, sub_sequence_shift, axis=1) 
    y_train_batched = tf.signal.frame(y_train, seq_len, sub_sequence_shift, axis=1) 
    if trim_y:
        y_train_batched = y_train_batched[:,:,filter_len-1:]


    x_train = np.expand_dims(x_train, axis=1) # 1 signal, 1-channel
    y_train = np.expand_dims(y_train, axis=1)

    if with_noise:
        return x_train, x_train_batched, y_train, y_train_batched, noise
    else:
        return x_train, x_train_batched, y_train, y_train_batched
    


def SIestimationLinear(x, y, chanLen):
    # Slightly modified compared to fd.SIestimationLinear to account for already trimmed y-signal.
	# Construct LS problem
	A = np.reshape([np.flip(x[i+1:i+chanLen+1],axis=0) for i in range(x.size-chanLen)], (x.size-chanLen, chanLen))
	# Solve LS problem
	h = np.linalg.lstsq(A, y[1:])[0]
	# Output estimated channels
	return h


def compute_GMACS(model, tensorSpec_shape, tensorSpec_dtype):

    @tf.function
    def model_fn(x):
        return model(x)

    concrete_fn = model_fn.get_concrete_function(tf.TensorSpec(tensorSpec_shape, tensorSpec_dtype))

    # Convert to a graph
    frozen_func = convert_variables_to_constants_v2(concrete_fn)
    graph_def = frozen_func.graph.as_graph_def()

    # Use the profiler
    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

    flops = tf.compat.v1.profiler.profile(graph=frozen_func.graph,
                                        run_meta=run_meta,
                                        cmd='op',
                                        options=opts)

    # GMACs = MACs / 2 / 1e9
    if flops is not None:
        total_flops = flops.total_float_ops
        gmacs = total_flops / 2 / 1e9
        return gmacs
    else:
        raise ValueError('Computation failed.')