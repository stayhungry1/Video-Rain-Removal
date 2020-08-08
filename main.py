import tensorflow as tf
import tflearn
from cell import ConvLSTMCell

num_epoch = 30
timestep = 9
num_input_channels = 1 #Y channel
batch_size = 16
filter_num = 24
conv_filter_num = 64
kernel = [3,3]
width = 64
height = 64
patch_size = 64
WEIGHT_DECAY = 0
IS_BATCH_NORM = False

def weight_variable(shape, name=None, is_reuse_var=False):
    initial = tf.truncated_normal(shape, stddev=0.1)
    weight = tf.Variable(initial, name=name)
    weight_decay = tf.multiply(tf.nn.l2_loss(weight), WEIGHT_DECAY)
    tf.add_to_collection('weight_dec', weight_decay)
    return weight

def bias_variable(shape, name=None, is_reuse_var=False):
    initial = tf.constant(0.1)
    bias = tf.Variable(initial, name=name)
    return bias

def activate(x, acti_mode, scope=None):
    if acti_mode==0:
        return x
    elif acti_mode==1:
        return tf.nn.relu(x)
    elif acti_mode==2:
        return tf.nn.sigmoid(x)
    elif acti_mode==3:
        return (tf.nn.tanh(x) + x) / 2
    elif acti_mode==4:
        return (tf.nn.sigmoid(x) + x) / 2
    elif acti_mode==5:
        return tf.nn.leaky_relu(x)
    elif acti_mode==6:
        return tflearn.prelu(x)

def overlap_conv(x, k_width, stride, n_channel_out, acti_mode, is_batch_norm = False):
    n_channel_x = int(x.get_shape()[3])
    w_conv = weight_variable([k_width, k_width, n_channel_x, n_channel_out])
    b_conv = bias_variable([n_channel_out])
    h_conv = tf.nn.conv2d(x, w_conv, strides = [1, stride, stride, 1], padding = 'SAME') + b_conv
    if is_batch_norm:
        h_conv = tf.layers.batch_normalization(h_conv)
    h_conv = activate(h_conv, acti_mode)
    print(h_conv)
    return(h_conv)

def overlap_conv_with_shortcut(x, x_shortcut, k_width, stride, n_channel_out, acti_mode, is_batch_norm = False, add_mode = 'sum'):
    if type(x_shortcut) == list:
        if add_mode == 'sum':
            x_shortcut_tensor = tf.add_n([x_shortcut])
        elif add_mode == 'concat':
            x_shortcut_tensor = tf.concat(x_shortcut, axis = 3)
    else:
        x_shortcut_tensor = x_shortcut
    x_input = x
    if add_mode == 'concat':
        x_input = tf.concat([x_input, x_shortcut_tensor], axis = 3)
    n_channel_x = int(x_input.get_shape()[3])
    with tf.variable_scope('conv_main'):
        w_conv = weight_variable([k_width, k_width, n_channel_x, n_channel_out])
        b_conv = bias_variable([n_channel_out])
    h_conv = tf.nn.conv2d(x_input, w_conv, strides = [1, stride, stride, 1], padding = 'SAME') + b_conv

    if add_mode == 'sum':
        n_channel_shortcut = int(x_shortcut_tensor.get_shape()[3])
        if n_channel_shortcut == n_channel_out:
            h_conv += x_shortcut_tensor
        else:
            with tf.variable_scope('conv_shortcut'):
                w_shortcut = weight_variable([1, 1, n_channel_shortcut, n_channel_out])
                b_shortcut = bias_variable([n_channel_out])
                h_shortcut = tf.nn.conv2d(x_shortcut_tensor, w_shortcut, strides = [1, 1, 1, 1], padding = 'SAME') + b_shortcut
                h_conv += h_shortcut

    if is_batch_norm:
        h_conv = tf.layers.batch_normalization(h_conv)
    h_conv = activate(h_conv, acti_mode)
    print(h_conv)
    return(h_conv)

def dense_unit_4_layer(x, is_training, is_final = False):
    n_channel_out = 12
    h_conv_1 = overlap_conv(x, k_width = 3, stride = 1, n_channel_out = n_channel_out, acti_mode = 6, is_batch_norm = IS_BATCH_NORM)
    h_conv_2 = overlap_conv_with_shortcut(x, h_conv_1, k_width = 3, stride = 1,
                                          n_channel_out = n_channel_out, acti_mode = 6, is_batch_norm = IS_BATCH_NORM, add_mode = 'concat')
    h_conv_3 = overlap_conv_with_shortcut(x, [h_conv_1, h_conv_2], k_width = 3, stride = 1,
                                          n_channel_out = n_channel_out, acti_mode = 6, is_batch_norm = IS_BATCH_NORM, add_mode = 'concat')
    if is_final == True:
        h_conv_4 = overlap_conv_with_shortcut(x, [h_conv_1, h_conv_2, h_conv_3], k_width = 3, stride = 1,
                                          n_channel_out = 1, acti_mode = 0, is_batch_norm = IS_BATCH_NORM, add_mode = 'concat')
    else:
        h_conv_4 = overlap_conv_with_shortcut(x, [h_conv_1, h_conv_2, h_conv_3], k_width = 3, stride = 1,
                                          n_channel_out = n_channel_out, acti_mode = 6, is_batch_norm = IS_BATCH_NORM, add_mode = 'concat')
    return h_conv_4

def shared_densenet(x, height, width, is_training, reuse = False, scope='shared_densenet'):
    with tf.variable_scope(scope, reuse=reuse):
        print("shape: %s %s" %(height, width))
        x1 = tf.reshape(x, [-1, height, width, 1])
        h_conv_pre = dense_unit_4_layer(x1, is_training)
        h_conv_mid = dense_unit_4_layer(h_conv_pre, is_training)
        return h_conv_mid

def biconvlstm_R(x, height, width, filter_num, conv_filter_num, kernel, batch_size, timestep, istraining, reuse=False, scope='R_LSTM_net'):
    with tf.variable_scope(scope, reuse=reuse):
        print("shape: %s %s" % (height, width))
        h_conv_post_1 = dense_unit_4_layer(x, is_training)
        h_conv_post_1 = tf.reshape(h_conv_post_1, [-1, timestep, height, width, 12])

        convlstm_layer_qx = ConvLSTMCell(shape=[height, width], filters=filter_num, kernel=kernel)
        convlstm_layer_hx = ConvLSTMCell(shape=[height, width], filters=filter_num, kernel=kernel)
        outputs, state = tf.nn.bidirectional_dynamic_rnn(convlstm_layer_qx, convlstm_layer_hx, h_conv_post_1, dtype=h_conv_post_1.dtype, scope = 'R_bi_dynamic_rnn')

        outputs = tf.concat(outputs, 4)

        convlstm_layer_qx_1 = ConvLSTMCell(shape=[height, width], filters=filter_num, kernel=kernel)
        convlstm_layer_hx_1 = ConvLSTMCell(shape=[height, width], filters=filter_num, kernel=kernel)
        outputs, state = tf.nn.bidirectional_dynamic_rnn(convlstm_layer_qx_1, convlstm_layer_hx_1, outputs, dtype=outputs.dtype, scope = 'R_bi_dynamic_rnn_1')
        outputs = tf.concat(outputs, 4)

        outputs = tf.reshape(outputs[:, (timestep - 1) // 2, :, :, :], [-1, height, width, 2 * filter_num])

        c11_w = tf.get_variable("R_c11_w", shape=[3, 3, 2 * filter_num, conv_filter_num],
                                initializer=tf.contrib.layers.xavier_initializer(uniform=True))
        c11_b = tf.get_variable("R_c11_b", shape=[conv_filter_num],
                                initializer=tf.constant_initializer(0.0))

        c12_w = tf.get_variable("R_c12_w", shape=[3, 3, conv_filter_num, (conv_filter_num // 2)],
                                initializer=tf.contrib.layers.xavier_initializer(uniform=True))
        c12_b = tf.get_variable("R_c12_b", shape=[(conv_filter_num // 2)],
                                initializer=tf.constant_initializer(0.0))

        c13_w = tf.get_variable("R_c13_w", shape=[3, 3, (conv_filter_num // 2), 1],
                                initializer=tf.contrib.layers.xavier_initializer(uniform=True))
        c13_b = tf.get_variable("R_c13_b", shape=[1],
                                initializer=tf.constant_initializer(0.0))

        c11 = tf.nn.conv2d(outputs, c11_w, strides=[1, 1, 1, 1], padding='SAME')
        c11 = tf.nn.bias_add(c11, c11_b)
        c11 = tflearn.activations.prelu(c11)

        c12 = tf.nn.conv2d(c11, c12_w, strides=[1, 1, 1, 1], padding='SAME')
        c12 = tf.nn.bias_add(c12, c12_b)
        c12 = tflearn.activations.prelu(c12)

        c13 = tf.nn.conv2d(c12, c13_w, strides=[1, 1, 1, 1], padding='SAME')
        c13 = tf.nn.bias_add(c13, c13_b)

        final_out = c13
        return outputs, final_out

def biconvlstm_B(x, R_lstmout, height, width, filter_num, conv_filter_num, kernel, batch_size, timestep, istraining, reuse = False, scope='B_LSTM_net'):
    with tf.variable_scope(scope, reuse=reuse):
        print("shape: %s %s" %(height, width))
        h_conv_post_1 = dense_unit_4_layer(x, is_training)
        h_conv_post_1 = tf.reshape(h_conv_post_1, [-1, timestep, height, width, 12])

        convlstm_layer_qx = ConvLSTMCell(shape=[height, width], filters=filter_num, kernel=kernel)
        convlstm_layer_hx = ConvLSTMCell(shape=[height, width], filters=filter_num, kernel=kernel)
        outputs, state = tf.nn.bidirectional_dynamic_rnn(convlstm_layer_qx, convlstm_layer_hx, h_conv_post_1, dtype=h_conv_post_1.dtype, scope = 'B_bi_dynamic_rnn')

        outputs = tf.concat(outputs, 4)

        convlstm_layer_qx_1 = ConvLSTMCell(shape=[height, width], filters=filter_num, kernel=kernel)
        convlstm_layer_hx_1 = ConvLSTMCell(shape=[height, width], filters=filter_num, kernel=kernel)
        outputs, state = tf.nn.bidirectional_dynamic_rnn(convlstm_layer_qx_1, convlstm_layer_hx_1, outputs, dtype=outputs.dtype, scope = 'B_bi_dynamic_rnn_1')
        outputs = tf.concat(outputs, 4)
        outputs = tf.reshape(outputs[:, (timestep - 1) // 2, :, :, :], [-1, height, width, 2 * filter_num])

        r = bias_variable([1])   #weight parameter: r
        outputs = outputs * (R_lstmout * (1 - r) + 1 * r)
        c11_w = tf.get_variable("B_c11_w", shape=[3, 3, 2 * filter_num, conv_filter_num],
                               initializer=tf.contrib.layers.xavier_initializer(uniform=True))
        c11_b = tf.get_variable("B_c11_b", shape=[conv_filter_num],
                               initializer=tf.constant_initializer(0.0))

        c12_w = tf.get_variable("B_c12_w", shape=[3, 3, conv_filter_num, (conv_filter_num // 2)],
                                initializer=tf.contrib.layers.xavier_initializer(uniform=True))
        c12_b = tf.get_variable("B_c12_b", shape=[(conv_filter_num // 2)],
                                initializer=tf.constant_initializer(0.0))

        c13_w = tf.get_variable("B_c13_w", shape=[3, 3, (conv_filter_num // 2), 1],
                                initializer=tf.contrib.layers.xavier_initializer(uniform=True))
        c13_b = tf.get_variable("B_c13_b", shape=[1],
                                initializer=tf.constant_initializer(0.0))

        c11 = tf.nn.conv2d(outputs, c11_w, strides=[1, 1, 1, 1], padding='SAME')
        c11 = tf.nn.bias_add(c11, c11_b)
        c11 = tflearn.activations.prelu(c11)

        c12 = tf.nn.conv2d(c11, c12_w, strides=[1, 1, 1, 1], padding='SAME')
        c12 = tf.nn.bias_add(c12, c12_b)
        c12 = tflearn.activations.prelu(c12)

        c13 = tf.nn.conv2d(c12, c13_w, strides=[1, 1, 1, 1], padding='SAME')
        c13 = tf.nn.bias_add(c13, c13_b)

        final_out = c13

        return final_out, r

def net(x, height, width, filter_num, conv_filter_num, kernel, batch_size, timestep, istraining, reuse = False):
    h_conv_mid = shared_densenet(x, height, width, is_training)
    R_lstmout, R_outputs = biconvlstm_R(h_conv_mid, height, width, filter_num, conv_filter_num, kernel, batch_size, timestep, istraining)
    B_outputs, r = biconvlstm_B(h_conv_mid, R_lstmout, height, width, filter_num, conv_filter_num, kernel, batch_size, timestep, istraining)
    return R_outputs, B_outputs, r

if __name__ == '__main__':

    videos = tf.placeholder(tf.float32, shape=(None, timestep, patch_size, patch_size, num_input_channels), name='rainy_videos')
    labels = tf.placeholder(tf.float32, shape=(None, patch_size, patch_size, num_input_channels), name='ground_truth')
    labels_rain = tf.placeholder(tf.float32, shape=(None, patch_size, patch_size, num_input_channels), name='ground_truth')
    is_training = tf.placeholder(tf.bool, name='is_training')

    #network
    R_outputs, B_outputs, r = net(videos, height, width, filter_num, conv_filter_num, kernel, batch_size, timestep, is_training) #R_outputs: rain_GT, B_outputs: background_GT

    #loss definition
    B_mse_loss = tf.reduce_mean(tf.pow(tf.subtract(labels, B_outputs), 2.0))
    R_mse_loss = tf.reduce_mean(tf.pow(tf.subtract(labels_rain, R_outputs), 2.0))
    total_loss_1 = R_mse_loss + 0.01 * B_mse_loss
    total_loss_2 = 0.01 * R_mse_loss + B_mse_loss

    for i_epoch in num_epoch:
        if i_epoch < 10: #set as 5,10,etc. Training stage1: use loss1,training stage2: use loss2
            feed data
            use total_loss_1
        if i_epoch >= 10:
            feed data
            use total_loss_2