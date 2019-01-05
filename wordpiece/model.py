from __future__ import print_function
from __future__ import division


import tensorflow as tf


class adict(dict):
    ''' Attribute dictionary - a convenience data structure, similar to SimpleNamespace in python 3.3
        One can use attributes to read/write dictionary content.
    '''
    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self

def normalize(inputs, 
              epsilon = 1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.
    
    Args:
      inputs: 2-dim以上的tensor类型，要求第一个维度是Batch.
      epsilon: 很小的浮点数. 用来避免 ZeroDivision Error.
      scope:  `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
      
    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
    
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta= tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta
        
    return outputs


def multihead_attention(queries, 
                        keys, 
                        num_units=None, 
                        num_heads=8, 
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention", 
                        reuse=None):
    
    '''Applies multihead attention.
    
    Args:
      queries: 3d tensor [Batch, query_time, query_dim].
      keys:    3d tensor [Batch, key_time,   key_dim].
      num_units: 标量 hidden_size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. 是否mask后续时间步. 
      num_heads: 多头attention头数.
      scope: `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns
      A 3d tensor with shape of (Batch,query_time,dim)  
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # 如果不设置 num_unit 进行等维映射
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]
        
        # QKV的映射
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu) # [Batch, query_time, query_dim]
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)    # [Batch, key_time,   key_dim]
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)    # [Batch, value_time, value_dim]
        
        # 多头attention的数据准备 split & concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # [Batch*head, query_time, query_dim/head]
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # [Batch*head, key_time,   key_dim/head]
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # [Batch*head, value_time, value_dim/head]

        # 相似度矩阵  QK^T
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))   # [Batch*head, query_time, key_dim/head]
        
        # Scale-dot QK^T/sqrt(K_dim)
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5) # [Batch*head, query_time, key_dim/head]
        
        # Key Masking
        '''
        key_dim : [Batch, key_time, key_dim]
        tf.reduce_sum--->padding=0 计算为0
        abs->sign 后ready mask的位置
        '''
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1))) # (Batch,      key_dim)
        key_masks = tf.tile(key_masks, [num_heads, 1])            # (Batch*head, key_dim)
        
        '''
        (Batch*head, key_dim)-->(Batch*head,1,key_dim)-->(Batch*head,query_time,key_dim)
        '''
        
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)

        # 构建了个和output同样size的padding矩阵 初始化为很小值
        paddings = tf.ones_like(outputs)*(-2**32+1)
        #tf.equal--->判断当前位置是否是<pad>如果是的话paddding替换
        outputs  = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)
  
        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :]) # (T_q, T_k)
            tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense() # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]) # (h*N, T_q, T_k)
   
            paddings = tf.ones_like(masks)*(-2**32+1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs) # (h*N, T_q, T_k)
  
        # Activation
        outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)
         
        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1))) # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (h*N, T_q, T_k)
        outputs *= query_masks # broadcasting. (N, T_q, C)
          
        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
               
        # Weighted sum
        outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C/h)
        
        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, C)
              
        # Residual connection
        outputs += queries
              
        # Normalize
        outputs = normalize(outputs) # (N, T_q, C)
 
    return outputs

def conv2d(input_, output_dim, k_h, k_w, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim])
        b = tf.get_variable('b', [output_dim])

    return tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], padding='VALID') + b


def linear(input_, output_size, scope=None):
    '''
    Linear map: output[k] = sum_i(Matrix[k, i] * args[i] ) + Bias[k]

    Args:
        args: a tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    scope: VariableScope for the created subgraph; defaults to "Linear".
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  '''

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "SimpleLinear"):
        matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
        bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)

    return tf.matmul(input_, tf.transpose(matrix)) + bias_term


def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).

    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """

    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = f(linear(input_, size, scope='highway_lin_%d' % idx))

            t = tf.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias)

            output = t * g + (1. - t) * input_
            input_ = output

    return output


def tdnn(input_, kernels, kernel_features, scope='TDNN'):
    '''

    :input:           input float tensor of shape [(batch_size*num_unroll_steps) x max_word_length x embed_size]
    :kernels:         array of kernel sizes
    :kernel_features: array of kernel feature sizes (parallel to kernels)
    '''
    assert len(kernels) == len(kernel_features), 'Kernel and Features must have the same size'

    max_word_length = input_.get_shape()[1]
    embed_size = input_.get_shape()[-1]	
    # input_: [batch_size*num_unroll_steps, 1, max_word_length, embed_size]
    input_ = tf.expand_dims(input_, 1)

    layers = []
    with tf.variable_scope(scope):
        for kernel_size, kernel_feature_size in zip(kernels, kernel_features):
            reduced_length = max_word_length - kernel_size + 1
            # [batch_size x max_word_length x embed_size x kernel_feature_size]
            conv = conv2d(input_, kernel_feature_size, 1, kernel_size, name="kernel_%d" % kernel_size)

            # [batch_size x 1 x 1 x kernel_feature_size]
            pool = tf.nn.max_pool(tf.tanh(conv), [1, 1, reduced_length, 1], [1, 1, 1, 1], 'VALID')

            layers.append(tf.squeeze(pool, [1, 2]))

        if len(kernels) > 1:
            output = tf.concat(layers, 1)
        else:
            output = layers[0]

    return output


def inference_graph(char_vocab_size, word_vocab_size,
                    char_embed_size=15,
                    batch_size=20,
                    num_highway_layers=2,
                    num_rnn_layers=2,
                    rnn_size=650,
                    max_word_length=32,
                    kernels         = [ 1,   2,   3,   4,   5,   6,   7],
                    kernel_features = [50, 100, 150, 200, 200, 200, 200],
                    num_unroll_steps=35,
                    dropout=0.0):

    assert len(kernels) == len(kernel_features), 'Kernel and Features must have the same size'

    input_ = tf.placeholder(tf.int32, shape=[batch_size, num_unroll_steps, max_word_length], name="input")

    '''Embedding层'''
    with tf.variable_scope('Embedding'):
        char_embedding = tf.get_variable('char_embedding', [char_vocab_size, char_embed_size])
        '''Index 0 位置通常是<PAD> char_embedding设置为[0.0 ....]'''
        clear_char_embedding_padding = tf.scatter_update(char_embedding, [0], tf.constant(0.0, shape=[1, char_embed_size]))

        # [batch_size x max_word_length, num_unroll_steps, char_embed_size]
        input_embedded = tf.nn.embedding_lookup(char_embedding, input_)
        input_embedded = tf.reshape(input_embedded, [-1, max_word_length, char_embed_size])
        print('>>> input_embeded:',input_embedded)
        
    with tf.variable_scope('Attention'):
        
        queries,keys = input_embedded,input_embedded
        
        input_embedded = multihead_attention(queries, 
                                             keys, 
                                             num_units=16, 
                                             num_heads=1, 
                                             dropout_rate=0,
                                             is_training=True,
                                             causality=False,
                                             scope="multihead_attention", 
                                             reuse=None)      

    '''Conv层'''
    # [batch_size x num_unroll_steps, cnn_size]  # where cnn_size=sum(kernel_features)
    input_cnn = tdnn(input_embedded, kernels, kernel_features)

    '''HighwayNetwork '''
    if num_highway_layers > 0:
        input_cnn = highway(input_cnn, input_cnn.get_shape()[-1], num_layers=num_highway_layers)

    '''LSTM '''
    with tf.variable_scope('LSTM'):
        ##################################
        def create_rnn_cell():
            cell = tf.contrib.rnn.BasicLSTMCell(rnn_size, state_is_tuple=True, forget_bias=0.0, reuse=False)
            if dropout > 0.0:
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.-dropout)
            return cell
        
        if num_rnn_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([create_rnn_cell() for _ in range(num_rnn_layers)], state_is_tuple=True)
        else:
            cell = create_rnn_cell()
        #################################
        '''设置初始化'''
        initial_rnn_state = cell.zero_state(batch_size, dtype=tf.float32)

        input_cnn  = tf.reshape(input_cnn, [batch_size, num_unroll_steps, -1])
        input_cnn2 = [tf.squeeze(x, [1]) for x in tf.split(input_cnn, num_unroll_steps, 1)]

        outputs, final_rnn_state = tf.contrib.rnn.static_rnn(cell, input_cnn2,
                                         initial_state=initial_rnn_state, dtype=tf.float32)

        # linear projection onto output (word) vocab
        logits = []
        with tf.variable_scope('WordEmbedding') as scope:
            for idx, output in enumerate(outputs):
                if idx > 0:
                    scope.reuse_variables()
                logits.append(linear(output, word_vocab_size))

    return adict(
        input = input_,
        clear_char_embedding_padding=clear_char_embedding_padding,
        input_embedded=input_embedded,
        input_cnn=input_cnn,
        initial_rnn_state=initial_rnn_state,
        final_rnn_state=final_rnn_state,
        rnn_outputs=outputs,
        logits = logits
    )


def loss_graph(logits, batch_size, num_unroll_steps):

    with tf.variable_scope('Loss'):
        targets = tf.placeholder(tf.int64, [batch_size, num_unroll_steps], name='targets')
        target_list = [tf.squeeze(x, [1]) for x in tf.split(targets, num_unroll_steps, 1)]

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = target_list), name='loss')

    return adict(
        targets=targets,
        loss=loss
    )


def training_graph(loss, learning_rate=1.0, max_grad_norm=5.0):
    ''' Builds training graph. '''
    global_step = tf.Variable(0, name='global_step', trainable=False)

    with tf.variable_scope('SGD_Training'):
        # SGD learning parameter
        learning_rate = tf.Variable(learning_rate, trainable=False, name='learning_rate')

        # collect all trainable variables
        tvars = tf.trainable_variables()
        grads, global_norm = tf.clip_by_global_norm(tf.gradients(loss, tvars), max_grad_norm)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)

    return adict(
        learning_rate=learning_rate,
        global_step=global_step,
        global_norm=global_norm,
        train_op=train_op)


def model_size():

    params = tf.trainable_variables()
    size = 0
    for x in params:
        sz = 1
        for dim in x.get_shape():
            sz *= dim.value
        size += sz
    return size


if __name__ == '__main__':

    with tf.Session() as sess:

        with tf.variable_scope('Model'):
            graph = inference_graph(char_vocab_size=51, word_vocab_size=10000, dropout=0.5)
            graph.update(loss_graph(graph.logits, batch_size=20, num_unroll_steps=35))
            graph.update(training_graph(graph.loss, learning_rate=1.0, max_grad_norm=5.0))

        with tf.variable_scope('Model', reuse=True):
            inference_graph = inference_graph(char_vocab_size=51, word_vocab_size=10000)
            inference_graph.update(loss_graph(graph.logits, batch_size=20, num_unroll_steps=35))

        print('Model size is:', model_size())

        # need a fake variable to write scalar summary
        tf.scalar_summary('fake', 0)
        summary = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter('./tmp', graph=sess.graph)
        writer.add_summary(sess.run(summary))
        writer.flush()
