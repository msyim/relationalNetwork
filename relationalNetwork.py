import tensorflow as tf
import itertools
import math
import matplotlib.image as mpimg
from scipy.misc import imresize
import numpy as np
import re
from questionReader import questionReader

# meta
# IMGDIR should be the directory where the input images are.
#   ex) ../../CLEVR_v1.0/images/train/
# TRAINQUESTIONS should be the json file containing the question, answer, obj info, etc.
#   ex) ../../CLEVR_v1.0/questions/CLEVR_train_questions.json
IMGDIR=
TRAINQUESTIONS=
EPOCHS = 20

# X : image(png) input, Y : output of RN, phase : indicates whether this is a training session or not.
X = tf.placeholder(tf.float32, shape=[None, 128, 128, 4])
Y = tf.placeholder(tf.float32, shape=[None, 29])
# batch_norm phase : train or not
phase = tf.placeholder(tf.bool)

# question_X is of the shape : [SentenceLength, wordIndex]
max_seq_length = 50
question_X = tf.placeholder(tf.float32, shape=[None, max_seq_length, 87])
sentenceLen = tf.placeholder(tf.int32)
nHidden = 128

# dropout rate
keep_prob = tf.placeholder(tf.float32)

# weight initializer helper
def weight_variable(name, shape, init = tf.contrib.layers.variance_scaling_initializer() ):
    return tf.get_variable(name, shape = shape, initializer = init)

# weights
weights={
    
    # CNN weights
    "CNNW1" : weight_variable("CNNW1", [3,3,4,24]),
    "CNNb1" : weight_variable("CNNb1", [24]),
    "CNNW2" : weight_variable("CNNW2", [3,3,24,24]),
    "CNNb2" : weight_variable("CNNb2", [24]),
    "CNNW3" : weight_variable("CNNW3", [3,3,24,24]),
    "CNNb3" : weight_variable("CNNb3", [24]),
    "CNNW4" : weight_variable("CNNW4", [3,3,24,24]),
    "CNNb4" : weight_variable("CNNb4", [24]),
    
    # g weights
    "gW1" : weight_variable("gW1", shape=[180,256]),
    "gb1" : weight_variable("gb1", shape=[256]),
    "gW2" : weight_variable("gW2", shape=[256,256]),
    "gb2" : weight_variable("gb2", shape=[256]),
    "gW3" : weight_variable("gW3", shape=[256,256]),
    "gb3" : weight_variable("gb3", shape=[256]),
    "gW4" : weight_variable("gW4", shape=[256,256]),
    "gb4" : weight_variable("gb4", shape=[256]),
    
    # f weights
    "fW1" : weight_variable("fW1", shape=[256,256]),
    "fb1" : weight_variable("fb1", shape=[256]),
    "fW2" : weight_variable("fW2", shape=[256,256]),
    "fb2" : weight_variable("fb2", shape=[256]),
    "fW3" : weight_variable("fW3", shape=[256,29]),
    "fb3" : weight_variable("fb3", shape=[29])
}

# CNN + coordinate tagging + concat question embedding
def mergeCNNLSTM(X, lstmo):
#def CNN_output(X):
    # CONV layer 1
    cnn1 = tf.nn.conv2d(X, weights["CNNW1"], strides=[1, 2, 2, 1], padding='SAME') + weights["CNNb1"]
    cnn1bn = tf.contrib.layers.batch_norm(cnn1, center=True, scale=True, is_training=phase, scope='bn')
    cnn1relu = tf.nn.relu(cnn1bn)
    
    # CONV layer 2
    cnn2 = tf.nn.conv2d(cnn1relu, weights["CNNW2"], strides=[1, 2, 2, 1], padding='SAME') + weights["CNNb2"]
    cnn2bn = tf.contrib.layers.batch_norm(cnn2, center=True, scale=True, is_training=phase, scope='bn')
    cnn2relu = tf.nn.relu(cnn2bn)
    
    # CONV layer 3
    cnn3 = tf.nn.conv2d(cnn2relu, weights["CNNW3"], strides=[1, 2, 2, 1], padding='SAME') + weights["CNNb3"]
    cnn3bn = tf.contrib.layers.batch_norm(cnn3, center=True, scale=True, is_training=phase, scope='bn')
    # is relu necessary for the final layer of CNN?
    cnn3relu = tf.nn.relu(cnn3bn)
    
    # CONV layer 4
    cnn4 = tf.nn.conv2d(cnn3relu, weights["CNNW4"], strides=[1, 2, 2, 1], padding='SAME') + weights["CNNb4"]
    cnn4bn = tf.contrib.layers.batch_norm(cnn4, center=True, scale=True, is_training=phase, scope='bn')
    cnn4relu = tf.nn.relu(cnn4bn)
    
    feat_width = int(cnn4relu.shape[2])
    feat_height = int(cnn4relu.shape[1])
    perm_helper_list = list(range(feat_width * feat_height))
    perm_helper_list = list(itertools.permutations(perm_helper_list, r=2))
   
    # coordinate tagging
    output_list = []
    for a,b in perm_helper_list :
        x_coord = a % feat_width
        y_coord = a / feat_height
        object1 = tf.concat([cnn4relu[0,x_coord, y_coord], [x_coord], [y_coord]], axis = 0)
        
        x_coord = b % feat_width
        y_coord = b / feat_height
        object2 = tf.concat([cnn4relu[0,x_coord, y_coord], [x_coord], [y_coord]], axis = 0)
        
        object_concat = tf.concat([object1,object2,lstmo], axis=0)
        output_list.append(object_concat)
        
    return tf.stack(output_list, axis=0)

# question embedding
def LSTM_output(question_X):
    # current shape: [batch:1, seqLength, 87]
    # want : seqLength tensors of shape: [batch:1, 87]
    question_X = tf.unstack(question_X, max_seq_length, 1)
    
    # Define an LSTM cell
    lstmCell = tf.contrib.rnn.BasicLSTMCell(nHidden)
    
    # Get LSTM cell output
    outputs, states = tf.contrib.rnn.static_rnn(lstmCell, question_X, dtype=tf.float32, sequence_length=sentenceLen)
   
    # obtain the final 'valid' output
    return tf.squeeze(tf.gather(outputs,sentenceLen-1))
    #return tf.squeeze(outputs[-1])


# Deprecated : merging cnn/lstm outputs are done in CNN_output(currently mergeCNNLSTM) defined above.
#def mergeCNNLSTM(visObjects, qEncoding):
#
#    visObjects = tf.unstack(visObjects, axis=0)
#    output_list = []
#    for el in visObjects :
#        output_list.append(tf.concat([el,qEncoding], axis=0))
#    return tf.stack(output_list, axis=0)

def gTheta(gInput):
    # 256 - 256 - 256 - 256 with ReLU non-linearities.
    h1 = tf.nn.relu(tf.matmul(gInput, weights["gW1"]) + weights["gb1"])
    h2 = tf.nn.relu(tf.matmul(h1, weights["gW2"]) + weights["gb2"])
    h3 = tf.nn.relu(tf.matmul(h2, weights["gW3"]) + weights["gb3"])
    h4 = tf.nn.relu(tf.matmul(h3, weights["gW4"]) + weights["gb4"])

    # element-wise sum.
    return tf.reshape(tf.reduce_sum(h4,0), [-1, 256])

def fPhi(fInput):
    # 256 - 256 - 29 with ReLU non-linearities.
    h1 = tf.nn.relu(tf.matmul(fInput, weights["fW1"]) + weights["fb1"])
    h2 = tf.nn.relu(tf.matmul(h1, weights["fW2"]) + weights["fb2"])
    h2do = tf.nn.dropout(h2, keep_prob=keep_prob)
    h3 = tf.matmul(h2do, weights["fW3"]) + weights["fb3"]
    
    return h3

# Updated : concatenation of visual objects and lstm output is done in mergeCNNLSTM
# this modification consumes less memory.
lstmo = LSTM_output(question_X)
gInput = mergeCNNLSTM(X, lstmo)
fInput = gTheta(gInput)
fOutput = fPhi(fInput)

# loss and optimizer
cost = tf.nn.softmax_cross_entropy_with_logits(logits=fOutput, labels=Y)
train = tf.train.AdamOptimizer(2.5*math.exp(-4)).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    qr = questionReader(TRAINQUESTIONS)
    qMap, aMap = qr.preprocess()
   
    for epochCount in range(EPOCHS):
        
        doneReading = False
        questionNum = 0
        
        while not doneReading:
            questionNum += 1
            imageFN, question, answer, doneReading =  qr.readNextPair()
            image = mpimg.imread(IMGDIR + "train/" + imageFN)
            image = np.reshape(imresize(image, [128,128,4]), [-1,128,128,4])
       
            # Convert the answer into one hot vector
            ansY = np.zeros(29)
            ansY[aMap[answer]] = 1
            ansY = np.reshape(ansY, [-1,29])
    
            # Convert the question into a list of word-indices
            question = re.sub('[\"\?]', '', question.lower())
            qList = question.split()
            senLen = len(qList)
            qIndices = np.reshape(np.asarray([qMap[el] for el in qList]), [len(qList),1])
        
            # One hot encode the question words
            questionOH = []
            for qIndex in range(max_seq_length) :
                temp = np.zeros(87)
                if qIndex < senLen:
                    temp[qIndices[qIndex]] = 1
                questionOH.append(temp)
            
            # Reshape question to [batch:1, sentenceLen, 87(vocab size)]
            questionOH = np.reshape(np.asarray(questionOH), [-1, max_seq_length, 87])
    
            fout, loss, _ = sess.run([fOutput, cost, train], feed_dict={X:image, phase:True, question_X: questionOH, sentenceLen: senLen, keep_prob: 0.5, Y:ansY})
    
            print "[EPOCH : %d][Q : %d] loss : %f" % (epochCount, questionNum, loss)
            print "question %s, answer: %d, mine: %d" % (question, aMap[answer], np.argmax(fout[0]))
