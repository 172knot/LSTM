import tensorflow as tf
import progressbar
import numpy as np
import argparse
import glob
import os



def parse_args():

    desc = "Basic LSTM for text prediction"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--text_path', type=str, default='./text.txt', help='File path of text file to be read')

    parser.add_argument('--logs_path', type=str, default='./logs', help='File path of logs for tensorboard')

    parser.add_argument('--n_hidden', type=int, default=512, help='Number of hidden units in a single LSTM cell')

    parser.add_argument('--n_input', type=int, default=3, help='Number of input words')

    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate for Adam optimizer')

    parser.add_argument('--num_epochs', type=int, default=1000, help='The number of epochs to run')

    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')

    return check_args(parser.parse_args())

def check_args(args):


    try:
        f = open(args.text_path,'r')
    except Exception:
        print("no such file found to be read")

    try:
        os.mkdir(args.logs_path)
    except Exception:
        pass

    files = glob.glob(args.logs_path+'/*')
    for f in files:
        os.remove(f)

    try:
        assert args.n_hidden >= 1
    except:
        print('number of hidden units must be larger than one')

    try:
        assert args.n_input >= 1
    except:
        print('number of words must be grater than or equal to one')

    try:
        assert args.learning_rate > 0
    except:
        print('learning rate must be positive')

    try:
        assert args.num_epochs >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args


tf.reset_default_graph()
vocab_size = 0


def get_data(args):
    data_path = args.text_path
    fp = open(data_path,"r")
    dict_ = []
    for word in fp.read().split():
        if(word in dict_):
            continue
        dict_.append(word)

    global vocab_size
    vocab_size = len(dict_)

    dict1 = {}
    dict2 = {}
    for i in range(len(dict_)):
        dict1[dict_[i]]  = i+1
        dict2[i+1] = dict_[i]
    return dict1, dict2



def gen_data(for_dic, inv_dic, args):
    data_path = args.text_path
    fp = open(data_path,"r")
    inp = []
    op = []

    temp = []
    ct = 0
    for word in fp.read().split():
        if(ct==3):
            inp.append(temp)
            ct = 0
            temp = []
            temp.append([for_dic[word]])
            ct+=1
            yu = np.zeros((vocab_size), dtype = float)
            yu[for_dic[word]] = 1.0
            op.append([yu])
            continue

        temp.append([for_dic[word]])
        ct += 1

    return inp, op



def rnn(inp_, weight, bias, args):

    n_input = args.n_input
    n_hidden = args.n_hidden
    inp_ = tf.reshape(inp_, [-1, n_input])
    inp_ = tf.split(inp_, n_input, 1)

    rnn_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
    outputs, states = tf.contrib.rnn.static_rnn(rnn_cell, inp_, dtype=tf.float32)

    return tf.matmul(outputs[-1], weight) + bias


def main(args):

    logs_path = args.logs_path
    n_input = args.n_input
    n_hidden = args.n_hidden
    epochs = args.num_epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size



    for_dic, inv_dic = get_data(args)
    weight = tf.Variable(tf.random_normal([n_hidden, vocab_size]))
    bias =  tf.Variable(tf.random_normal([vocab_size]))

    x = tf.placeholder(tf.float32, shape = (None, n_input, 1))
    y = tf.placeholder(tf.float32, shape = (None, vocab_size))

    pred = rnn(x, weight, bias, args)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

    correct = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    tf.summary.scalar("cost", cost)
    tf.summary.scalar("accuracy ", accuracy)

    summary_op = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    with tf.Session() as sess:
        # sess.run(init)
        saver.restore(sess,"./freeze/model.ckpt")
        X, Y = gen_data(for_dic, inv_dic, args)
        X = np.array(X)
        Y = np.array(Y)
        fp = open("text.txt","r")
        ct = 0
        pt = 0
        for epoch in range(epochs):
            epoch_loss = 0
            avg_acc = 0
            if(epoch%50==0 and epoch!=0):
                learning_rate = learning_rate*0.7
            for i in progressbar.progressbar(range(len(X))):
                batchx = np.zeros((1,3,1))
                batchy = np.zeros((1,112))

                batchx[0] = X[i,:,:]
                batchy[0] = Y[i,:,:]
                # print(inv_dic[batchx[0,0,0]], inv_dic[batchx[0,1,0]], inv_dic[batchx[0,2,0]])
                batchx = np.array(batchx)
                batchy = np.array(batchy)
                _, loss, onehot_pred, acc, summary = sess.run([optimizer, cost, pred, accuracy, summary_op], feed_dict={x: batchx, y: batchy})
                writer.add_summary(summary, pt)
                pt+=1
                avg_acc +=  acc
                saver.save(sess, "./freeze/model.ckpt")
                temp = onehot_pred[0,:]
                temp = temp.tolist()
                oht_pred_index = int(np.argmax(temp))

                temp2 = batchy[0,:]
                temp2 = temp2.tolist()
                oht_pred_index2 = int(np.argmax(temp2))
                epoch_loss += loss
            avg_acc /= len(X)
            print("Epoch: ",epoch,"loss: ", epoch_loss, "acc: ", avg_acc)


if(__name__=="__main__"):
    args = parse_args()
    if args is None:
        exit()
    main(args)
