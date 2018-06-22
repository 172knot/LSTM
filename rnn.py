import numpy as np
import tensorflow as tf
import progressbar

tf.reset_default_graph()


logs_path = "./logs"
vocab_size = 0
n_input = 3
n_hidden = 512
epochs = 100000
learning_rate = 0.00001
batch_size = 8
def get_data():
    fp = open("text.txt","r")
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



def gen_data(for_dic, inv_dic):
    fp = open("text.txt","r")
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



def rnn(inp_, weight, bias):

    inp_ = tf.reshape(inp_, [-1, n_input])
    inp_ = tf.split(inp_, n_input, 1)

    rnn_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
    outputs, states = tf.contrib.rnn.static_rnn(rnn_cell, inp_, dtype=tf.float32)

    return tf.matmul(outputs[-1], weight) + bias


def main():
    global learning_rate
    for_dic, inv_dic = get_data()
    weight = tf.Variable(tf.random_normal([n_hidden, vocab_size]))
    bias =  tf.Variable(tf.random_normal([vocab_size]))

    x = tf.placeholder(tf.float32, shape = (None, n_input, 1))
    y = tf.placeholder(tf.float32, shape = (None, vocab_size))

    pred = rnn(x, weight, bias)
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
        X, Y = gen_data(for_dic, inv_dic)
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
    main()
