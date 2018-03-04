import tensorflow as tf
import numpy as np

def preprocess():
    dict_words = {"i":2,"am":1,"rohan":4}
    input = [["i","am"],["am","i"],["rohan","i"]]
    label = ["anger","happy","anger"]
    return (dict_words,input,label)

def getUniqueWords(dict_words,threshold):
    list_words = {}
    for word in dict_words:
        if dict_words[word] > threshold:
            list_words[word] = len(list_words)
    return list_words

def createOneHotInputVector(input, words_index):
    data = []
    for sent in input:
        oneHot = np.zeros((len(sent),len(words_index)+1))
        for index in range(0,len(sent)):
            dimension = len(words_index)
            if (sent[index] in words_index):
                dimension = words_index[sent[index]]
            oneHot[index][dimension] = 1
        data.append(oneHot)
    return data

def createOneHotLabelVector(label):
    uniqueLabel = set(label)
    dict_labels = {}
    for val in uniqueLabel:
        if val not in dict_labels:
            dict_labels[val] = len(dict_labels)

    train_label = []
    for val in label:
        oneHot = np.zeros(len(dict_labels))
        oneHot[dict_labels[val]] = 1
        train_label.append(oneHot)

    return (train_label,len(dict_labels))

def main():
    session = tf.Session()
    dict_words,input,label = preprocess()
    threshold = 1
    words_index = getUniqueWords(dict_words,threshold)
    train_data = createOneHotInputVector(input,words_index)
    train_label,hidden_size = createOneHotLabelVector(label)
    numSent = 100
    numWords = len(words_index)+1

    """
    print(dict_words,input,label)
    print(words_index)
    for ele in train_data:
        print(ele)
    print("----------------")
    for ele in train_label:
        print(ele) 
    """

    Xtrain = tf.placeholder(tf.float32, [None, numSent, numWords])
    Ytrain = tf.placeholder(tf.float32, [None])

    lstm_cell_1 = tf.contrib.rnn.LSTMCell(hidden_size)
    lstm_cell_2 = tf.contrib.rnn.LSTMCell(hidden_size)
    multi_lstm_cells = tf.contrib.rnn.MultiRNNCell(cells=[lstm_cell_1, lstm_cell_2], state_is_tuple=True)
    _, final_state = tf.nn.dynamic_rnn(multi_lstm_cells,numSent,dtype=tf.float32)


    prediction = tf.nn.softmax(final_state[-1][-1],1)

    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=Ytrain)
    loss = tf.reduce_mean(loss)

    pred = tf.argmax(prediction,1)
    pred_err = tf.to_float(tf.not_equal(pred,tf.argmax(Ytrain,1)))
    pred_err = tf.reduce_sum(pred_err)

    optimizer = tf.train.AdamOptimizer().minimize(loss)

    tf.global_variables_initializer().run(session=session)


    num_steps = 10
    batch_size = 64

    for step in range(num_steps):
        # get data for a batch
        offset = (step * batch_size) % (len(train_data) - batch_size)
        batch_train = train_data[offset: (offset + batch_size)]
        batch_labels = train_label[offset: (offset + batch_size)]

        # put this data into a dictionary that we feed in when we run
        # the graph.  this data fills in the placeholders we made in the graph.
        data = {Xtrain: batch_train, Ytrain: batch_labels}

        # run the 'optimizer', 'loss', and 'pred_err' operations in the graph
        _, loss_value_train, error_value_train = session.run(
            [optimizer, loss, pred_err], feed_dict=data)

        # print stuff every 50 steps to see how we are doing
        if (step % 50 == 0):
            print("Minibatch train loss at step", step, ":", loss_value_train)
            print("Minibatch train error: %.3f%%" % error_value_train)

            # get test evaluation

            """
            test_loss = []
            test_error = []
            for batch_num in range(int(len(test_data) / batch_size)):
                test_offset = (batch_num * batch_size) % (len(test_data) - batch_size)
                test_batch_tweets = one_hot_test_tweets[test_offset: (test_offset + batch_size)]
                test_batch_labels = test_labels[test_offset: (test_offset + batch_size)]
                data_testing = {tweets: test_batch_tweets, labels: test_batch_labels}
                loss_value_test, error_value_test = session.run([loss, pred_err], feed_dict=data_testing)
                test_loss.append(loss_value_test)
                test_error.append(error_value_test)

            print("Test loss: %.3f" % np.mean(test_loss))
            print("Test error: %.3f%%" % np.mean(test_error))
            """


if __name__ == "__main__":
    main()
