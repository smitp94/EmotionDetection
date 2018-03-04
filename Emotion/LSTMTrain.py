import tensorflow as tf
import numpy as np
import datetime
from EmotionDetection.Emotion.Preprocessor import pre_process

def getUniqueWords(dict_words,threshold):
    list_words = {}
    for word in dict_words:
        if dict_words[word] > threshold:
            list_words[word] = len(list_words)
    return list_words

def createOneHotInputVector(input, words_index,numSent,start,size):
    data = []
    for sent in input[start:start+size]:
        oneHot =[[0 for i in range(0,len(words_index)+1)]for j in range(0,numSent)]
        for index in range(0,min(len(sent),numSent)):
            #dimension = len(words_index)
            dimension = 0
            if (sent[index] in words_index) and (words_index[sent[index]] + 1 < min(len(sent),numSent)):
                dimension = words_index[sent[index]] + 1
            oneHot[index][dimension] = 1
        data.append(oneHot)
    return data

def uniqueLabels(label):
    uniqueLabel = set(label)
    dict_labels = {}
    for val in uniqueLabel:
        if val not in dict_labels:
            dict_labels[val] = len(dict_labels)
    return(dict_labels,len(dict_labels))

def createOneHotLabelVector(dict_labels,label,start,size):
    train_label = []
    for val in label[start:start+size]:
        #oneHot = np.zeros(len(dict_labels))
        oneHot = [0 for i in range(0,len(dict_labels))]
        oneHot[dict_labels[val]] = 1
        train_label.append(oneHot)

    return train_label

def main():
    session = tf.Session()
    dict_words,input,label = pre_process()
    #input = input[:10000]
    #print("input:",np.array(input).shape)
    threshold = 50
    numSent = 85
    words_index = getUniqueWords(dict_words,threshold)
    label_index,hidden_size =uniqueLabels(label)

    #train_data = createOneHotInputVector(input,words_index,start,size)
    #train_label = createOneHotLabelVector(label_index,label)
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

    Xtrain = tf.placeholder(tf.float32, [None,numSent, numWords])
    Ytrain = tf.placeholder(tf.float32, [None,hidden_size])

    lstm_cell_1 = tf.contrib.rnn.LSTMCell(hidden_size)
    #lstm_cell_2 = tf.contrib.rnn.LSTMCell(hidden_size)
    #lstm_cell_3 = tf.contrib.rnn.LSTMCell(hidden_size)
    multi_lstm_cells = tf.contrib.rnn.MultiRNNCell(cells=[lstm_cell_1], state_is_tuple=True)
    _, final_state = tf.nn.dynamic_rnn(multi_lstm_cells,Xtrain,dtype=tf.float32)


    prediction = tf.nn.softmax(final_state[-1][-1],1)

    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=Ytrain)
    loss = tf.reduce_mean(loss)

    pred = tf.argmax(prediction,1)
    pred_err = tf.to_float(tf.equal(pred,tf.argmax(Ytrain,1)))
    error = tf.reduce_sum(pred_err)

    optimizer = tf.train.AdamOptimizer().minimize(loss)

    tf.global_variables_initializer().run(session=session)


    num_steps = 10000
    batch_size = 50

    tf.summary.scalar('error',error)
    time_string = datetime.datetime.now().isoformat()
    experiment_name = "error"
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('data/{experiment_name}',session.graph)

    for step in range(num_steps):
        offset = (step * batch_size) % (len(input) - batch_size)
        batch_train = createOneHotInputVector(input,words_index,numSent,offset, batch_size)
        #batch_labels = train_label[offset: (offset + batch_size)]
        batch_labels = createOneHotLabelVector(label_index, label, offset, batch_size)

        # print("train")
        # print(np.array(train_data).shape)
        # print(np.array(batch_train).shape)
        # print("label")
        # print(np.array(batch_labels).shape)
        # put this data into a dictionary that we feed in when we run
        # the graph.  this data fills in the placeholders we made in the graph.
        data = {Xtrain: batch_train, Ytrain: batch_labels}

        _, loss_value_train, error_value_train = session.run(
            [optimizer, loss,error], feed_dict=data)

        if (step % 50 == 0):
            print("Minibatch train loss at step", step, ":", loss_value_train)
            print("Minibatch train error: %.2f%%" % (error_value_train))

        summary,acc = session.run([merged,error],feed_dict=data)
        train_writer.add_summary(summary,step)
            # get test evaluation
if __name__ == "__main__":
    main()
