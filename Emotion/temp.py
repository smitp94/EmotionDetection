import tensorflow as tf

# constants
"""
a = tf.constant([2])
b = tf.constant([3])
c = tf.add(a,b)
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
    result = session.run(c)
    print(result)
"""
# Variables
"""
state = tf.Variable(0)
one = tf.constant(1)
new_value = tf.add(state,one)
update = tf.assign(state,new_value)
init_op = tf.initialize_all_variables()

with tf.Session() as session:
    session.run(init_op)
    print(session.run(state))
    for _ in range(3):
        session.run(update)
        print(session.run(state))

"""
"""
print("Rohan")


# placeholders to feed tensordlow data outside of a model
a = tf.placeholder(tf.float32)
dictionary = {a: [[1, 2], [2, 3]]}
b = a * 2
with tf.Session() as session:
    result = session.run(b, feed_dict=dictionary)
    print(result)
"""


list = [[2,1,2,1],[2,None,2,None],[None,1,None,1],[None,4,None,4]]
for y in range(0,4):
    for x in range(0,4):
        print(list[y][x]," ",end="")
    print("")
for y in range(0,3):
    for x in range(0,3):
        if list[x+1][y] != None:
            if list[x+1][y] == list[x][y]:
                list[x][y] *= 2
                list[x + 1][y] = None
            if list[x][y] == None:
                list[x][y] = list[x+1][y]
            list[x + 1][y] = None
print("----------------")
for y in range(0,4):
    for x in range(0,4):
        print(list[y][x]," ",end="")
    print("")

