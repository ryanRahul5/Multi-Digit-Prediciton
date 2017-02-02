import tensorflow as tf

'''
i/p> weight> hidden layer1 (activation (non-linear)fumnction) >weights> hidden layer2
> weights > o/p layer 

in neural network we pass the data straight through which is feed-forward neural network

later we compare intended o/p > cost/loss function (cross entropy)
optimizer > minimize cost (SGD, AdaGrad,..) so it goes backwards and manipulate the weights
known as backpropogation

feed forward + backprop = epoch (1 cycle)
'''

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

'''
one hot rule

0 = [1,0,0,0,0,0,0,0,0,0]
1 = [0,1,0,0,0,0,0,0,0,0]
'''

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100 # batches of 100 features to feed at a time

#matrix is height x width
x = tf.placeholder('float',[None, 784]) #784 pixels 28x28
y = tf.placeholder('float',[None,10])


def neural_network_model(data):
	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
					  'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
		# we have biases in case you have zero input val(no neurons will be fired) then it'll fire the neuron with some val 

	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
					  'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
					  'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
					  'biases': tf.Variable(tf.random_normal([n_classes]))}

	# (input_data * weights) + biases
	#building model

	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1) #rectified linear activation function

	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']),  hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3)	

	output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

	return output

	# till here we have coded the model..now as computation graph is ready will execute this in sessions
	

def train_neural_network(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y))

	optimizer = tf.train.AdamOptimizer().minimize(cost)
	#Adamopt is similar to that of SGD.. it has a default learning rate of 0.001

	#cycles feed forward + backdrop
	num_epochs = 10

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())

		for epoch in range(num_epochs):
			epoch_loss = 0
			total_batch = int(mnist.train.num_examples/batch_size)

			for _ in range(total_batch):
				batch_x, batch_y = mnist.train.next_batch(batch_size)
				_, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})

				epoch_loss += c
			print 'Epoch', epoch#, 'completed out of', hm_epochs, 'loss:', epoch_loss
		#training is complete

		#test the data
		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))

		#Calculate accuracy
		accuracy = tf.reduce_mean(tf.cast(correct,'float'))
		print 'Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels})

train_neural_network(x)













