from keras.layers import Conv2D, Dense, MaxPooling2D, Input, Flatten, Dropout, Activation, BatchNormalization
from keras import Model


def EthnicityNet():
	inp = Input(shape=(200, 200, 3,))

	net = Conv2D(filters=32, strides=(2,2), kernel_size=(3,3))(inp)
	net = BatchNormalization()(net)
	net = Activation('elu')(net)
	net = Dropout(0.5)(net)

	net = Conv2D(filters=64, strides=(2,2), kernel_size=(3,3))(net)
	net = BatchNormalization()(net)
	net = Activation('elu')(net)
	net = Dropout(0.5)(net)

	net = Conv2D(filters=128, strides=(2,2), kernel_size=(3,3))(net)
	net = BatchNormalization()(net)
	net = Activation('elu')(net)
	net = Dropout(0.5)(net)

	net = Conv2D(filters=128, strides=(2,2), kernel_size=(3,3))(net)
	net = BatchNormalization()(net)
	net = Activation('elu')(net)
	net = Dropout(0.5)(net)

	net = Conv2D(filters=128, strides=(2,2), kernel_size=(3,3))(net)
	net = BatchNormalization()(net)
	net = Activation('elu')(net)
	net = Dropout(0.5)(net)

	net = Conv2D(filters=256, kernel_size=(3,3))(net)
	net = BatchNormalization()(net)
	net = Activation('elu')(net)
	net = Dropout(0.5)(net)

	net = Conv2D(filters=256, kernel_size=(3,3))(net)
	net = BatchNormalization()(net)
	net = Activation('elu')(net)
	net = Dropout(0.5)(net)

	net = Flatten()(net)

	net = Dense(256, activation='relu')(net)
	net = Dense(512, activation='relu')(net)
	out = Dense(5, activation='softmax')(net)

	model = Model(inputs=[inp], outputs=[out])

	return model