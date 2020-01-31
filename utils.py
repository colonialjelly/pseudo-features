import numpy as np
import tensorflow as tf


def out_loss(y_true, y_pred):
	in_samples = y_true[:, 1] < 1
	y_true_in = tf.boolean_mask(y_true, in_samples)[:, 0]
	y_pred_in = tf.squeeze(tf.boolean_mask(y_pred, in_samples), axis=1)
	y_pred_out = tf.squeeze(tf.boolean_mask(y_pred, ~in_samples), axis=1)
	bce = binary_crossentropy(y_true_in, y_pred_in)

	q = tf.cast(tf.fill((tf.size(y_pred_out), 2), 0.5), dtype=tf.float64)
	p = tf.stack([y_pred_out, 1 - y_pred_out], axis=1)
	kl = tf.reduce_mean(kl_divergence(p, q))

	return bce + 2*kl


def binary_crossentropy(y_true, y_pred):
	y_pred = tf.keras.backend.flatten(y_pred)
	loss = -y_true*tf.math.log(y_pred) - (1. - y_true)*tf.math.log(1. - y_pred)
	return tf.reduce_mean(loss)


def kl_divergence(p, q):
	return tf.reduce_sum(p * (tf.math.log(p) - tf.math.log(q)), axis=1)


def accuracy(y_true, y_pred):
	if y_true.ndim > 1:
		y_true = y_true[:, 0]
	y_pred = np.array(y_pred)
	y_pred[y_pred >= 0.5] = 1
	y_pred[y_pred < 0.5] = 0
	y_pred = y_pred.flatten()
	return np.mean(np.equal(y_true, y_pred))


def acc_out(y_true, y_pred):
	in_samples = y_true[:, 1] < 1
	y_true = tf.boolean_mask(y_true, in_samples)[:, :1]
	y_pred = tf.boolean_mask(y_pred, in_samples)

	return accuracy(y_true, y_pred)