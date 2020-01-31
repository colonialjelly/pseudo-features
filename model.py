import tensorflow as tf
import utils


class Model:
    def __init__(self, num_dim):
        self.W = tf.Variable(tf.random.normal(shape=(num_dim, 1), dtype=tf.float64))
        self.b = tf.Variable(tf.zeros(shape=(1, ), dtype=tf.float64))

    def forward(self, x):
        return tf.nn.sigmoid(tf.add(tf.matmul(x, self.W), self.b))

    def fit(self, X_train, y_train, loss_type, lr=1., epochs=100, verbose=0):
        if loss_type != 'bce' and loss_type != 'out':
            raise Exception('Invalid loss type')
        for i in range(epochs):
            gradients, loss_val = self.step(X_train, y_train, loss_type)
            dW, db = gradients
            self.W.assign_sub(lr * dW)
            self.b.assign_sub(lr * db)
            y_pred = self.forward(X_train)
            if verbose:
                acc = utils.acc_out(y_train, y_pred)
                if i % 10 == 0:
                    print(f"loss: {loss_val} accuracy: {acc}")
                    print("-" * 50)

    def step(self, x, y, loss_type):
        with tf.GradientTape() as t:
            y_pred = self.forward(x)
            if loss_type == 'bce':
                loss_val = utils.binary_crossentropy(y, y_pred)
            else:
                loss_val = utils.out_loss(y, y_pred)
        return t.gradient(loss_val, [self.W, self.b]), loss_val

    def predict(self, x):
        y_pred = self.forward(x).numpy()
        msk = y_pred >= 0.5
        y_pred[msk] = 1
        y_pred[~msk] = 0
        return y_pred