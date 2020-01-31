from itertools import product
import numpy as np

NUM_VARS = 7


def gen_bool_table(num_vars):
    return np.array(list(product([0, 1], repeat=num_vars))).astype('float64')


def func(r):
    x1, x2, x3, x4, x5, x6, x7 = r
    return (x1 and not x2) and (x3 or x4) or x1 or (x5 and x6) and not x7


def shuffle(X, y):
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]


def split(X, y, randomize_last=True):
    # Split the dataset into train and test and randomize the pseudo feature for the test set.
    idx = np.arange(X.shape[0])
    idx_train, idx_test = np.array_split(idx, 2)
    X_train = X[idx_train]
    y_train = y[idx_train]
    X_test = X[idx_test]
    y_test = y[idx_test]

    if randomize_last:
        # The test set will not have the pseudo feature perfectly correlated with the label
        X_test[:, -1] = np.random.randint(2, size=X_test.shape[0])
    
    return X_train, y_train, X_test, y_test


def generate_data():
    # Generate the dataset
    X = gen_bool_table(NUM_VARS)
    y = np.apply_along_axis(func, 1, X)
    X, y = shuffle(X, y)

    # Store uncorrupted data
    data_original = split(X.copy(), y.copy(), randomize_last=False)

    # Add the pseudo feature to the data
    X = np.hstack((X, y.reshape(X.shape[0], 1)))
    data_in = split(X, y)

    # Create OOD data:
    # It has zeroes for all feature values except for the pseudo feature
    # pseudo feature corresponds to the last column
    X_prime = np.zeros_like(X)
    X_prime[:, -1] = y
    
    # OOD data don't have class labels, therefore we're randomizing this column
    y_prime_in = np.random.randint(2, size=X.shape[0])
    
    # This column indicates if the sample is from in our out distribution
    # all ones means that all of the samples are OOD
    y_prime_out = np.ones_like(y_prime_in)
    y_prime = np.column_stack((y_prime_in , y_prime_out))

    # Need the extra column to distinguish between in and out distribution data
    # 0 means that the sample is from in distribution, 1 means out distribution
    y = np.column_stack((y , np.zeros_like(y)))
    X_train, y_train, X_test, y_test = split(X, y)
    
    # Create a dataset that contains both in and out distribution samples
    X_train_combined = np.vstack((X_train, X_prime))
    y_train_combined = np.vstack((y_train, y_prime))
    X_train_combined, y_train_combined = shuffle(X_train_combined, y_train_combined)
    
    data_in_out = X_train_combined, y_train_combined, X_test, y_test
    
    return data_in, data_in_out, data_original

