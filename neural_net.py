import tensorflow as tf

def build_model(input_shape):
    modelo = tf.keras.Sequential()
    modelo.add(tf.keras.layers.Dense(units=100, activation='relu', input_shape=(input_shape,)))
    modelo.add(tf.keras.layers.Dense(units=100, activation='relu'))
    modelo.add(tf.keras.layers.Dense(units=100, activation='relu'))
    modelo.add(tf.keras.layers.Dense(units=1, activation='linear'))
    
    modelo.compile(optimizer='ADAM', loss='mean_squared_error')
    return modelo