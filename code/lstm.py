import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Conv1D, MaxPooling1D,Activation


'''

def build_lstm_model(time_steps, features, output_features):
    model = Sequential([
        Input(shape=(time_steps, features)),  
        LSTM(128, return_sequences=False),
        #Dropout(0.3),
        Dense(output_features, activation='linear')  
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model
'''

def build_lstm_model(time_steps, features, output_steps, output_features):
    model = Sequential([
        Input(shape=(time_steps, features)),  
        LSTM(16, return_sequences=False),    
        #Dropout(0.3),
        Dense(output_steps * output_features, activation='linear'),  
        #Dense(output_features, activation='linear')  
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model

'''
def build_lstm_complex_model(input_shape, forward, layer1, layer2):
    model = Sequential()
    # 添加卷积层
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    # 第一个LSTM层
    model.add(LSTM(layer1, return_sequences=True))
    model.add(Dropout(0.3))
    # 第二个LSTM层
    model.add(LSTM(layer2, return_sequences=False))
    model.add(Dropout(0.3))
    # 全连接层
    model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.2))
    # 输出层
    model.add(Dense(forward))
    model.add(Activation("relu"))
    return model
'''

'''

def build_lstm_model(time_steps, features, output_features):
    model = Sequential([
        Input(shape=(time_steps, features)),  
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(output_features, activation='linear')  
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model
'''

'''
def build_lstm_model(time_steps, input_features, output_features):
    model = Sequential([
        Input(shape=(time_steps, input_features)),  
        LSTM(64, return_sequences=False),  
        Dense(output_features, activation='relu') 
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model
'''


'''
def build_and_train_lstm(X_train, y_train, time_steps, features, save_path="lstm_model.h5"):

    model = build_lstm_model(time_steps, features)

    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

    model.save(save_path)

    return model
'''