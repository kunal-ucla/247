from tensorflow import keras
import tensorflow as tf

def skipnet(input_shape=(44*22, 62, 1), num_filters1=16, kernel_size1=(44*22,1), stride1=(1,1), num_filters2=16, kernel_size2=(1,3), stride2=(1,1), fc_num=128, num_classes=4):
    inputs = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv2D(num_filters1, kernel_size1, stride1, padding='valid', activation='relu')(inputs)
    bn1 = keras.layers.BatchNormalization()(conv1)

    conv2 = keras.layers.Conv2D(num_filters2, kernel_size2, stride2, padding='same', activation='relu')(bn1)
    bn2 = keras.layers.BatchNormalization()(conv2)

    addition = bn1 + bn2
    flat = keras.layers.Flatten()(addition)

    fc = keras.layers.Dense(fc_num, activation='relu')(flat)
    drop = keras.layers.Dropout(0.5)(fc)

    outputs = keras.layers.Dense(num_classes, activation='softmax')(drop)

    model = keras.models.Model(inputs, outputs)
    return model

def cnn_lstm():
  # Building the CNN model using sequential class
  model = tf.keras.models.Sequential()

  # Conv. block 1
  model.add(tf.keras.layers.Conv2D(filters=25, kernel_size=(10,1), padding='same', activation='elu', input_shape=(100,1,22)))
  model.add(tf.keras.layers.MaxPooling2D(pool_size=(3,1), padding='same')) # Read the keras documentation
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.Dropout(0.5))

  # Conv. block 2
  model.add(tf.keras.layers.Conv2D(filters=50, kernel_size=(10,1), padding='same', activation='elu'))
  model.add(tf.keras.layers.MaxPooling2D(pool_size=(3,1), padding='same'))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.Dropout(0.5))

  # Conv. block 3
  model.add(tf.keras.layers.Conv2D(filters=100, kernel_size=(10,1), padding='same', activation='elu'))
  model.add(tf.keras.layers.MaxPooling2D(pool_size=(3,1), padding='same'))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.Dropout(0.5))

  # Conv. block 4
  model.add(tf.keras.layers.Conv2D(filters=200, kernel_size=(10,1), padding='same', activation='elu'))
  model.add(tf.keras.layers.MaxPooling2D(pool_size=(3,1), padding='same'))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.Dropout(0.5))

  # FC+LSTM layers
  model.add(tf.keras.layers.Flatten()) # Adding a flattening operation to the output of CNN block
  model.add(tf.keras.layers.Dense((100))) # FC layer with 100 units
  model.add(tf.keras.layers.Reshape((100,1))) # Reshape my output of FC layer so that it's compatible
  model.add(tf.keras.layers.LSTM(20, dropout=0.6, recurrent_dropout= 0.1, input_shape=(100,1), return_sequences=False))

  model.add(tf.keras.layers.Dense(4, activation='softmax')) # Output FC layer with softmax activation
  return model

def cnn(input_shape=(250,1,22), num_classes=4):
    # Building the CNN model using sequential class
    model = tf.keras.models.Sequential()

    # Conv. block 1
    model.add(tf.keras.layers.Conv2D(filters=25, kernel_size=(10,1), padding='same', activation='elu', input_shape=input_shape))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(3,1), padding='same')) # Read the keras documentation
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))

    # Conv. block 2
    model.add(tf.keras.layers.Conv2D(filters=50, kernel_size=(10,1), padding='same', activation='elu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(3,1), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))

    # Conv. block 3
    model.add(tf.keras.layers.Conv2D(filters=100, kernel_size=(10,1), padding='same', activation='elu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(3,1), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))

    # Conv. block 4
    model.add(tf.keras.layers.Conv2D(filters=200, kernel_size=(10,1), padding='same', activation='elu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(3,1), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))

    # Output layer with Softmax activation
    model.add(tf.keras.layers.Flatten()) # Flattens the input
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax')) # Output FC layer with softmax activation

    return model