# SHI Yunjiao 3036191025, Total number of parameters: 693951; Test accuracy: 99.56%
import numpy as np
import urllib
import warnings
import optuna
from keras.backend import clear_session
from keras.datasets import mnist
from tensorflow import keras
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras

opener = urllib.request.build_opener()
opener.addheaders = [("User-agent", "Mozilla/5.0")]
urllib.request.install_opener(opener)

N_TRAIN_EXAMPLES = 60000
N_VALID_EXAMPLES = 10000
BATCHSIZE = 128
CLASSES = 10
EPOCHS = 80

gen = ImageDataGenerator(rotation_range=12, width_shift_range=0.1, shear_range=0.1,
                         height_shift_range=0.1, zoom_range=0.1, fill_mode='nearest')

early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.4, min_lr=0.00002)

def create_model(trial):
    model = Sequential()

    # 使用6x6的卷积核，输出24个特征图，步幅为1，激活函数为ReLU，添加批量归一化
    model.add(keras.layers.Conv2D(filters=24, kernel_size=6, strides=1, padding='same', input_shape=(28, 28, 1)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dropout(0.3))  # 参考代码中的pkeep_conv

    # 使用5x5的卷积核，输出48个特征图，步幅为2，激活函数为ReLU，添加批量归一化
    model.add(keras.layers.Conv2D(filters=48, kernel_size=5, strides=2, padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dropout(0.3))  # 参考代码中的pkeep_conv

    # 使用4x4的卷积核，输出64个特征图，步幅为2，激活函数为ReLU，添加批量归一化
    model.add(keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dropout(0.3))  # 参考代码中的pkeep_conv

    model.add(keras.layers.Flatten())

    # 密集层有200个单元，激活函数为ReLU，添加批量归一化
    model.add(keras.layers.Dense(195))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dropout(0.3))  # 参考代码中的pkeep

    # 输出层
    model.add(keras.layers.Dense(CLASSES, activation='softmax'))

    model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def objective(trial):
    clear_session()

    (x_train, y_train), (x_valid, y_valid) = mnist.load_data()
    x_train = np.expand_dims(x_train, -1).astype("float32") / 255
    x_valid = np.expand_dims(x_valid, -1).astype("float32") / 255
    y_train = y_train.astype("float32")
    y_valid = y_valid.astype("float32")

    train_generator = gen.flow(x_train, y_train, batch_size=BATCHSIZE)

    model = create_model(trial)
    
    model.fit(x_train, y_train, batch_size=100, epochs=10, validation_data=(x_valid, y_valid))

    result = model.evaluate(x_valid, y_valid)
    print("zhunquelv!!!(%): ", result[1]*100)

    history = model.fit(
        train_generator,
        steps_per_epoch=N_TRAIN_EXAMPLES // BATCHSIZE,
        epochs=EPOCHS,
        validation_data=(x_valid, y_valid),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    score = model.evaluate(x_valid, y_valid, verbose=0)
    return score[1]

if __name__ == "__main__":
    warnings.warn(
        "Recent Keras release (2.4.0) simply redirects all APIs "
        "in the standalone keras package to point to tf.keras. "
        "There is now only one Keras: tf.keras. "
        "There may be some breaking changes for some workflows by upgrading to keras 2.4.0. "
        "Test before upgrading. "
        "REF:https://github.com/keras-team/keras/releases/tag/2.4.0"
    )

    (x_train, y_train), (x_valid, y_valid) = mnist.load_data()
    x_train = np.expand_dims(x_train, -1).astype("float32") / 255
    x_valid = np.expand_dims(x_valid, -1).astype("float32") / 255
    y_train = y_train.astype("float32")
    y_valid = y_valid.astype("float32")

    # use optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, timeout=600)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)

    best_model = create_model(trial)
    best_model.build(input_shape=(None, 28, 28, 1))
    best_model.summary()

    best_model.fit(
        x_train, y_train,
        batch_size=BATCHSIZE,
        epochs=EPOCHS,
        validation_data=(x_valid, y_valid),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    best_model.save('best_model.keras')

    model = keras.models.load_model('best_model.keras')

    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_test = np.expand_dims(x_test, -1).astype("float32") / 255
    y_test = y_test.astype("float32")

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    print(f"acc: {test_accuracy * 100:.2f}%")

