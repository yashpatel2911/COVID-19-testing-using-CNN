import pickling_dataset
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding='valid', activation=tf.nn.relu, input_shape=(100, 100, 3)),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3, 3), padding='valid', activation=tf.nn.relu),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3, 3), padding='valid', activation=tf.nn.relu),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(units=4096, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(units=512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid)

])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

x_train, y_train, x_test, y_test = pickling_dataset.load_dataset(image_size=100)

model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=5,
    validation_data=(x_test, y_test),
    shuffle=True,
    verbose=1
)

model.save('Output_model/Model.h5')
