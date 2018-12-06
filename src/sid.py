import numpy as np
import keras
import cv2
import tensorflow as tf

model = keras.models.load_model('cnn_model.h5')
print("Loading model successfully")
graph = tf.get_default_graph()


def ml_predict(image):
    with graph.as_default():
        prediction = model.predict(image, verbose=0)
    return prediction


def clean(image):
    # Resize the student number picture
    image = cv2.resize(image, (252, 28))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cv2.imwrite('../intermediate/thresh_student_number.png', thresh)

    # vertically split student number into 9 digits
    cells = [np.hsplit(thresh, 9)]

    # the shape is (1, 9, 28, 28), which stands for 9 numbers, each number consists of 28*28 pixels
    x = np.array(cells)
    new = []
    for i in range(9):
        filename = "single" + str(i) + ".png"
        cv2.imwrite('../intermediate/' + filename, np.reshape(x[:, i, :, :], (28, 28)))

        # TODO: try crop the middle part
        # centroid = int(28*(float(i)+0.5))
        # cropped = x[:, i, :, :][centroid-8:centroid+8, centroid-8:centroid+8]
        # new.append(cropped)
        # cv2.imwrite('../intermediate/' + filename, np.resize(cropped, (28, 28)))

    # the shape of flattened matrix is (9, 784), which stands for 9 arrays and length of each array is 784
    # student_number = x.reshape(-1, 784).astype(np.float32)
    student_number = x.reshape(9, 1, 28, 28).astype(np.float32)
    # new_array = np.asarray(new)
    # student_number = new_array.reshape(9, 1, 28, 28).astype(np.float32)
    return student_number


def read_student_id(image):
    cleaned_sid = clean(image)
    scores = ml_predict(cleaned_sid)
    result = np.argmax(scores, axis=1).tolist()
    return result