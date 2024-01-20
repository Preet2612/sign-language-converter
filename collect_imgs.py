import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 26
dataset_size = 100

cap = cv2.VideoCapture(0)

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {}'.format(j))

    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Press "K" to capture images or "Q" to quit', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        key = cv2.waitKey(1)
        if key == ord('K') or key == ord('k'):
            break
        elif key == ord('Q') or key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)

        img_path = os.path.join(class_dir, '{}.jpg'.format(counter))
        cv2.imwrite(img_path, frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()
