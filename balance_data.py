import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import cv2


file_name = 'training_data.npy'

training_data = np.load(file_name)

df = pd.DataFrame(training_data)
print(df.head(1))
print(Counter(df[1].apply(str)))

lefts = []
rights = []
forwards = []

shuffle(training_data)

for data in training_data:
    img = data[0]
    choice = data[1]

    if choice == [1, 0, 0]:
        lefts.append([img, choice])
    elif choice == [0, 0, 1]:
        rights.append([img, choice])
    elif choice == [0, 1, 0]:
        forwards.append([img, choice])
    else:
        print('error')

forwards = forwards[:len(rights)][:len(lefts)]
lefts = lefts[:len(forwards)]
rights = rights[:len(forwards)]

balanced_data = forwards + lefts + rights

shuffle(balanced_data)

print(len(balanced_data))

np.save('balanced_data.npy', balanced_data)


# for data in training_data:
#     img = data[0]
#     choice = data[1]

#     cv2.imshow('test', img)
#     print(choice)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         cv2.destroyAllWindows()
#         break