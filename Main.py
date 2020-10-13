from Adaline import *
from Generator import *
from Testowanie import *

test()
# training_set = generate(10)
# labels=[]
# sets=[]
# for i in range(len(training_set)):
#     set = training_set[i]
#     labels.append(set[0])
#     sets.append(set[1:])
# adaline = Adaline(2, 0.02,1000, -1, 1)
# adaline.learn(sets,labels, 0.25)
# print(adaline.predict([-1, -1]))
# print(adaline.predict([-1, 1]))
# print(adaline.predict([1, -1]))
# print(adaline.predict([1, 1]))