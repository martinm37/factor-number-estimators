
import numpy as np



vector = np.arange(60).reshape(-1,1)

matrix = np.concatenate([vector,vector,vector],axis = 1)



training_sample_width = 10

out_of_sample_width = 3


daily_returns = np.zeros(len(matrix) - training_sample_width)

#num_iter = 3
num_iter = int(np.floor((len(matrix) - training_sample_width) / out_of_sample_width))


for t in range(num_iter):

    whole_sample = matrix[(t * out_of_sample_width) : (training_sample_width + out_of_sample_width) + (t * out_of_sample_width), :]

    training_sample = whole_sample[0:training_sample_width, :]

    out_of_sample = whole_sample[training_sample_width: (training_sample_width + out_of_sample_width), :]

    for i in range(out_of_sample_width):
        daily_returns[(t*out_of_sample_width) + i] = i



whole_sample = matrix[0: (training_sample_width + out_of_sample_width) , :]



training_sample  =  whole_sample[0:training_sample_width , :]

out_of_sample = whole_sample[ training_sample_width : (training_sample_width + out_of_sample_width) , :]






print("hello there xddd")


