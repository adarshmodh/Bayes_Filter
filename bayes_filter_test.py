import numpy as np
import matplotlib.pyplot as plt
import random


if __name__ == "__main__":

    # Load the data
    data = np.load(open('data/starter.npz', 'rb'))
    cmap = data['arr_0']
    actions = data['arr_1']
    observations = data['arr_2']
    belief_states_given = data['arr_3']

    print(belief_states_given)

    init_states = np.empty(cmap.shape)
    init_states.fill(1.0/(cmap.size))

    for iteration in range(0,observations.size-1):
	    belief_states = np.zeros(cmap.shape)
	    belief_states_pred = np.zeros(cmap.shape)

	    for ix,iy in np.ndindex(init_states.shape):
	    	temp = np.zeros(init_states.shape)
	    	
	    	if((ix - actions[iteration][1]>=0 and ix - actions[iteration][1]<20) and (iy + actions[iteration][0]>=0 and iy + actions[iteration][0]<20)):
	    		temp[ix - actions[iteration][1]][iy + actions[iteration][0]] = 0.9*init_states[ix][iy]
	    		temp[ix][iy] = 0.1*init_states[ix][iy]
	    	else: 
	    		temp[ix][iy] = 1.0*init_states[ix][iy]

	    	belief_states = belief_states + temp

	    	if(cmap[ix][iy] == observations[iteration]):
	    		belief_states_pred[ix][iy] = belief_states[ix][iy]*0.9
	    	else:
	    		belief_states_pred[ix][iy] = belief_states[ix][iy]*0.1

	    # print belief_states_pred
	    # print np.sum(belief_states)
	    
	    mle_tup =  np.unravel_index(np.argmax(belief_states_pred, axis=None), belief_states_pred.shape)
	    mle = np.empty([2])
	    mle[0] = mle_tup[1]
	    mle[1] = 19 - mle_tup[0]
	    print(mle)
	    init_states = belief_states_pred