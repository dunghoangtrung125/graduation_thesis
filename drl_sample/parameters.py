nu = 0.1 # probability that the jammer is idle
arrival_rate = 3 # mean rate of the data arrival process
nu_p = [0.6, 0.2, 0.2] # Probability that the jammer attacks at power levels 1, 2, 3
d_t = 4 # number of packages per active transmission
e_t = 1 # number of energy unit required for actively transmitting a package
d_bj_arr = [1, 2, 3] # the number of packages that can be backscattered corresponding to jamming power levels 1, 2, 3
e_hj_arr = [1, 2, 3] # the number of energy units that can be harvested corresponding to jamming power levels 1, 2, 3
dt_ra_arr = [2, 1, 0] # the number of packets that can be transmitted by using the RA technique corresponding to jamming power levels 1, 2, and 3
d_queue_size = 10 # capacity of data queue
e_queue_size = 10 # capacity of energy queue
b_dagger = 3 # fix number packages that can be backscattered
num_actions = 7 # number of actions in the action space
num_states = 2 * (d_queue_size + 1) * (e_queue_size + 1)

learning_rate_Q = 0.1
gamma_Q = 0.9 # discount factor
learning_rate_deepQ = 0.0001
gamma_deepQ = 0.99 # discount factor

num_features = 3 # number of features in state space
memory_size = 10000
batch_size = 16
update_target_network = 5000
step = 1000 # print reward after every step
T = 10_000 # number of training iterations