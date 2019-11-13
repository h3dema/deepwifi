# What's is implemented


This directory contains classes for implementing the following rl methods:
* `deepQL.py`: implements Deep Q-learning with memory replay using only one network
* `dql.py`: implements Deep Q-learning with memory replay and target network (to obtain Q-max)
* `ddql.py`: implements Double Deep Q-learning with memory replay and target network. Uses two networks when predicting the Q-max value.

The classes of these modules interact with the __env__ class defined in the __Environment__ directory of this repository. Objects in this class implement (and return) states, actions, and rewards.




```
from DQL.ddql import DDQL

(...)


agent = DDQL(env=env,
             model=model,
             memory=memory,
             timesteps=args.timesteps)
```


# Running

We can call directly the Deep RL modules. They run a test using a dummy environment (grid world).
Thus call

```
python3 deepQL.py
```

or

```
python3 ddql.py
```


# Helper modules

`replay.py` implements the memory class used in the memory replay procedure. This memory is implemented as a finite (and fixed) circular list where the new element occupies an empty position or overlaps the oldest element (if the list is full).