Download Link: https://assignmentchef.com/product/solved-homework-3-policy-gradient-cmu10703
<br>
<h1>Introduction</h1>

In this assignment, you will implement different RL algorithms and evaluate them on the OpenAI Gym LunarLander-v2 environment. This environment is considered solved if the agent can achieve an average score of at least 200.

You may need additional compute resources for this assignment. The extra credit will almost certainly require additional compute than a laptop CPU. Time estimates for training are given in the titles for each problem next to the point values. An AWS setup guide can be found at <a href="https://aws.amazon.com/ec2/getting-started/">https://aws.amazon.com/ec2/getting-started/</a><a href="https://aws.amazon.com/ec2/getting-started/">.</a> We recommend using p2-* or p3-* GPU instances.

This is a challenging assignment. <strong>Please start early!</strong>

<h2>Installation instructions (Linux)</h2>

We’ve provided Python packages that you may need in requirements.txt. To install these packages using pip and virtualenv, run the following commands:

apt-get install swig virtualenv env source env/bin/activate pip install -U -r requirements.txt

If your installation is successful, then you should be able to run the provided template code:

python reinforce.py python a2c.py

Note: You will need to install swig and box2d in order to install gym[box2d], which contains the LunarLander-v2 environment. You can install box2d by running pip install git+https://github.com/pybox2d/pybox2d

If you simply do pip install box2d, you may get an error because the pip package for box2d depends on an older version of swig.<a href="#_ftn2" name="_ftnref2"><sup>[2]</sup></a> For additional installation instructions, see <a href="https://github.com/openai/gym">https://github.com/openai/gym</a><a href="https://github.com/openai/gym">.</a>

<strong>Problem 0: Collaborators</strong>

Please list your name and Andrew ID, as well as those of your collaborators.

<h1>Problem 1: REINFORCE (30 pts)</h1>

In this section, you will implement episodic REINFORCE [6], a policy-gradient learning algorithm. Please write your code in reinforce.py; the template code provided inside is there to give you an idea on how you can structure your code, but is not mandatory to use.

Policy gradient methods directly optimize the policy <em>π</em>(<em>A </em>| <em>S,θ</em>), which is parameterized by <em>θ</em>. The REINFORCE algorithm proceeds as follows. We generate an episode by following policy <em>π</em>. After each episode ends, for each time step <em>t </em>during that episode, we update the policy parameters <em>θ </em>with the REINFORCE update. This update is proportional to the product of the return <em>G<sub>t </sub></em>experienced from time step <em>t </em>until the end of the episode and the gradient of log<em>π</em>(<em>A<sub>t </sub></em>| <em>S<sub>t</sub>,θ</em>). See Algorithm <strong>?? </strong>for details.

<strong>Algorithm 1 </strong>REINFORCE

1: <strong>procedure </strong>REINFORCE

2:         <em>Start with policy model π<sub>θ </sub></em>3:     <strong>repeat:</strong>

4:                                 <em>Generate an episode S</em><sub>0</sub><em>,A</em><sub>0</sub><em>,r</em><sub>0</sub><em>,…,S<sub>T</sub></em><sub>−1</sub><em>,A<sub>T</sub></em><sub>−1</sub><em>,r<sub>T</sub></em><sub>−1 </sub><em>following π<sub>θ</sub></em>(·)

5:                        <strong>for </strong><em>t from</em>

<sub>6:                            </sub>P <em>k t</em>

7:

8:                        <em>Optimize π<sub>θ </sub>using </em>∇<em>L</em>(<em>θ</em>)

9: <strong>end procedure</strong>

For the policy model <em>π</em>(<em>A </em>| <em>S,θ</em>), we recommend starting with a model that has:

<ul>

 <li>three fully connected layers with 16 units each, each followed by ReLU activations</li>

 <li>another fully connected layer with 4 units (the number of actions)</li>

 <li>a softmax activation (so the output is a proper distribution)</li>

</ul>

Initialize bias for each layer to zero. We recommend using a variance scaling kernel initializer

that draws samples from a uniform distribution over [−<em>α,α</em>] for <em>α </em>= <sup>p</sup>(3 ∗ scale<em>/n</em>) where scale = 1<em>.</em>0 and <em>n </em>is the average of the input and output units. HINT: Read the Keras documentation.

You can use the model.summary() and model.get config() calls to inspect the model architecture.

You can choose which optimizer and hyperparameters to use, so long as they work for learning on LunarLander-v2. We recommend using Adam as the optimizer. It will automatically adjust the learning rate based on the statistics of the gradients it’s observing. You can think of it like a fancier SGD with momentum. Keras provides a version of Adam <a href="https://keras.io/optimizers/">https: </a><a href="https://keras.io/optimizers/">//keras.io/optimizers/</a><a href="https://keras.io/optimizers/">.</a>

Train your implementation on the LunarLander-v2 environment until convergence<a href="#_ftn3" name="_ftnref3"><sup>[3]</sup></a>. Be sure to keep training your policy for at least 1000 more episodes after it reaches 200 reward so that you are sure it consistently achieves 200 reward and so that this convergence is reflected in your graphs. Then, answer the following questions.

<ol>

 <li><strong>[</strong>10 pts] Describe your implementation, including the optimizer and any hyperparameters you used (learning rate, <em>γ</em>, etc.). Your description should be detailed enough that someone could reproduce your results.</li>

 <li><strong>[</strong>20 pts] Plot the learning curve: Every <em>k </em>episodes, freeze the current cloned policy and run 100 test episodes, recording the mean and standard deviation of the cumulative reward. Plot the mean cumulative reward on the y-axis with the standard deviation as error-bars against the number of training episodes on the x-axis. Write a paragraph or two describing your graph(s) and the learning behavior you observed. Be sure to address the following questions:

  <ul>

   <li>What trends did you see in training?</li>

   <li>How does the final policy perform?</li>

   <li>The REINFORCE algorithm may be unstable. If you observe such instability in your implementation, what could be the reason?</li>

  </ul></li>

</ol>

Hint: You can use matplotlib’s plt.errorbar() function. <a href="https://matplotlib.org/api/_as_gen/matplotlib.pyplot.errorbar.html">https://matplotlib.org/ </a><a href="https://matplotlib.org/api/_as_gen/matplotlib.pyplot.errorbar.html">api/_as_gen/matplotlib.pyplot.errorbar.html</a>

<h1>Problem 2: Advantage-Actor Critic (40 pts)</h1>

In this section, you will implement N-step Advantage Actor Critic (A2C) [1]. Please write your code in a2c.py; the template code provided inside is there to give you an idea on how you can structure your code, but is not mandatory to use.

N-step A2C provides a balance between bootstraping using the value function and using the full Monte-Carlo return, using an N-step trace as the learning signal. See Algorithm <strong>?? </strong>for details. N-step A2C includes both REINFORCE with baseline (<em>N </em>= ∞) and the 1-step A2C covered in lecture (<em>N </em>= 1) as special cases and is therefore a more general algorithm.

The critic updates the state-value parameters <em>ω</em>, and the actor updates the policy parameters <em>θ </em>in the direction suggested by the N-step trace.

<strong>Algorithm 2 </strong>N-step Advantage Actor-Critic

<table width="624">

 <tbody>

  <tr>

   <td colspan="2" width="624">1: <strong>procedure </strong>N-Step Advantage Actor-Critic2:              <em>Start with policy model π<sub>θ </sub>and value model V<sub>ω</sub></em></td>

  </tr>

  <tr>

   <td width="50">3:</td>

   <td width="574"><strong>repeat:</strong></td>

  </tr>

  <tr>

   <td width="50">4:</td>

   <td width="574"><em>Generate an episode S</em><sub>0</sub><em>,A</em><sub>0</sub><em>,r</em><sub>0</sub><em>,…,S<sub>T</sub></em><sub>−1</sub><em>,A<sub>T</sub></em><sub>−1</sub><em>,r<sub>T</sub></em><sub>−1 </sub><em>following π<sub>θ</sub></em>(·)</td>

  </tr>

  <tr>

   <td width="50">5:</td>

   <td width="574"><strong>for </strong><em>t from T </em>− 1 <em>to </em>0:</td>

  </tr>

  <tr>

   <td width="50">6:</td>

   <td width="574"><em>V<sub>end </sub></em>= 0 if (<em>t </em>+ <em>N </em>≥ <em>T</em>) <em>else V<sub>ω</sub></em>(<em>s<sub>t</sub></em><sub>+<em>N</em></sub>)</td>

  </tr>

  <tr>

   <td width="50">7:</td>

   <td width="574"><em> else </em>0)</td>

  </tr>

  <tr>

   <td width="50">8:9: 10:</td>

   <td width="574"><em>Optimize π<sub>θ </sub>using </em>∇<em>L</em>(<em>θ</em>)</td>

  </tr>

  <tr>

   <td colspan="2" width="624">11:                      <em>Optimize V<sub>ω </sub>using </em>∇<em>L</em>(<em>ω</em>)12: <strong>end procedure</strong></td>

  </tr>

 </tbody>

</table>

Start off with the same policy architecture described in Problem 1 for both the actor and the critic. Play around with the network architecture of the critic’s state-value approximator to find one that works for LunarLander-v2. Once again, you can choose which optimizer and hyperparameters to use, so long as they work for learning on LunarLander-v2.

Answer the following questions:

<ol>

 <li><strong>[</strong>10 pts] Describe your implementation, including the optimizer, the critic’s network architecture, and any hyperparameters you used (learning rate, <em>γ</em>, etc.).</li>

 <li><strong>[</strong>20 pts] Train your implementation on the LunarLander-v2 environment several times with N varying as [1, 20, 50, 100] (it’s alright if the N=1 case is hard to get working). Plot the learning curves for each setting of N in the same fashion as Problem 1. You may find that your plots with error bars will be too busy to plot all values of N on the same graph. If this is the case, make a different plot for each value of N. Once again, write a paragraph or two describing your graph(s) and the learning behavior you observed. Be sure to address the following questions:

  <ul>

   <li>What trends did you observe in training?</li>

   <li>How does the final policy perform?</li>

   <li>If you found A2C to be unstable or otherwise difficult to train, why might this be the case? What about the algorithm formulation could cause training instability, and what improvements might be made to improve it?</li>

  </ul></li>

 <li><strong>[</strong>10 pts] Discuss how the performance of your implementation of A2C compares with REINFORCE and how A2C’s performance varies with N. Which algorithm and N setting learns faster, and why do you think this is the case?</li>

</ol>

<h1>Extra credit (up to 15 pts)</h1>

A major bottleneck in training policy gradient algorithms is that only one episode (or batch of states) is generated at a time. However, once the policy has been updated once, the training data is no longer drawn from the current policy distribution, becoming “invalid” in a sense. A similar challenge occurs when parallelizing training, since once a parameter update is performed by one worker, the policy distribution changes and invalidates the data gathered and gradients computed by the other workers. Mnih <em>et al. </em>argue that the exploration noise from asynchronous policy updates can be beneficial to learning [2].

First, let’s introduce a more complex environment. Many deep reinforcement learning papers (at least in the past few years) have used Atari games as performance benchmarks due to their greater complexity. Apply your implementation of A2C to any of the OpenAI gym Breakout environments. We recommend either Breakout-v0 or BreakoutNoFrameskip-v4 environments. You will need to use a larger, more complex policy network than the one you used in Problem 1 and 2, as well as some tricks like learning rate decay. Think carefully about your hyperparameters, particularly <em>N</em>. You should be able to reach at least 200 average reward after 10-20 hours of training on AWS; note that running these large networks on a laptop may take up to two weeks.

Then, implement multi-threaded synchronous Advantage Actor-Critic by gathering episode rollouts in parallel and performing a single gradient update. What speedup can you achieve? How might you measure this? Then, implement Asynchronous Advantage Actor-Critic (A3C) with multiple threads, using your multi-threaded synchronous Advantage Actor-Critic as a starting point. Do you see a learning speedup or increased stability compared to a synchronous implementation?

Up to 15 points extra credit will be awarded total, contingent on implementation, results, and analysis. Describe how you implemented the task and provide metrics and graphs showing improvement as well as explanations as to why that might be the case. You may also wish to include links to videos of your trained policies. If nothing else, it is entertaining and rewarding to see an agent you trained play Breakout at a superhuman level.

<h1>Guidelines on implementation</h1>

This homework requires a significant implementation effort. It is hard to read through the papers once and know immediately what you will need to be implement. We suggest you to think about the different components (e.g., model definition, model updater, model runner, …) that you will need to implement for each of the different methods that we ask you about, and then read through the papers having these components in mind. By this we mean that you should try to divide and implement small components with well-defined functionalities rather than try to implement everything at once. Much of the code and experimental setup is shared between the different methods so identifying well-defined reusable components will save you trouble.

Some hyperparameter and implementation tips and tricks:

<ul>

 <li>For efficiency, you should try to vectorize your code as much as possible and use <strong>as few loops as you can </strong>in your code. In particular, in lines 5 and 6 of Algorithm 1 (REINFORCE) and lines 5 to 7 of Algorithm 2 (A2C) you should not use two nested loops. How can you formulate a single loop to calculate the cumulative discounted rewards? Hint: Think backwards!</li>

 <li>Moreover, it is likely that it will take between 10K and 50K episodes for your model to converge, though you should see improvements within 5K episodes (about 30 minutes to one hour). On a NVIDIA GeForce GTX 1080 Ti GPU, it takes about five hours to run 50K training episodes with our REINFORCE implementation.</li>

 <li>For A2C, downscale the rewards by a factor of 1e-2 (i.e., divide by 100) when training (but not when plotting the learning curve) This will help with the optimization since the initial weights of the critic are far away from being able to predict a large range such as [−200<em>,</em>200]. You are welcome to try downscaling the rewards of REINFORCE as well.</li>

 <li>Normalizing the returns <em>G<sub>t </sub></em>over each episode by subtracting the mean and dividing by the standard deviation may improve the performance of REINFORCE.</li>

 <li>Likewise, batch normalization between layers can improve stability and convergence rate of both REINFORCE and A2C. Keras has a built-in batch normalization layer <a href="https://keras.io/layers/normalization/">https://keras.io/layers/normalization/</a><a href="https://keras.io/layers/normalization/">.</a></li>

 <li>Feel free to experiment with different policy architectures. Increasing the number of hidden units in earlier layers may improve performance.</li>

 <li>We recommend using a discount factor of <em>γ </em>= 0<em>.</em></li>

 <li>Try out different learning rates. A good place to start is in the range [1e-5<em>,</em>1e-3]. Also, you may find that varying the actor and critic learning rates for A2C can help performance. There is no reason that the actor and critic must have the same learning rate.</li>

 <li>Policy gradient algorithms can be fairly noisy. You may have to run your code for several tens of thousand training episodes to see a consistent improvement for REINFORCE and A2C.</li>

 <li>Instead of training one episode at a time, you can try generating a fixed number of steps in the environment, possibly encompassing several episodes, and training on such a batch instead.</li>

</ul>

<h1>References</h1>

<ul>

 <li>Vijay R. Konda and John N. Tsitsiklis. Actor-critic algorithms. In S. A. Solla, T. K. Leen, and K. Mu¨ller, editors, <em>Advances in Neural Information Processing Systems 12</em>, pages 1008–1014. MIT Press, 2000.</li>

 <li>Volodymyr Mnih, Adria` Puigdom`enech Badia, Mehdi Mirza, Alex Graves, Timothy P. Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. Asynchronous methods for deep reinforcement learning. <em>CoRR</em>, abs/1602.01783, 2016.</li>

 <li>John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, and Pieter Abbeel. Trust region policy optimization. <em>CoRR</em>, abs/1502.05477, 2015.</li>

 <li>John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. High-Dimensional Continuous Control Using Generalized Advantage Estimation. <em>arXiv e-prints</em>, page arXiv:1506.02438, Jun 2015.</li>

 <li>John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. <em>CoRR</em>, abs/1707.06347, 2017.</li>

 <li>Ronald J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. In <em>Machine Learning</em>, pages 229–256, 1992.</li>

</ul>

<a href="#_ftnref1" name="_ftn1">[1]</a> <a href="https://www.cmu.edu/policies/">https://www.cmu.edu/policies/</a>

<a href="#_ftnref2" name="_ftn2">[2]</a> <a href="https://github.com/openai/gym/issues/100">https://github.com/openai/gym/issues/100</a>

<a href="#_ftnref3" name="_ftn3">[3]</a> LunarLander-v2 is considered solved if your implementation can attain an average score of at least 200.