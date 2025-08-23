## Image classification using Rust's `burn`-crate


https://github.com/user-attachments/assets/4f8fb0bc-8aeb-4235-a20e-76b192014883


### Goal
This project does not try to be the best possible solution for the problem of image classification. Instead I just wanted to try out a more dynamic approach to image classification while also improving my skills with rust and the `burn`-crate.
### Technique
Instead of getting the entire image, the model only has access to a dynamic window that is scaled down to a constant size. After each iteration step, the model outputs a tensor that corresponds to the choice of window for the next step. The accumulated information is stored via the state of an Lstm layer.
The classification process finishes once the maximum iteration count is reached or once the certainty of the model that it is a specific class is high enough.
### Efficiency
I am not doing a lot of optimization, the training is quite slow and not as parallelizable, I can't use many of `burn`'s default techniques like batching since it conflicts with the dynamic aspect of the model. Thus, this probably is one of the worst ways of doing image classification.
I still wanted to do this though as I am interested in whether this could work and which parts of an image the model would look at.
### Data
The model has support for two datasets, although the "smoking"-dataset has basically fully been abandoned as most of the modules now only support the second, the "Covid"-, dataset. Since the dataset isn't that big and the fact that it is unbalanced has caused some issues, I have decided to heavily augment it using some python tools.
### Training
The training uses the `TrainingManager` which does some adjustments over time. It combines the RL-approach that is used for optimizing the position-output and the supervised-learning the classification relies on. Essential for this working is the balancing of the two parts. This is because I wanted a high degree of interconnectivity, which is why I chose to only really separate the two parts of the model in the last layer. This means that any optimization for the output position may accidentally wreak havoc on the classification. I had an issue at the start, where the model would adapt the strategy of only guessing one possible class as different classes had different time-requirements and the classification optimizer would try to choose a category that had the best improvement over time (As that is one of the only metrics I have of whether the window-position is chosen adequately). To counteract that, I introduced a balancing parameter that would detect whether the model was simply prioritizing one class and would add a large penalty.
To keep the iteration count more or less constant during the training process (to avoid cases in which all the steps would max out the iterations or where no step would make it past the 4th iteration) I also added an adaptive concentration goal that raises the bar if the system detects that the current cutoff is being achieved easily.
