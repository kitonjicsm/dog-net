# dog-net
`dog-net` is my convolutional neural network for classifying 120 different dog breeds. This is a five layer deep neural network, with two convolutional layers at the top and two fully connected (dense) layers at the bottom and it uses the `Keras API` with a `tensorflow` backend. 

Max pooling layers and batch normalization is used for efficiency. Dropout is used to reduce overfitting and bucketed padded batches are used to make the network flexible when it comes to image resolution. 

Instead of specifying a fixed resulotion for images that can be accepted, I decided to grab the largest image in a given batch using it as a pivot and just pad the other smaller ones with zeros. (Of course, it comes with compromises but I decided to try it out anyways)

The training is done using the `Adam` optimizer with an initial learning rate of `0.001` and using `categorical_crossentropy` as our loss function.

# Setup
1) Clone the repository.
2) Install Pipenv - `pip install pipenv`.
3) Setup the environment and install dependencies - `pipenv install`.
4) Activate the virtual environment and start a python shell - `pipenv shell`, `python`.
5) Import the model - `import src.dog_net as model`.

# How to Use
- To start training the model use `model.train_from_files()`. You can tweak addotional parameters like the `batch_size` or the number of `epochs` or you can just leave them at default. The model will be saved automatically after each iteration.
- To predict use `model.predict_from_file(path)` where the path is an relative path to the image file. (All major image formats are supported)
- To load a previously saved model use `model.load_model()`. This can be used in order to avoid 'pre-training' the model each and every time when you restart the shell.
