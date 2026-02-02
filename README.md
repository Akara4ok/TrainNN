# TrainNN

1. sudo apt install libopencv-dev python3-opencv
2. pip3 install torchvision

3. In the project root:
- python3 ./Data/mnist_download.py
- ./build.sh && ./build/TrainNN

4. testModelTraining() -- trains the model, saves it to Models/model.txt(now i see that it should be a binary file), writes training history to Logs/STANDARD/history.csv, append experiment info(batch_size, batch_count, epochs, params count, training time) to metadata.csv
5. testModelInference() -- reads Models/model.txt, predicts image and prints the probabilities to the console