# TrainNN

1. sudo apt install libopencv-dev python3-opencv
2. pip install torchvision

3. In root of the project:
- python3 ./Data/mnist_download.py
- ./build.sh && ./build/TrainNN

4. testModelTraining() -- trains model, saves to Models/model.txt(now i see that it should binary file), writes training history to Logs/STANDARD/history.csv, append experiment info(batch_size, batch_count, epochs, params cout, training time) to metadata.csv
5. testModeInference() -- reads Models/model.txt, predict image and writes to console probabilities