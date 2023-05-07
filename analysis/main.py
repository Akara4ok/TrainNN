import argparse
import pandas as pd
import matplotlib.pyplot as plt

def build_plot(filename):
    dataframe = pd.read_csv(filename)

    train_loss = dataframe[["epoch", "batch_no", "loss"]]
    train_loss.dropna(subset=['loss'], inplace=True)
    batch_count = int(train_loss['batch_no'].max() + 1)
    train_loss["abs_batch"] = train_loss["epoch"] * batch_count + train_loss["batch_no"]
    train_loss['loss_avg'] = train_loss['loss'].rolling(batch_count, min_periods=1).mean()

    val_loss = dataframe[["epoch", "val_loss"]]
    val_loss.dropna(subset=['val_loss'], inplace=True)
    val_loss["abs_batch"] = (val_loss["epoch"] + 1) * batch_count
    fig, ax = plt.subplots()

    ax.plot(train_loss['abs_batch'], train_loss['loss'], label='train_loss vs batch number')
    ax.plot(train_loss['abs_batch'], train_loss['loss_avg'], label='avg_train_loss vs batch number')
    ax.plot(val_loss['abs_batch'], val_loss['val_loss'], label='avg_train_loss vs batch number')

    ax.set_xlabel('batch')
    ax.set_ylabel('loss')
    ax.set_title('Dependency of loss on batch')
    ax.legend()

    plt.show()
    
if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    
    parser.add_argument("--source", "-s", help="file to analyze", type=str)
    
    args = vars(parser.parse_args())
    
    build_plot(args["source"])