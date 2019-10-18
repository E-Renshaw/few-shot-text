import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from flair.embeddings import WordEmbeddings
from tqdm import tqdm
import multiprocessing as mp
from preprocessing import read_images
from prototypicalNet import PrototypicalNet, train_step, test_step, load_weights

tqdm.pandas(desc="my bar!")


def main():
    embedding = WordEmbeddings("glove")
    sentence_1 = Sentence("") # add your sentences here
    sentence_2 = Sentence("")
    use_gpu = torch.cuda.is_available()
    num_episode = 16000
    frame_size = 1000
    trainx = trainx.permute(0, 3, 1, 2)
    testx = testx.permute(0, 3, 1, 2)

    # Initializing prototypical net
    protonet = PrototypicalNet(use_gpu)
    # Training loop
    frame_loss = 0
    frame_acc = 0
    for i in range(num_episode):
        loss, acc = train_step(protonet, trainx, trainy, 5, 60, 5)
        frame_loss += loss.data
        frame_acc += acc.data
        if (i + 1) % frame_size == 0:
            print(
                "Frame Number:",
                ((i + 1) // frame_size),
                "Frame Loss: ",
                frame_loss.data.cpu().numpy().tolist() / frame_size,
                "Frame Accuracy:",
                (frame_acc.data.cpu().numpy().tolist() * 100) / frame_size,
            )
            frame_loss = 0
            frame_acc = 0

    # Test loop
    num_test_episode = 2000
    avg_loss = 0
    avg_acc = 0
    for _ in range(num_test_episode):
        loss, acc = test_step(protonet, testx, testy, 5, 60, 15)
        avg_loss += loss.data
        avg_acc += acc.data
    print(
        "Avg Loss: ",
        avg_loss.data.cpu().numpy().tolist() / num_test_episode,
        "Avg Accuracy:",
        (avg_acc.data.cpu().numpy().tolist() * 100) / num_test_episode,
    )


if __name__ == "__main__":
    main()
