#!/usr/bin/env python

import os, sys, h5py, argparse
from tqdm import tqdm
import numpy as np
import torch as th
import torch.nn as nn
import matplotlib.pyplot as plt

from scripts.evaluation import evaluation_model


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, help = 'Path containing data generated with ACTS (with hits.root and measurements.root files)')
    parser.add_argument('--out_dir', type=str, help = 'Path to store processed data')
    parser.add_argument('-b', '--batch_size', type=int, default=1000, help = 'Batch size for training')
    parser.add_argument('-o', '--overwrite', action='store_true', help = 'Overwrite existing output file')
    parser.add_argument('-c', '--ckpt', type=str, default="", help = 'Path to checkpoint file to resume training')
    parser.add_argument('-e', '--epochs', type=int, default=1, help = 'Number of epochs to train for')
    args = parser.parse_args()

    freeze_linear = False
    freeze_transformer = False
    class_threshold = 0.5

    #---------------------------- IMPORT DATA ----------------------------

    tracks_file = h5py.File(args.in_dir+"tracks.hdf5","r")

    labels = tracks_file["tracks"]["truth"]["NN_label"]
    inputs = tracks_file["tracks"]["measurements"]

    ntracks, nhits = labels.shape
    ninputs = len(list(inputs.dtype.names))
    train_idx = int(0.8*ntracks)
    val_idx = int(0.9*ntracks)

    # TODO: Do dataset split ahead of time
    train_mask = np.arange(ntracks) < 0.8*ntracks
    val_mask = np.logical_and(np.arange(ntracks) >= 0.8*ntracks, np.arange(ntracks) < 0.9*ntracks)
    test_mask = np.arange(ntracks) >= 0.9*ntracks
    train_dataset = evaluation_model.TracksDataset(inputs, labels, train_mask)
    val_dataset = evaluation_model.TracksDataset(inputs, labels, val_mask)
    test_dataset = evaluation_model.TracksDataset(inputs, labels, test_mask)

    # TODO: Add masks for padded elements
    train_loader = th.utils.data.DataLoader(train_dataset, shuffle=False, batch_size=args.batch_size)
    val_loader = th.utils.data.DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size)
    test_loader = th.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size)

    #---------------------------- SET UP NETWORK ----------------------------
    
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    if not os.path.exists(args.out_dir+"/ckpts"):
        os.makedirs(args.out_dir+"/ckpts")

    if th.cuda.is_available():
        device = th.device('cuda')
        print("Found {} GPUs".format(th.cuda.device_count()))
    else:
        device = th.device('cpu')
        print("No GPUs found, using CPU")

    model = evaluation_model.EvalNN(ninputs, 128, 2, 4, False, 0.1, nhits).double().to(device)
    optimizer = th.optim.Adam(model.parameters(), lr=0.001)
    loss = nn.BCELoss()

    #load existing checkpoint
    if args.ckpt and os.path.exists(args.ckpt):
        checkpoint = th.load(args.ckpt, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']+1
        print("Loading previous model. Starting from epoch {}.".format(start_epoch), flush=True)
    else:
        start_epoch = 1

    #print model parameters
    print("Model built. Parameters:", flush=True)
    for name, param in model.named_parameters():
        param.requires_grad = True
        if freeze_linear and "linear" in name:
            param.requires_grad = False
        if freeze_transformer and "transformer" in name:
            param.requires_grad = False
        print(name, param.size(), param.requires_grad, flush=True)
    print("", flush=True)

    if start_epoch == args.epochs+1:
        print("Model already trained. Skipping training")

    #---------------------------- TRAIN NETWORK ----------------------------

    train_loss_array = np.zeros(args.epochs)
    val_loss_array = np.zeros(args.epochs)

    for epoch in range(start_epoch,args.epochs+1):
        print("Epoch: {}".format(epoch), flush=True)

        for data in tqdm(train_loader):
            batch, train_labels = data
            batch = batch.to(device)
            train_labels = train_labels.to(device)

            pred = model(batch)
            pred_lt = loss(pred, train_labels)
            train_loss_array[epoch-1] += pred_lt.item()*batch.shape[0]

            optimizer.zero_grad()
            pred_lt.backward()
            optimizer.step()
        
        train_loss_array[epoch-1] = train_loss_array[epoch-1]/(train_idx+1)
        print("Training loss: {}".format(train_loss_array[epoch-1]), flush=True)

        model.eval()
        for data in val_loader:
            batch, val_labels = data
            batch = batch.to(device)
            val_labels = val_labels.to(device)

            pred = model(batch)
            pred_lt = loss(pred, val_labels)
            val_loss_array[epoch-1] += pred_lt.item()*batch.shape[0]
            
        val_loss_array[epoch-1] = val_loss_array[epoch-1]/(val_idx-train_idx+1)
        print("Validation loss: {}".format(val_loss_array[epoch-1]), flush=True)
        print("--------------------------------------------")

        #save checkpoint
        th.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, args.out_dir+"/ckpts/eval_"+str(epoch)+"_val_"+str(val_loss_array[epoch-1])+".ckpt") 

    #---------------------------- EVALUATE NETWORK ----------------------------

    ntest = len(test_dataset)
    test_loss = track_index = 0
    test_results = np.zeros((ntest, nhits, 2))
    with th.no_grad():
        model.eval()
        for data in test_loader:
            batch, test_labels = data
            batch = batch.to(device)
            test_labels = test_labels.to(device)

            pred = model(batch)
            pred_lt = loss(pred, test_labels)
            test_loss += pred_lt.item()*batch.shape[0]

            test_labels = test_labels.cpu()
            batch = batch.cpu()
            pred = pred.cpu()

            test_labels[batch[:,:,0] == 0] = -1 #mark empty hits as -1

            test_results[track_index:track_index+batch.shape[0],:,0] = pred.squeeze(-1)
            test_results[track_index:track_index+batch.shape[0],:,1] = test_labels.squeeze(-1)

            track_index += batch.shape[0]

    plt.figure(figsize=(8,6))
    plt.hist(test_results[:,:,0], bins=20)
    plt.xlabel("Prediction")
    plt.title("Distribution of test set predictions")
    plt.savefig(args.out_dir+"/test_pred.png")

    test_results[:,:,0] = test_results[:,:,0] > class_threshold #round predictions based on chosen threshold

    predicted = test_results[:,:,0].flatten()
    actual = test_results[:,:,1].flatten()
    predicted = predicted[actual >= 0]
    actual = actual[actual >= 0]
    fpr = np.sum(1 - predicted[actual < 0.5]) / np.sum(actual < 0.5)
    fnr = np.sum(predicted[actual > 0.5]) / np.sum(actual > 0.5)
    print(fpr, fnr, np.sum(actual < 0.5) / np.sum(actual > 0.5))
    print("FPR:", fpr, " FNR:", fnr)
    print("Accuracy:", np.sum(test_results[:,:,0] == test_results[:,:,1]) / np.sum(test_results[:,:,1] > -0.5))

    test_loss = test_loss/ntest
    print("--------------------------------------------")
    print("--------------------------------------------")
    print("Test loss: {}".format(test_loss), flush=True)

    eval_array = np.zeros((ntest, 2))
    for itrack in range(ntest):
        track_len = np.sum(test_results[itrack,:,1] != -1)
        correct = np.sum(test_results[itrack,:,0] == test_results[itrack,:,1])
        eval_array[itrack] = [track_len, correct]

    if not os.path.exists(args.out_dir+"/plots"):
        os.makedirs(args.out_dir+"/plots")

    #plot number of correct hits vs number of total hits
    bin_edges = np.arange(-0.5,nhits+1.5,1)
    fig1 = plt.figure()
    plt.hist2d(eval_array[:,0], eval_array[:,1], bins=[bin_edges, bin_edges])
    #plt.scatter(eval_array[:,0], eval_array[:,1])
    plt.xlabel("Total hits")
    plt.ylabel("Correct hits")
    plt.savefig(args.out_dir+"/plots/correct_hits.png")

    #plot loss
    fig2 = plt.figure()
    plt.ioff()
    plt.plot(range(args.epochs), train_loss_array, label="Training")
    plt.plot(range(args.epochs), val_loss_array, label="Validation")
    plt.legend()
    plt.xlabel("Epoch")
    plt.savefig(args.out_dir+"/plots/lossplot.png")

    tracks_file.close()


if __name__ == '__main__':
    main(sys.argv)
