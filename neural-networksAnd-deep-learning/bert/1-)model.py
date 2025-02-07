"""
Problem Description:

We have a collection of SMS messages.
Some of these messages are spam and the rest are genuine.
Our task is to build a system that would automatically detect whether a message is spam or not.

"""
import time

import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.throughput_benchmark import format_time
from transformers import AutoModel, BertTokenizerFast
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from BertArch import BertArchNew
from transformers import AdamW
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import torch.nn as nn


if __name__ == "__main__":
    # choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """ hangi encodingte olduğunu anlamak için
    with open("data_sets/spam.csv", "rb") as f:
        result = chardet.detect(f.read(100000))  # İlk 100000 baytı oku
        print(result)"""

    # example data
    spam_data = [
        (0, "Hey, how are you doing today?"),
        (0, "Don't forget our meeting at 3 PM."),
        (0, "Can you send me the report by tomorrow?"),
        (0, "Let's grab lunch together this weekend."),
        (1, "Congratulations! You've won a free vacation. Call now!"),
        (1, "You have been selected fo  r a cash prize! Click the link to claim."),
        (1, "Urgent! Your bank account is at risk. Verify now."),
        (1, "Limited offer! Buy one, get one free!"),
        (0, "I'll call you back in 10 minutes."),
        (0, "Are you coming to the party tonight?"),
        (1, "Win an iPhone! Just answer this simple question."),
        (0, "Please let me know if you need any help."),
        (0, "Can you pick up some groceries on the way home?"),
        (1, "Get a loan approved instantly! No credit check required."),
        (1, "You've been pre-approved for a special deal!"),
        (1, "Your Netflix subscription has expired. Click here to renew."),
        (0, "Did you watch the new episode of that show?"),
        (0, "Meeting rescheduled to 5 PM."),
        (1, "Congratulations! You've won $1000! Click here to claim."),
        (1, "Hurry up! This offer expires in 24 hours!"),
    ]

    # read csv file
    # df = pd.read_csv("data_sets/spam.csv)

    # create data frame
    df = pd.DataFrame(data=spam_data, columns=["Label", "Text"])

    # learn info about dataset
    print(df.head())
    print(df.info())
    print(df.describe())

    # check class distirbution
    print(df['Label'].value_counts(normalize=True))

    """
        Train, Validation ve Test Setlerinin Anlamı
    Train Set (Eğitim Seti)
    
    Modelin öğrenmesi için kullanılan veridir.
    Modelin parametreleri bu veriyle eğitilir.
    Genellikle verinin %70-80'i eğitim için kullanılır.
    
    Validation Set (Doğrulama Seti)
    Modeli eğitirken, modelin performansını ölçmek için kullanılır.
    Modelin hata oranı (loss) ve doğruluğu (accuracy) bu veriyle değerlendirilir.
    Hyperparameter tuning (örneğin, öğrenme oranı veya katman sayısını ayarlamak) için kullanılır.
    Genellikle verinin %10-15'i doğrulama için kullanılır.
   
    Test Set (Test Seti)
    Eğitim tamamlandıktan sonra, modelin gerçek dünyada nasıl çalışacağını görmek için kullanılır.
    Modelin hiç görmediği veriler ile test edilir.
    Genellikle verinin %10-15'i test için kullanılır.
    """

    # split data into train, validation and test set

    train_X, temp_X, train_y, temp_y = train_test_split(df["Text"], df["Label"], random_state=42, test_size=0.3)

    valid_X, test_X, valid_y, test_y = train_test_split(temp_X, temp_y, random_state=42, test_size=0.3)

    print("bert model loading....")
    # import BERT-base pretrained model
    bert = AutoModel.from_pretrained("bert-base-uncased")
    print("bert model finished")

    print("bert tokenizer loading....")
    # Load Bert Tokenizer
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    print("bert tokenizer finished")

    # get length of all the messages in the train set
    seq_len = [len(i.split()) for i in train_X]

    pd.Series(seq_len).hist(bins=30)
    plt.show()

    # tokenize and encode sequences in the training set
    tokens_train = tokenizer.batch_encode_plus(
        train_X,
        max_length=25,
        pad_to_max_length=True,
        truncation=True
    )

    # tokenize and encode sequeneces in validation set
    tokens_valid = tokenizer.batch_encode_plus(
        valid_X,
        max_length=25,
        pad_to_max_length=True,
        truncation=True
    )

    # tokenize and encode sequeneces in test set
    tokens_test = tokenizer.batch_encode_plus(
        test_X,
        max_length=25,
        pad_to_max_length=True,
        truncation=True
    )

    ## convert lists to tensors
    train_seq = torch.tensor(tokens_train["input_ids"])
    train_mask = torch.tensor(tokens_train["attention_mask"])
    train_y = torch.tensor(train_y.tolist())

    valid_seq = torch.tensor(tokens_valid["input_ids"])
    valid_mask = torch.tensor(tokens_valid["attention_mask"])
    valid_y = torch.tensor(valid_y.tolist())

    test_seq = torch.tensor(tokens_test["input_ids"])
    test_mask = torch.tensor(tokens_test["attention_mask"])
    test_y = torch.tensor(test_y.tolist())


    # Data Loader

    # define a batch size
    batch_size = 32

    # wrap tensors
    train_data = TensorDataset(train_seq, train_mask, train_y)

    # sampler for sampling the data during training
    train_sampler = RandomSampler(train_data)

    # dataLoader for train set
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # wrap tensors
    val_data = TensorDataset(valid_seq, valid_mask, valid_y)

    # sampler for sampling the data during training
    val_sampler = SequentialSampler(val_data)

    # dataLoader for validation set
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)


    # freeze all the parameters
    for param in bert.parameters():
        param.requires_grad = False


    # pass the pre-trained BERT to our define architecture
    model = BertArchNew(bert)

    # push the model to GPU
    model = model.to(device)

    # define the optimizer
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # compute the class weights
    class_weights = compute_class_weight('balanced', np.unique(train_y), train_y)
    print("Class Weights:", class_weights)

    # converting list of class weights to a tensor
    weights = torch.tensor(class_weights, dtype=torch.float)

    # push to GPU
    weights = weights.to(device)

    # define the loss function
    cross_entropy = nn.NLLLoss(weight=weights)

    # number of training epochs
    epochs = 10

 # function to train the model
    def train():

        model.train()
        total_loss, total_accuracy = 0, 0

        # empty list to save model predictions
        total_preds = []

        # iterate over batches
        for step, batch in enumerate(train_dataloader):

            # progress update after every 50 batches.
            if step % 50 == 0 and not step == 0:
                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

            # push the batch to gpu
            batch = [r.to(device) for r in batch]

            sent_id, mask, labels = batch

            # clear previously calculated gradients
            model.zero_grad()

            # get model predictions for the current batch
            preds = model(sent_id, mask)

            # compute the loss between actual and predicted values
            loss = cross_entropy(preds, labels)

            # add on to the total loss
            total_loss = total_loss + loss.item()

            # backward pass to calculate the gradients
            loss.backward()

            # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # update parameters
            optimizer.step()

            # model predictions are stored on GPU. So, push it to CPU
            preds = preds.detach().cpu().numpy()

        # append the model predictions
        total_preds.append(preds)

        # compute the training loss of the epoch
        avg_loss = total_loss / len(train_dataloader)

        # predictions are in the form of (no. of batches, size of batch, no. of classes).
        # reshape the predictions in form of (number of samples, no. of classes)
        total_preds = np.concatenate(total_preds, axis=0)

        # returns the loss and predictions
        return avg_loss, total_preds


    # function for evaluating the model
    def evaluate(t0=None):

        print("\nEvaluating...")

        # deactivate dropout layers
        model.eval()

        total_loss, total_accuracy = 0, 0

        # empty list to save the model predictions
        total_preds = []

        # iterate over batches
        for step, batch in enumerate(val_dataloader):

            # Progress update every 50 batches.
            if step % 50 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))

            # push the batch to gpu
            batch = [t.to(device) for t in batch]

            sent_id, mask, labels = batch

            # deactivate autograd
            with torch.no_grad():

                # model predictions
                preds = model(sent_id, mask)

                # compute the validation loss between actual and predicted values
                loss = cross_entropy(preds, labels)

                total_loss = total_loss + loss.item()

                preds = preds.detach().cpu().numpy()

                total_preds.append(preds)

        # compute the validation loss of the epoch
        avg_loss = total_loss / len(val_dataloader)

        # reshape the predictions in form of (number of samples, no. of classes)
        total_preds = np.concatenate(total_preds, axis=0)

        return avg_loss, total_preds


    # set initial loss to infinite
    best_valid_loss = float('inf')

    # empty lists to store training and validation loss of each epoch
    train_losses = []
    valid_losses = []

    # for each epoch
    for epoch in range(epochs):

        print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))

        # train model
        train_loss, _ = train()

        # evaluate model
        valid_loss, _ = evaluate()

        # save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'saved_weights.pt')

        # append training and validation loss
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print(f'\nTraining Loss: {train_loss:.3f}')
        print(f'Validation Loss: {valid_loss:.3f}')

    # load weights of best model
    path = 'saved_weights.pt'
    model.load_state_dict(torch.load(path))

    # get predictions for test data
    with torch.no_grad():
        preds = model(test_seq.to(device), test_mask.to(device))
        preds = preds.detach().cpu().numpy()

    # model's performance
    preds = np.argmax(preds, axis=1)
    print(classification_report(test_y, preds))

