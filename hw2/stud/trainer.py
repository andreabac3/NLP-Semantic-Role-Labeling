#!/usr/bin/env python3

from torch import nn
from torch.utils.data import DataLoader
import torch
from torchtext.vocab import Vocab
from tqdm import tqdm

from stud.srl_utility import print_output_docker, evaluate_argument_identification, evaluate_argument_classification

'''
class trainer took inspiration from notebook 3 nlp sapienza group
'''


class Trainer():

    def __init__(
            self,
            model: nn.Module,
            loss_function,
            optimizer,
            label_vocab: Vocab,
            device: str,
            log_steps: int = 1_000,
            log_level: int = 2,
    ):

        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.device = device

        self.label_vocab = label_vocab
        self.log_steps = log_steps
        self.log_level = log_level

    def train(self, train_dataloader: DataLoader,
              dev_dataloader: DataLoader,
              epochs: int = 1, plot_loss: bool = False):
        if plot_loss:
            loss_summary_train = []
            loss_summary_valid = []
            f1_validation_summary_valid = []
            f1_classification_summary_valid = []
            epochs_list = list(range(epochs))

        train_loss = 0.0
        for epoch in range(epochs):
            # Save the model and optimizer state in order to resume the train phase at the right epochs.
            torch.save(self.model.state_dict(), "../../model/test_model_" + str(epoch) + ".pth")
            torch.save(self.optimizer.state_dict(), "../../model/test_optimizer_" + str(epoch) + ".pth")

            if plot_loss:
                # save at each epoch the loss and f1 into a object.
                print(loss_summary_train)
                print(loss_summary_valid)
                print(f1_validation_summary_valid)
                print(f1_classification_summary_valid)
                epochs_list_tmp = range(len(f1_classification_summary_valid))
                dict_loss = {"loss_train": loss_summary_train,
                             "loss_valid": loss_summary_valid,
                             "f1_valid_identification": f1_validation_summary_valid,
                             "f1_valid_classification": f1_classification_summary_valid,
                             "epochs_list": epochs_list_tmp
                             }
                print(dict_loss)
                torch.save(dict_loss, "../../model/DICT_LOSS" + str(epoch) + ".pth")

            if self.log_level > 0:
                print(' Epoch {:03d}'.format(epoch + 1))

            epoch_loss = 0.0
            self.model.train()

            for step, sample in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
                label: torch.LongTensor = sample['label']

                self.optimizer.zero_grad()

                output = self.model(sample)
                if self.model.use_crf:
                    mask = (label != self.label_vocab['<pad>'])
                    loss = self.model.crf(output, label, mask=mask) * -1
                else:
                    output = output.view(-1, output.shape[-1])
                    label = label.view(-1)
                    loss = self.loss_function(output, label)

                #  backpropagation step
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.tolist()

                if self.log_level > 1 and step % self.log_steps == self.log_steps - 1:
                    # print intermediate average train loss
                    mid_loss = epoch_loss / (step + 1)
                    print('\t[E: {:2d} @ step {}] current avg loss = {:0.4f}'.format(epoch, step, mid_loss))

            avg_epoch_loss = epoch_loss / len(train_dataloader)
            train_loss += avg_epoch_loss
            if plot_loss:
                loss_summary_train.append(avg_epoch_loss)
                print(loss_summary_train)
            if self.log_level > 0:
                print('\t[E: {:2d}] train loss = {:0.4f}'.format(epoch, avg_epoch_loss))  # print train loss at the end of the epoch

            if dev_dataloader is not None:
                if self.label_vocab is not None:
                    # at each epoch show the dev f1 score
                    result = print_output_docker(self.model, dev_dataloader, self.label_vocab)
                    identification_valid_result = evaluate_argument_identification(dev_dataloader.dataset.labels, result)
                    f1_validation_summary_valid.append(identification_valid_result['f1'])  # append the identification dev f1 score, in order to plot it then
                    print("DEV IDENTIFICATION: ", identification_valid_result)
                    classification_valid_result = evaluate_argument_classification(dev_dataloader.dataset.labels, result)
                    f1_classification_summary_valid.append(classification_valid_result['f1'])  # append the identification dev f1 score, in order to plot it then
                    print("DEV CLASSIFICATION: ", classification_valid_result)
                valid_loss = self.evaluate(dev_dataloader)

                if plot_loss:
                    loss_summary_valid.append(valid_loss)
                    print(loss_summary_valid)

                print('  [E: {:2d}] valid loss = {:0.4f}'.format(epoch, valid_loss))

        avg_epoch_loss = train_loss / epochs
        print(loss_summary_train)
        print(loss_summary_valid)
        print(f1_validation_summary_valid)
        print(f1_classification_summary_valid)
        dict_loss = {"loss_train": loss_summary_train,
                     "loss_valid": loss_summary_valid,
                     "f1_valid_identification": f1_validation_summary_valid,
                     "f1_valid_classification": f1_classification_summary_valid,
                     "epochs_list": epochs_list
                     }
        print(dict_loss)
        torch.save(dict_loss, "../../model/DICT_LOSS" + str(epoch) + ".pth")
        return (avg_epoch_loss, dict_loss)

    def evaluate(self, valid_dataset):
        valid_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for sample in valid_dataset:
                labels = sample['label']
                predictions = self.model(sample)
                if self.model.use_crf:
                    mask = (labels != self.label_vocab['<pad>'])
                    loss = -1 * self.model.crf(predictions, labels, mask=mask)
                else:
                    labels = labels.view(-1)
                    predictions = predictions.view(-1, predictions.shape[-1])
                    loss = self.loss_function(predictions, labels)
                valid_loss += loss.tolist()
        return valid_loss / len(valid_dataset)
