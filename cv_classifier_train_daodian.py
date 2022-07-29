import sys
import os
import torch
import torch.nn as nn
import timm
import urllib
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torchvision.transforms as transforms
import torchvision
from cv_classifier import CvClassifier
from timm.data import ImageDataset, IterableImageDataset, AugMixDataset, create_loader, create_dataset
import timm.optim
import timm.scheduler
from tqdm.auto import tqdm
import torchmetrics
from torch.utils.tensorboard import SummaryWriter
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import timm
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from torch.optim import Adam, lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
import albumentations
from albumentations.pytorch.transforms import ToTensorV2
from cv_dataset import CvDataset
import pandas as pd
from sklearn import metrics
from datetime import date
from sklearn.metrics import f1_score, accuracy_score

MODEL_PATH = '/home/ma-user/work/MultimodalSimilar/cv_model'

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class CFG:
    seed = 123
    img_size = 512
    classes = 11014
    fc_dim = 512
    epochs = 100
    batch_size = 24
    num_workers = 3
    model_name = 'tf_efficientnet_b4'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scheduler_params = {
        "lr_start": 1e-3,
        "lr_max": 1e-5 * batch_size,
        "lr_min": 1e-6,
        "lr_ramp_ep": 5,
        "lr_sus_ep": 0,
        "lr_decay": 0.8,
    }


def getAugmentation(IMG_SIZE, isTraining=False):
    if isTraining:
        return albumentations.Compose([
            albumentations.Resize(IMG_SIZE, IMG_SIZE, always_apply=True),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.Rotate(limit=120, p=0.75),
            albumentations.RandomBrightness(limit=(0.09, 0.6), p=0.5),
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(p=1.0)
        ])
    else:
        return albumentations.Compose([
            albumentations.Resize(IMG_SIZE, IMG_SIZE, always_apply=True),
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(p=1.0)
        ])

import numpy as np
def get_class_weights(data):
    weight_dict=dict()
    # Format of row : PostingId, Image, ImageHash, Title, LabelGroup
    # LabelGroup index is 4 and it is representating class information
    for i, row in data.iterrows():
        weight_dict[row['tag_new_id']]=0
    # Word dictionary keys will be label and value will be frequency of label in dataset
    for i, row in data.iterrows():
        weight_dict[row['tag_new_id']]+=1
    # for each data point get label count data
    class_sample_count= np.array([weight_dict[row['tag_new_id']] for i, row in data.iterrows()])
    # each data point weight will be inverse of frequency
    weight = 1. / class_sample_count
    weight=torch.from_numpy(weight)
    return weight


def training_one_epoch(epoch_num, model, dataloader, optimizer, scheduler, device, loss_criteria):
    avgloss = 0.0
    # put model in traning model
    model.train()
    tq = tqdm(enumerate(dataloader), total=len(dataloader))
    train_accuracy = torchmetrics.Accuracy()
    train_accuracy.to(device)
    for idx, data in tq:
        batch_size = data[0].shape[0]
        images = data[0]
        targets = data[1]
        # zero out gradient
        optimizer.zero_grad()
        # put input and target to device
        images = images.to(device)
        targets = targets.to(device)
        # pass input to the model
        output = model(images, targets)
        # get loss
        loss = loss_criteria(output, targets)
        # backpropogation
        loss.backward()
        # update learning rate step
        optimizer.step()
        # avg loss
        avgloss += loss.item()

        tq.set_postfix({'loss': '%.6f' % float(avgloss / (idx + 1)), 'LR': optimizer.param_groups[0]['lr']})

        predicted_label = torch.argmax(output, 1)
        train_accuracy(predicted_label, targets)
    # lr scheduler step after each epoch
    scheduler.step()
    acc_score_metric = train_accuracy.compute().detach().cpu().numpy()
    return avgloss / len(dataloader), acc_score_metric


def validation_one_epoch(model, dataloader, epoch, device, loss_criteria):
    avgloss = 0.0
    # put model in traning model
    model.eval()
    tq = tqdm(enumerate(dataloader), desc="Training Epoch { }" + str(epoch + 1))

    test_accuracy = torchmetrics.Accuracy()
    test_accuracy.to(device)

    y_true = []
    y_pred = []
    with torch.no_grad():
        for idx, data in tq:
            batch_size = data[0].shape[0]
            images = data[0]
            targets = data[1]

            images = images.to(device)
            targets = targets.to(device)
            output = model(images, targets, is_test=True)
            predicted_label = torch.argmax(output, 1)
            y_true.extend(targets.detach().cpu().numpy())
            y_pred.extend(predicted_label.detach().cpu().numpy())
            loss = loss_criteria(output, targets)

            avgloss += loss.item()
            test_accuracy(predicted_label, targets)
            tq.set_postfix({'validation loss': '%.6f' % float(avgloss / (idx + 1))})
    f1_score_metric = f1_score(y_true, y_pred, average='micro')
    acc_score_metric = test_accuracy.compute().detach().cpu().numpy()
    tq.set_postfix({'validation f1 score': '%.6f' % float(f1_score_metric)})
    return avgloss / len(dataloader), f1_score_metric, acc_score_metric

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

if __name__ == '__main__':
    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter()
    #定义模型
    os.system("mkdir -p /home/ma-user/.cache/torch/hub/checkpoints/")
    os.system("cp ../efficientnet_b4_ra2_320-7eb33cd5.pth /home/ma-user/.cache/torch/hub/checkpoints/")
    #pretrained_model = timm.create_model('nfnet_l0', pretrained=True)
    #pretrained_model = timm.create_model('efficientnet_b4', pretrained=True)
    model = CvClassifier('efficientnet_b4', fc_dim=CFG.fc_dim, num_labels=4181)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    #model = torch.load('./cv_model/44000.pt.bak')
    model.to(device)

    #加载数据
    # training augmentation
    train_aug = getAugmentation(CFG.img_size,isTraining=True )
    validation_aug = getAugmentation(CFG.img_size, isTraining=False)
    #
    # config = resolve_data_config({}, model=pretrained_model)
    # transform_eff = create_transform(**config)

    # train_dateset = create_dataset(name='', root='./goodssku_image_train', transform=transform_eff, is_training=True)
    # test_dateset = create_dataset(name='', root='./goodssku_image_test', transform=transform_eff, is_training=False)
    #

    train_df = pd.read_csv('./train_cv_data.csv')
    test_df = pd.read_csv('./test_cv_data.csv')
    # get weights for  classes
    samples_weight = get_class_weights(train_df)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, num_samples=len(samples_weight))
    print(samples_weight, flush=True)

    trainset = CvDataset(train_df, '/home/ma-user/work/MultimodalSimilar/goodssku_image', train_aug)
    validset = CvDataset(test_df, '/home/ma-user/work/MultimodalSimilar/goodssku_image', validation_aug)



    # training_dataloader = create_loader(
    #     train_dateset,
    #     input_size=(3, 224, 224),
    #     batch_size=112,
    #     is_training=True,
    #     num_workers=12
    # )
    # testing_dataloader = create_loader(
    #     test_dateset,
    #     input_size=(3, 224, 224),
    #     batch_size=112,
    #     is_training=True,
    #     num_workers=12
    # )
    train_dataloader = DataLoader(trainset, batch_size=CFG.batch_size,
                                  drop_last=True, pin_memory=True, sampler=sampler, collate_fn=collate_fn,
                                  num_workers=10)

    validation_dataloader = DataLoader(validset, batch_size=CFG.batch_size,
                                       drop_last=True, pin_memory=True,
                                       collate_fn=collate_fn,
                                       num_workers=10)

    # train loop
    training_epochs = 300
    cooldown_epochs = 10
    num_epochs = training_epochs + cooldown_epochs

    # optimizer_classifer = timm.optim.AdamP(model.classifier.parameters(), lr=0.1)
    # scheduler_classifer = timm.scheduler.CosineLRScheduler(optimizer_classifer, t_initial=training_epochs, warmup_t=5, warmup_lr_init=0.1)
    # optimizer_embedder = timm.optim.AdamP(model.emb_layer.parameters(), lr=0.0001, weight_decay=0.00001)
    # scheduler_embedder = timm.scheduler.CosineLRScheduler(optimizer_embedder, t_initial=training_epochs, warmup_t=5, warmup_lr_init=0.00001)

    # optimizer_classifer = timm.optim.AdamP(model.classifier.parameters(), lr=0.1)
    # scheduler_classifer = timm.scheduler.CosineLRScheduler(optimizer_classifer, t_initial=training_epochs, warmup_t=0, warmup_lr_init=0.1)
    #
    # optimizer_embedder = timm.optim.AdamP(model.emb_layer.parameters(), lr=0.0001, weight_decay=0.00001)
    # scheduler_embedder = timm.scheduler.CosineLRScheduler(optimizer_embedder, t_initial=training_epochs, warmup_t=5, warmup_lr_init=0.00001)

    #optimizer = torch.optim.Adam(model.parameters(), lr=scheduler_params['lr_start'])
    # Defining LR SCheduler
    #scheduler = ShopeeScheduler(optimizer, **scheduler_params)

    # define optimzer
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.scheduler_params['lr_start'])

    # learning rate scheudler
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=7, T_mult=1, eta_min=1e-6, last_epoch=-1)
    history = {'train_loss':[],'validation_loss':[]}

    entroy = nn.CrossEntropyLoss()

    for epoch in range(CFG.epochs):
        # get current epoch training loss
        avg_train_loss, train_acc = training_one_epoch(epoch_num=epoch,
                                            model=model,
                                            dataloader=train_dataloader,
                                            optimizer=optimizer,
                                            scheduler=scheduler,
                                            device=CFG.device,
                                            loss_criteria=entroy)

        # get current epoch validation loss
        avg_validation_loss, test_f1, test_acc = validation_one_epoch(model=model,
                                                   dataloader=validation_dataloader,
                                                   epoch=epoch,
                                                   device=CFG.device,
                                                   loss_criteria=entroy)

        print('train_acc: {} , test_acc: {}, test_f1: {}'.format(train_acc, test_acc, test_f1), flush=True)

        #modify arcface m + 0.16
        model.classifier.update_m(0.04)

        history['train_loss'].append(avg_train_loss)
        history['validation_loss'].append(avg_validation_loss)

        # save model
        torch.save(model.state_dict(), MODEL_PATH + '_{}_'.format(epoch) + str(date.today()) + '_softmax_512x512_{}.pt'.format(CFG.model_name))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            #             'scheduler': lr_scheduler.state_dict()
        },
            MODEL_PATH + '_{}_'.format(epoch) + str(date.today()) + '_softmax_512x512_{}_checkpoints.pt'.format(CFG.model_name)
        )

    # num_steps_per_epoch = len(train_dataloader)
    # progress_bar = tqdm(range(num_steps_per_epoch * num_epochs))
    # global_step = 0
    # test_acc = 0
    # for epoch in range(num_epochs):
    #     num_updates = epoch * num_steps_per_epoch
    #     model.train()
    #     for batch in train_dataloader:
    #         global_step += 1
    #         inputs, targets = batch
    #         inputs.to(device)
    #         targets.to(device)
    #
    #         optimizer.zero_grad()
    #
    #
    #         outputs = model(inputs, targets)
    #
    #         #calc train acc
    #         predictions = torch.argmax(outputs, dim=-1)
    #         train_acc = train_accuracy(predictions, targets)
    #         train_acc_step = train_accuracy.compute()
    #
    #         loss = entroy(outputs, targets)
    #         writer.add_scalar('Loss/train', loss.cpu().detach().numpy(), global_step)
    #         writer.add_scalar('Acc/train', train_acc_step.cpu().detach().numpy(), global_step)
    #
    #         loss.backward()
    #         optimizer.step()
    #
    #         # optimizer_classifer.step()
    #         # scheduler_classifer.step_update(num_updates=num_updates)
    #         # optimizer_classifer.zero_grad()
    #         #
    #         # optimizer_embedder.step()
    #         # scheduler_embedder.step_update(num_updates=num_updates)
    #         # optimizer_embedder.zero_grad()
    #
    #
    #         progress_bar.set_postfix(loss="%.2f" % loss.cpu().detach().numpy(), acc="%.2f" % train_acc_step, test_acc="%.2f" % test_acc)
    #         progress_bar.update(1)
    #
    #         # save model
    #         if global_step % 1000 == 0:
    #             create_dir('./cv_model_daodian/')
    #             torch.save(model,'./cv_model_daodian/{}.pt'.format(global_step))
    #
    #     # scheduler_classifer.step(epoch + 1)
    #     # scheduler_embedder.step(epoch + 1)
    #
    #     if scheduler is not None:
    #         scheduler.step()
    #
    #     train_accuracy_batch = train_accuracy.compute()
    #     if model.classifier.m < 0.6:
    #         model.classifier.update_m(0.02)
    #
    #     # eval after each batch
    #     model.eval()
    #     test_step = 0
    #     for batch in testing_dataloader:
    #         inputs, targets = batch
    #         inputs.to(device)
    #         targets.to(device)
    #         with torch.no_grad():
    #             preds = model(inputs, is_test=True)
    #         predictions = torch.argmax(preds, dim=-1)
    #         test_accuracy(predictions, targets)
    #         test_step += 1
    #     test_acc_total = test_accuracy.compute()
    #     test_acc = test_acc_total.cpu().detach().numpy()
    #     writer.add_scalar('Acc/test', test_acc, global_step)
    #     print(f"{test_acc_total}")