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

if __name__ == '__main__':
    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter()
    #定义模型
    os.system("mkdir -p /home/ma-user/.cache/torch/hub/checkpoints/")
    os.system("cp /home/ma-user/work/efficientnet_b4_ra2_320-7eb33cd5.pth /home/ma-user/.cache/torch/hub/checkpoints/")
    #pretrained_model = timm.create_model('nfnet_l0', pretrained=True)
    pretrained_model = timm.create_model('efficientnet_b4', pretrained=True)
    model = CvClassifier(pretrained_model, 1792, 4234)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = torch.load('./cv_model/44000.pt.bak')
    model.to(device)

    print(model.classifier)

    #加载数据
    config = resolve_data_config({}, model=pretrained_model)
    transform_eff = create_transform(**config)
    train_dateset = create_dataset(name='', root='./spusn_image_v2', transform=transform_eff, is_training=True)
    test_dateset = create_dataset(name='', root='./spusn_image_v2_val', transform=transform_eff, is_training=False)
    training_dataloader = create_loader(
        train_dateset,
        input_size=(3, 224, 224),
        batch_size=96,
        is_training=True,
        num_workers=6
    )
    testing_dataloader = create_loader(
        test_dateset,
        input_size=(3, 224, 224),
        batch_size=64,
        is_training=True,
        num_workers=4
    )

    # train loop
    training_epochs = 300
    cooldown_epochs = 10
    num_epochs = training_epochs + cooldown_epochs

    # optimizer_classifer = timm.optim.AdamP(model.classifier.parameters(), lr=0.01)
    # scheduler_classifer = timm.scheduler.CosineLRScheduler(optimizer_classifer, t_initial=training_epochs, warmup_t=5, warmup_lr_init=0.1)
    # optimizer_embedder = timm.optim.AdamP(model.emb_layer.parameters(), lr=0.00001, weight_decay=0.00001)
    # scheduler_embedder = timm.scheduler.CosineLRScheduler(optimizer_embedder, t_initial=training_epochs, warmup_t=5, warmup_lr_init=0.0001)

    optimizer_classifer = timm.optim.AdamP(model.classifier.parameters(), lr=0.0001)
    scheduler_classifer = timm.scheduler.CosineLRScheduler(optimizer_classifer, t_initial=training_epochs, warmup_t=5, warmup_lr_init=0.001)

    optimizer_embedder = timm.optim.AdamP(model.emb_layer.parameters(), lr=0.0001, weight_decay=0.00001)
    scheduler_embedder = timm.scheduler.CosineLRScheduler(optimizer_embedder, t_initial=training_epochs, warmup_t=5, warmup_lr_init=0.001)


    entroy = nn.CrossEntropyLoss()
    train_accuracy = torchmetrics.Accuracy()
    test_accuracy = torchmetrics.Accuracy()
    train_accuracy.to(device)
    test_accuracy.to(device)

    # model = torch.load('./cv_model/4000.pt')
    # model.to(device)
    # model.eval()
    # test_step = 0
    # for batch in testing_dataloader:
    #     inputs, targets = batch
    #     inputs.to(device)
    #     targets.to(device)
    #     with torch.no_grad():
    #         preds = model(inputs, is_test=True)
    #     predictions = torch.argmax(preds, dim=-1)
    #     test_acc = test_accuracy(predictions, targets)
    #     test_step += 1
    # test_acc_total = test_accuracy.compute()
    # print(f"{test_acc_total}")
    #
    #
    # sys.exit(-1)

    num_steps_per_epoch = len(training_dataloader)
    progress_bar = tqdm(range(num_steps_per_epoch * num_epochs))
    global_step = 0
    test_acc = 0
    for epoch in range(num_epochs):
        num_updates = epoch * num_steps_per_epoch
        model.train()
        for batch in training_dataloader:
            global_step += 1
            inputs, targets = batch
            inputs.to(device)
            targets.to(device)
            outputs = model(inputs, targets)

            #calc train acc
            predictions = torch.argmax(outputs, dim=-1)
            train_acc = train_accuracy(predictions, targets)
            train_acc_step = train_accuracy.compute()

            loss = entroy(outputs, targets)
            writer.add_scalar('Loss/train', loss.cpu().detach().numpy(), global_step)
            writer.add_scalar('Acc/train', train_acc_step.cpu().detach().numpy(), global_step)

            loss.backward()
            optimizer_classifer.step()
            scheduler_classifer.step_update(num_updates=num_updates)
            optimizer_classifer.zero_grad()

            optimizer_embedder.step()
            scheduler_embedder.step_update(num_updates=num_updates)
            optimizer_embedder.zero_grad()

            progress_bar.set_postfix(loss="%.2f" % loss.cpu().detach().numpy(), acc="%.2f" % train_acc_step, test_acc="%.2f" % test_acc)
            progress_bar.update(1)

            # save model
            if global_step % 1000 == 0:
                torch.save(model,'./cv_model/{}.pt'.format(global_step))

        scheduler_classifer.step(epoch + 1)
        scheduler_embedder.step(epoch + 1)
        train_accuracy_batch = train_accuracy.compute()

        # eval after each batch
        model.eval()
        test_step = 0
        for batch in testing_dataloader:
            inputs, targets = batch
            inputs.to(device)
            targets.to(device)
            with torch.no_grad():
                preds = model(inputs, is_test=True)
            predictions = torch.argmax(preds, dim=-1)
            test_accuracy(predictions, targets)
            test_step += 1
        test_acc_total = test_accuracy.compute()
        test_acc = test_acc_total.cpu().detach().numpy()
        writer.add_scalar('Acc/test', test_acc, global_step)
        print(f"{test_acc_total}")