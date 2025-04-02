import torch
import torch.nn as nn
from amr.dataloaders.dataloader2 import *
from amr.utils import *
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from complexity import *

# 在数据加载器中添加排序逻辑



def main(cfgs):
    logger.info('=> PyTorch Version: {}'.format(torch.__version__))

    # Environment initialization

    device, pin_memory = init_device(cfgs.opt_params.seed, cfgs.opt_params.cpu, cfgs.opt_params.gpu)
    print(device, pin_memory)

    train_loader, valid_loader, test_loader,  mods = AMRDataLoader(dataset=cfgs.data_settings.dataset,
                                                                        Xmode=cfgs.data_settings.Xmode,
                                                                        hdf5_path = r"D:\Desktop\AMC competition\data_fil_total\data_fil_split3.h5",
                                                                        batch_size=cfgs.opt_params.batch_size,
                                                                        num_workers=cfgs.opt_params.workers,
                                                                        pin_memory=pin_memory,
                                                                        ddp = cfgs.modes.ddp,
                                                                        random_mix = cfgs.data_settings.Xmode.options.random_mix,
                                                                        mod_type=cfgs.data_settings.mod_type,
                                                                        )()

    model = init_model(cfgs)
    #! 冻结除了'TADBlock以外的所有层
    # for name, param in model.named_parameters():
    #     if 'TADBlock' not in name:  # 如果参数名称中不包含 'TADBlock'，则冻结
    #         param.requires_grad = False
    #!
    # model_complexity(cfgs, model)
    model.to(device)
    # if cfgs.modes.ddp == True:
    #     # DistributedDataParallel
    #     model= torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    #     model = DDP(model, device_ids=[local_rank], output_device=local_rank)


    criterion = init_loss(cfgs.modes.loss)

    if cfgs.modes.train:
        print("training start...")
        #! 过滤被冻结的参数
        # optimizer_params = filter(lambda p: p.requires_grad, model.parameters())
        #!
        if cfgs.modes.optimizer == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfgs.opt_params.lr, weight_decay=1e-4, amsgrad=True)
        elif cfgs.modes.optimizer == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=cfgs.opt_params.lr, momentum=0.9, weight_decay=0.02)
        else:
            print("no available optimizer")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel')
        trainer = Trainer(model=model, device=device, optimizer=optimizer, scheduler = scheduler, criterion=criterion,
                          save_path='results/' + cfgs.modes.method + '/' + cfgs.modes.path + '/' + cfgs.data_settings.dataset + '/checkpoints',
                          early_stop=cfgs.opt_params.early_stop,
                          mask_flag = cfgs.data_settings.Xmode.options.zero_mask,
                          random_mix_flag = cfgs.data_settings.Xmode.options.random_mix,
                          Random_Matrix = None, 
                          batch_size = cfgs.opt_params.batch_size,
                          loss_name = cfgs.modes.loss,
                          snrs = None,
                          model_name = cfgs.modes.method)
        train_loss, train_CQ_score, valid_loss, valid_CQ_score = trainer.loop(cfgs.opt_params.epochs, train_loader, valid_loader)
        draw_train(train_loss, train_CQ_score, valid_loss, valid_CQ_score,
                   save_path='./results/' + cfgs.modes.method + '/' + cfgs.modes.path + '/' + cfgs.data_settings.dataset + '/draws')

    # 测试前重新加载最优模型
    cfgs.modes.train = False
    model = init_model(cfgs)
    model.to(device)
    if cfgs.modes.ddp == True:
        torch.distributed.barrier()

    print("testing start...")
    test_loss, test_CQ_score, _ = Tester(model=model, device=device, criterion=criterion,
                                                                         classes=len(cfgs.data_settings.mod_type),
                                                                         snrs=None,
                                                                         batch_size = cfgs.opt_params.batch_size,
                                                                         loss_name = cfgs.modes.loss,
                                                                         model_name = cfgs.modes.method)(test_loader)
    # draw_conf(test_conf, save_path='./results/' + cfgs.modes.method + '/' + cfgs.modes.path + '/' + cfgs.data_settings.dataset + '/draws',
    #           labels=mods, order="total")

    # for i in range(len(snrs)):
    #     logger.info(f'test_snr : {snrs[i]:.0f} | '
    #                 f'test_acc : {test_acc_snr[i]:.3f}')
    #     draw_conf(test_conf_snr[i],
    #               save_path='./results/' + cfgs.modes.method + '/' + cfgs.modes.path + '/' + cfgs.data_settings.dataset + '/draws', labels=mods,
    #               order=str(snrs[i]))
    # draw_acc(snrs, test_acc_snr,
    #          save_path='./results/' + cfgs.modes.method + '/' + cfgs.modes.path + '/' + cfgs.data_settings.dataset + '/draws')
    # print(test_acc_snr)
    logger.info(f'test_loss : {test_loss:.3e} | '
                f'test_CQ_score : {test_CQ_score:.4f}')
    #! Y_pred 的形状是 (batch_size,)，每个值是一个整数，范围是 [0, num_classes - 1]，记得提交时全体要加上1

if __name__ == '__main__':
    cfgs = get_cfgs()
    main(cfgs)