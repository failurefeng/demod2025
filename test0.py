import torch
import torch.nn as nn
from amr.dataloaders.dataloader2 import *
from amr.utils import *
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from complexity import *
import tqdm


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

    def look_for_invalid(loader):
        invalid_count = 0
        invalid_count2 = 0
        for _, batch in enumerate(loader):
            X = batch['signals']
            symbol_widths = batch['symbol_widths']
            data_lens = batch['data_lens']
            for i in range(X.shape[0]):
                    # signal = X[i]  # 形状为 (2, data_lens[i])
                    symbol_width = round(20 * symbol_widths[i].item())  # 码元宽度
                    
                    data_len = data_lens[i]  # 信号的实际长度

                    if data_len % symbol_width != 0:
                        invalid_count += 1
                        print(f'symbol_widths[{i}]:',symbol_widths[i],'symbol_width: ', symbol_width,'data_len: ', data_len)
                        if data_len % (symbol_width+1) != 0:
                            invalid_count2 += 1
                    # print(f'signal: {signal}')
                    # if data_len % symbol_width != 0:
                    #     pad_len =  symbol_width - (data_len % symbol_width)
                    # else:
                    #     pad_len = 0
                    # 分割信号，每个码元切片长度为 symbol_width
                    # effective_signal = signal[:, :data_len]  # 取出当前通道的信号，形状为 (2, data_len)
                    # if pad_len > 0:
                    #     effective_signal = torch.cat((effective_signal, torch.zeros((2,pad_len))), dim=1)
                    # effective_split_pieces = torch.split(effective_signal, symbol_width, dim=1)  # 按 symbol_width 分割, 形状为 (2, symbol_width)
                    # effective_split_signal = torch.stack(effective_split_pieces, dim=0)  # 按码元切片维度堆叠, 形状为 (num_symbol, 2, symbol_width)
                    # print(f'effective_split_pieces:', effective_split_pieces)
                    # print(f'effective_split_signal:', effective_split_signal)
        print(str(loader),':')
        print(f'invalid_count:', invalid_count)
        print(f'invalid_count2:', invalid_count2)
        return invalid_count, invalid_count2
    train_c, train_c2 = look_for_invalid(train_loader)
    valid_c, valid_c2 = look_for_invalid(valid_loader)
    test_c, test_c2 = look_for_invalid(test_loader)
    c = train_c + valid_c + test_c
    c2 = train_c2 + valid_c2 + test_c2
    print(f'total invalid count:', c)
    print(f'total invalid count2:', c2)
if __name__ == '__main__':
    cfgs = get_cfgs()
    main(cfgs)