"""
Implementation of "Error Correction Code Transformer" (ECCT)
https://arxiv.org/abs/2203.14966
@author: Yoni Choukroun, choukroun.yoni@gmail.com
"""
from __future__ import print_function
import argparse
import random
import os
from torch.utils.data import DataLoader
from torch.utils import data
from datetime import datetime
import logging
from Codes import *
import time
from torch.optim.lr_scheduler import CosineAnnealingLR
from Model import ECC_Transformer
import matplotlib.pyplot as plt
import numpy as np
import math
import pdb

#python Proto.py --gpus=0 --N_dec=6 --d_model=32 --code_type=BCH --code_n=31 --code_k=16 --standardize --channel=awgn

##################################################################
##################################################################

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

##################################################################

class Code():
    pass

class ECC_Dataset(data.Dataset):
    def __init__(self, code, sigma, dataset_length, channel, params, zero_cw=True):
        self.code = code
        self.sigma = sigma
        self.dataset_length = dataset_length
        self.channel = channel
        self.generator_matrix = code.generator_matrix.transpose(0, 1)
        self.pc_matrix = code.pc_matrix.transpose(0, 1)

        self.zero_word = torch.zeros((self.code.k)).long() if zero_cw else None
        self.zero_cw = torch.zeros((self.code.n)).long() if zero_cw else None
        
        #params에서 값 가져오기
        if self.channel in ["awgn", "rayleigh", "raician"] and (self.sigma is None or len(self.sigma) == 0):
            self.sigma = params.get("sigma", [1.0])  # params에서 가져오기
        if self.channel == "bsc":
            self.p = params.get("p", [0.01])  # params에서 가져오기
        if self.channel == "raician":
            self.K = params.get("K", 3.0)  # params에서 가져오기

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, index):
        # 메시지 및 코드워드 생성
        if self.zero_cw is None:
            m = torch.randint(0, 2, (self.code.k,))
            x = torch.matmul(m, self.generator_matrix) % 2
        else:
            m = self.zero_word
            x = self.zero_cw
        
        # 채널별로 처리
        if self.channel == "awgn":
            # AWGN 채널: BPSK 변조 후 noise 추가
            chosen_sigma = random.choice(self.sigma)
            z = torch.randn(self.code.n) * random.choice(self.sigma)
            y = bin_to_sign(x) + z
            magnitude = torch.abs(y)
            syndrome = torch.matmul(sign_to_bin(torch.sign(y)).long(), self.pc_matrix) % 2
            syndrome = bin_to_sign(syndrome)
            return m.float(), x.float(), z.float(), y.float(), magnitude.float(), syndrome.float(), chosen_sigma

        elif self.channel == "bsc":
            # BSC 채널: 각 비트를 확률 p로 반전
            p_val = random.choice(self.p)
            x_bpsk = bin_to_sign(x)
            flips = torch.bernoulli(torch.full((self.code.n,), p_val)).long()
            sign_flip = torch.where(flips == 1, -1, 1)
            y = x_bpsk * sign_flip
            magnitude = torch.abs(y)
            syndrome = torch.matmul(sign_to_bin(torch.sign(y)).long(), self.pc_matrix) % 2
            syndrome = bin_to_sign(syndrome)
            return m.float(), x.float(), flips.float(), y.float(), magnitude.float(), syndrome.float()

        elif self.channel == "rayleigh":
            # Rayleigh 채널: 레일리 페이딩 계수 적용
            chosen_sigma = random.choice(self.sigma)
            h = np.random.rayleigh(2, (self.code.n,))
            h = torch.from_numpy(h).float()
            z = torch.randn(self.code.n) * random.choice(self.sigma)
            y = bin_to_sign(x) * h + z
            magnitude = torch.abs(y)
            syndrome = torch.matmul(sign_to_bin(torch.sign(y)).long(), self.pc_matrix) % 2
            syndrome = bin_to_sign(syndrome)
            return m.float(), x.float(), z.float(), h.float(), y.float(), magnitude.float(), syndrome.float(), chosen_sigma

        else:
            raise ValueError("지원하지 않는 채널 타입입니다.")
    # def __getitem__(self, index):
    #     if self.zero_cw is None:
    #         m = torch.randint(0, 2, (1, self.code.k)).squeeze()
    #         x = torch.matmul(m, self.generator_matrix) % 2
    #     else:
    #         m = self.zero_word
    #         x = self.zero_cw
    #     z = torch.randn(self.code.n) * random.choice(self.sigma)
    #     z = 1

    #     y = bin_to_sign(x) + z
    #     magnitude = torch.abs(y)
    #     syndrome = torch.matmul(sign_to_bin(torch.sign(y)).long(),
    #                             self.pc_matrix) % 2
        
    #     syndrome = bin_to_sign(syndrome)
    #     temp = sign_to_bin(y)
    #     print(x.numpy().astype(int))
    #     print(temp.numpy().astype(int))
    #     print(syndrome)
    #     exit()

    #     return m.float(), x.float(), z.float(), y.float(), magnitude.float(), syndrome.float()


##################################################################
##################################################################

# EbN0 (dB) = 10 * log10( 1 / (2 * R * sigma^2) )
def std_to_EbN0(sigma, rate):
    return 10 * math.log10(1 / (2 * rate * sigma**2))

def train(model, device, train_loader, optimizer, epoch, LR, channel, params):
    model.train()
    cum_loss = cum_ber = cum_fer = cum_samples = 0
    t = time.time()
    # for batch_idx, (m, x, z, y, magnitude, syndrome) in enumerate(
    #         train_loader):
    for batch_idx, data in enumerate(train_loader):
        if channel == "bsc":
            m, x, flips, y, magnitude, syndrome = data
        elif channel == "rayleigh":
            m, x, z, h, y, magnitude, syndrome, sigma_batch = data
            sigma_avg = sigma_batch.mean().item()
            EbN0_val = std_to_EbN0(sigma_avg, params["rate"])  # (# 수정)
        elif channel == "awgn":
            m, x, z, y, magnitude, syndrome, sigma_batch = data
            sigma_avg = sigma_batch.mean().item()
            rate = code.k / code.n
            EbN0_val = std_to_EbN0(sigma_avg, rate)  # (# 수정)



        else:
            raise ValueError("지원하지 않는 채널 타입입니다.")
        z_mul = (y * bin_to_sign(x))
        z_pred = model(magnitude.to(device), syndrome.to(device))
        
        
        loss, x_pred = model.loss(-z_pred, z_mul.to(device), y.to(device))
        model.zero_grad()
        loss.backward()
        optimizer.step()
        ###
        ber = BER(x_pred, x.to(device))
        fer = FER(x_pred, x.to(device))

        cum_loss += loss.item() * x.shape[0]
        pdb.set_trace()
        print(x.shape[0])
        cum_ber += ber * x.shape[0]
        cum_fer += fer * x.shape[0]
        cum_samples += x.shape[0]

        if (batch_idx+1) % 500 == 0 or batch_idx == len(train_loader) - 1:
            if channel == "bsc":
                logging.info(
                    f'Training epoch {epoch}, Batch {batch_idx + 1}/{len(train_loader)}: '
                    f'LR={LR:.2e}, Loss={cum_loss / cum_samples:.2e} '
                    f'p={params["p"]}, BER={cum_ber / cum_samples:.2e}, FER={cum_fer / cum_samples:.2e}')
            else:
                logging.info(
                    f'Training epoch {epoch}, Batch {batch_idx + 1}/{len(train_loader)}: '
                    f'LR={LR:.2e}, Loss={cum_loss / cum_samples:.2e} '
                    f'EbN0={EbN0_val:.3f} dB, BER={cum_ber / cum_samples:.2e}, FER={cum_fer / cum_samples:.2e}')  # EbN0를 소수점 3자리로 출력 (# 수정)

        # if (batch_idx+1) % 500 == 0 or batch_idx == len(train_loader) - 1:
        #     logging.info(
        #         f'Training epoch {epoch}, Batch {batch_idx + 1}/{len(train_loader)}: LR={LR:.2e}, Loss={cum_loss / cum_samples:.2e} BER={cum_ber / cum_samples:.2e} FER={cum_fer / cum_samples:.2e}')
    logging.info(f'Epoch {epoch} Train Time {time.time() - t}s\n')
    return cum_loss / cum_samples, cum_ber / cum_samples, cum_fer / cum_samples


##################################################################

def test(model, device, test_loader_list, test_range, channel, min_FER=100):
    model.eval()
    test_loss_list, test_loss_ber_list, test_loss_fer_list, cum_samples_all = [], [], [], []
    t = time.time()
    with torch.no_grad():
        for ii, test_loader in enumerate(test_loader_list):
            test_loss = test_ber = test_fer = cum_count = 0.
            data_iter = iter(test_loader)
            for data in test_loader:
                if channel == "awgn":
                     m, x, z, y, magnitude, syndrome, chosen_sigma = data
                elif channel == "bsc":
                    m, x, flips, y, magnitude, syndrome = data
                elif channel == "rayleigh":
                    m, x, z, h, y, magnitude, syndrome, sigma_batch = data
                #(m, x, z, y, magnitude, syndrome) = next(iter(test_loader))
                z_mul = (y * bin_to_sign(x))
                z_pred = model(magnitude.to(device), syndrome.to(device))
                loss, x_pred = model.loss(-z_pred, z_mul.to(device), y.to(device))

                test_loss += loss.item() * x.shape[0]

                test_ber += BER(x_pred, x.to(device)) * x.shape[0]
                test_fer += FER(x_pred, x.to(device)) * x.shape[0]
                cum_count += x.shape[0]
                
                # ✅ BSC 채널이면 `p 값`, 다른 채널이면 `EbN0 값`을 출력하도록 조건 추가
                if (min_FER > 0 and test_fer > min_FER and cum_count > 1e5) or cum_count >= 1e9:
                    if cum_count >= 1e9:
                        print(f'Current frame error : {test_fer}, target frame error : {min_FER}, cum_count : {cum_count}')
                        print(f'Number of samples threshold reached for {"p" if channel == "bsc" else "EbN0"}:{test_range[ii]}')
                    else:
                        print(f'FER count threshold reached for {"p" if channel == "bsc" else "EbN0"}:{test_range[ii]}')
                        break
                # if (min_FER > 0 and test_fer > min_FER and cum_count > 1e5) or cum_count >= 1e9:
                #     if cum_count >= 1e9:
                #         print(f'Number of samples threshold reached for EbN0:{test_range[ii]}')
                #     else:    
                #         print(f'FER count threshold reached for EbN0:{test_range[ii]}')
                #     break
            cum_samples_all.append(cum_count)
            test_loss_list.append(test_loss / cum_count)
            test_loss_ber_list.append(test_ber / cum_count)
            test_loss_fer_list.append(test_fer / cum_count)
            # ✅ BSC 채널이면 `p 값`, 다른 채널이면 `EbN0 값`을 출력
            print(f'Test {"p" if channel == "bsc" else "EbN0"}={test_range[ii]}, BER={test_loss_ber_list[-1]:.2e}')
            # print(f'Test EbN0={test_range[ii]}, BER={test_loss_ber_list[-1]:.2e}')
        ###

        ### ✅ 로그 출력 수정 (BSC 채널일 경우 `p`, 아닌 경우 `EbN0` 출력)
        logging.info('\nTest Loss ' + ' '.join(
            ['{}: {:.2e}'.format("p" if channel == "bsc" else ebno, elem) for (elem, ebno) in zip(test_loss_list, test_range)]))
        logging.info('Test FER ' + ' '.join(
            ['{}: {:.2e}'.format("p" if channel == "bsc" else ebno, elem) for (elem, ebno) in zip(test_loss_fer_list, test_range)]))
        logging.info('Test BER ' + ' '.join(
            ['{}: {:.2e}'.format("p" if channel == "bsc" else ebno, elem) for (elem, ebno) in zip(test_loss_ber_list, test_range)]))
        logging.info('Test -ln(BER) ' + ' '.join(
            ['{}: {:.2e}'.format("p" if channel == "bsc" else ebno, -np.log(elem)) for (elem, ebno) in zip(test_loss_ber_list, test_range)]))
        # logging.info('\nTest Loss ' + ' '.join(
        #     ['{}: {:.2e}'.format(ebno, elem) for (elem, ebno)
        #      in
        #      (zip(test_loss_list, test_range))]))
        # logging.info('Test FER ' + ' '.join(
        #     ['{}: {:.2e}'.format(ebno, elem) for (elem, ebno)
        #      in
        #      (zip(test_loss_fer_list, test_range))]))
        # logging.info('Test BER ' + ' '.join(
        #     ['{}: {:.2e}'.format(ebno, elem) for (elem, ebno)
        #      in
        #      (zip(test_loss_ber_list, test_range))]))
        # logging.info('Test -ln(BER) ' + ' '.join(
        #     ['{}: {:.2e}'.format(ebno, -np.log(elem)) for (elem, ebno)
        #      in
        #      (zip(test_loss_ber_list, test_range))]))
    logging.info(f'# of testing samples: {cum_samples_all}\n Test Time {time.time() - t} s\n')
    return test_loss_list, test_loss_ber_list, test_loss_fer_list

##################################################################
##################################################################
##################################################################


def main(args):
    logging.info(f"Using channel(s): {args.channel}")
    print(args.channel)
    code = args.code
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #################################
    model = ECC_Transformer(args, dropout=0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    # 올바른 params 설정 (bsc가 아닐 경우 sigma와 rate 모두 포함)
    params = {"sigma": args.sigma} if args.channel != "bsc" else {"p": args.p}


    logging.info(model)
    logging.info(f'# of Parameters: {np.sum([np.prod(p.shape) for p in model.parameters()])}')
    #################################
    if args.channel == "bsc":
        BSC_test_p_range = (0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05)
        BSC_train_p_range = (0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05,0.055,0.06,0.065,0.07,0.075,0.08,0.085,0.09,0.095,0.1)
        test_range = BSC_test_p_range
        train_range = BSC_train_p_range
        std_train = list(train_range)
        std_test = list(test_range)
    else:
        EbNo_range_test = range(4, 7)
        EbNo_range_train = range(2, 8)
        test_range = EbNo_range_test
        train_range = EbNo_range_train

        std_train = [EbN0_to_std(ii, code.k / code.n) for ii in train_range]
        std_test = [EbN0_to_std(ii, code.k / code.n) for ii in test_range]

    params = {"p" : args.p} if args.channel == "bsc" else {"sigma" : args.sigma}

 
    # train_dataloader = DataLoader(ECC_Dataset(code, std_train, len=args.batch_size * 500, channel=args.channel, params=params, zero_cw=False), batch_size=int(args.batch_size)
    #                               shuffle=True, num_workers=args.workers)
    
    # test_dataloader_list = [DataLoader(ECC_Dataset(code, [std_test[ii]], len=int(args.test_batch_size), channel=args.channel, params=params, zero_cw=False)
    #                                    batch_size=int(args.test_batch_size), shuffle=False, num_workers=args.workers) for ii in range(len(std_test))]

    train_dataloader = DataLoader(
    ECC_Dataset(code, std_train, dataset_length=args.batch_size * 1000, channel=args.channel, params=params, zero_cw=False), 
    batch_size=int(args.batch_size),
    shuffle=True, 
    num_workers=args.workers
    )

    test_dataloader_list = [
    DataLoader(ECC_Dataset(code, [std_test[ii]], dataset_length=int(args.test_batch_size), channel=args.channel, params=params, zero_cw=False),
               batch_size=int(args.test_batch_size), shuffle=False, num_workers=args.workers) 
    for ii in range(len(std_test))
    ]
   
    #################################
    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        loss, ber, fer = train(model, device, train_dataloader, optimizer,
                               epoch, LR=scheduler.get_last_lr()[0], channel=args.channel, params=params)
        
        scheduler.step()
        if loss < best_loss:
            best_loss = loss
            torch.save(model, os.path.join(args.path, 'best_model'))
        if epoch % 50 == 0 or epoch in [1, args.epochs]:
            test(model, device, test_dataloader_list, test_range, channel = args.channel)
    
#################################
# # 성능 지표를 시각화하고 저장하는 함수
# def plot_comparison_metrics(ebn0_range, ber_values_dict, fer_values_dict=None, save_path='comparison_performance_plot.png'):
#     plt.figure(figsize=(14, 7))
    
#     # 각 코드에 대한 BER 그래프
#     for code_type, ber_values in ber_values_dict.items():
#         plt.plot(ebn0_range, ber_values, marker='o', linestyle='-', label=f'BER - {code_type}')
    
#     # 각 코드에 대한 FER 그래프가 있는 경우 함께 표시
#     if fer_values_dict is not None:
#         for code_type, fer_values in fer_values_dict.items():
#             plt.plot(ebn0_range, fer_values, marker='s', linestyle='--', label=f'FER - {code_type}')
    
#     # 축 설정
#     plt.xscale('linear')  # x축은 일반 선형 축
#     plt.yscale('log')     # y축은 로그 축 (BER/FER는 보통 로그 축으로 표현)
    
#     plt.xlabel('Eb/N0 (dB)')
#     plt.ylabel('Error Rate')
#     plt.title('Comparison of Error Performance for LDPC, Polar, and BCH Codes')
#     plt.grid(True, which="both", linestyle='--', linewidth=0.5)
#     plt.legend()
    
#     # 그래프 저장
#     plt.savefig(save_path)
#     print(f'Comparison performance plot saved at: {save_path}')
#     plt.show()
# ##################################################################################################################
# ##################################################################################################################
##################################################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ECCT')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gpus', type=str, default='-1', help='gpus ids')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=42)

    # Code args
    parser.add_argument('--code_type', type=str, default='POLAR',
                        choices=['BCH', 'POLAR', 'LDPC', 'CCSDS', 'MACKAY'])
    parser.add_argument('--code_k', type=int, default=32)
    parser.add_argument('--code_n', type=int, default=64)
    parser.add_argument('--standardize', action='store_true')

    # model args
    parser.add_argument('--N_dec', type=int, default=6)
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--h', type=int, default=8)

    #채널 종류 삽입
    #채널 params값들은 중복되지않도록 복잡하게 설정
    parser.add_argument('--channel', type=str, default='awgn',
                    choices=['awgn', 'bsc', 'rayleigh', 'raician'],
                    help="시뮬레이션할 채널 타입")
    parser.add_argument('--sigma', nargs='+', type=float, default=[1.0],
                        help="AWGN, Rayleigh, Raician 채널의 노이즈 표준편차")
    parser.add_argument('--p', nargs='+', type=float, default=[0.01],
                        help="BSC 채널에서 사용할 flip 확률")
    parser.add_argument('--K', type=float, default=3.0,
                        help="Raician 채널의 K-factor 값")

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    set_seed(args.seed)


    ####################################################################

    code = Code()
    code.k = args.code_k
    code.n = args.code_n
    code.code_type = args.code_type
    G, H = Get_Generator_and_Parity(code,standard_form=args.standardize)
    code.generator_matrix = torch.from_numpy(G).transpose(0, 1).long()
    code.pc_matrix = torch.from_numpy(H).long()
    args.code = code
    ####################################################################
    model_dir = os.path.join('Results_ECCT',
                             args.code_type + '__Code_n_' + str(
                                 args.code_n) + '_k_' + str(
                                 args.code_k) + '__' + datetime.now().strftime(
                                 "%d_%m_%Y_%H_%M_%S"))
    os.makedirs(model_dir, exist_ok=True)
    args.path = model_dir
    handlers = [
        logging.FileHandler(os.path.join(model_dir, 'logging.txt'))]
    handlers += [logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO, format='%(message)s',
                        handlers=handlers)
    logging.info(f"Path to model/logs: {model_dir}")
    logging.info(args)

    main(args)
