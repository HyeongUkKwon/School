from PIL import Image
import argparse
import torch
import os
import logging
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import warnings
import pdb



def Read_pc_matrixrix_alist(fileName):
    with open(fileName, 'r') as file:
        lines = file.readlines()
        columnNum, rowNum = np.fromstring(lines[0].rstrip('\n'), dtype=int, sep=' ')
        H = np.zeros((rowNum, columnNum)).astype(int)
        for column in range(4, 4 + columnNum):
            nonZeroEntries = np.fromstring(lines[column].rstrip('\n'), dtype=int, sep=' ')
            for row in nonZeroEntries:
                if row > 0:
                    H[row - 1, column - 4] = 1
        return H

def get_standard_form(pc_matrix_):
    pc_matrix = pc_matrix_.copy().astype(bool)
    next_col = min(pc_matrix.shape)
    for ii in range(min(pc_matrix.shape)):
        while True:
            rows_ones = ii + np.where(pc_matrix[ii:, ii])[0]
            if len(rows_ones) == 0:
                new_shift = np.arange(ii, min(pc_matrix.shape) - 1).tolist() + [min(pc_matrix.shape) - 1, next_col]
                old_shift = np.arange(ii + 1, min(pc_matrix.shape)).tolist() + [next_col, ii]
                pc_matrix[:, new_shift] = pc_matrix[:, old_shift]
                next_col += 1
            else:
                break
        pc_matrix[[ii, rows_ones[0]], :] = pc_matrix[[rows_ones[0], ii], :]
        other_rows = pc_matrix[:, ii].copy()
        other_rows[ii] = False
        pc_matrix[other_rows] = pc_matrix[other_rows] ^ pc_matrix[ii]
    return pc_matrix.astype(int)

def row_reduce(mat, ncols=None):
    assert mat.ndim == 2
    ncols = mat.shape[1] if ncols is None else ncols
    mat_row_reduced = mat.copy()
    p = 0
    for j in range(ncols):
        idxs = p + np.nonzero(mat_row_reduced[p:, j])[0]
        if idxs.size == 0:
            continue
        mat_row_reduced[[p, idxs[0]], :] = mat_row_reduced[[idxs[0], p], :]
        idxs = np.nonzero(mat_row_reduced[:, j])[0].tolist()
        idxs.remove(p)
        mat_row_reduced[idxs, :] = mat_row_reduced[idxs, :] ^ mat_row_reduced[p, :]
        p += 1
        if p == mat_row_reduced.shape[0]:
            break
    return mat_row_reduced, p

def get_generator(pc_matrix_):
    assert pc_matrix_.ndim == 2
    pc_matrix = pc_matrix_.copy().astype(bool).transpose()
    pc_matrix_I = np.concatenate((pc_matrix, np.eye(pc_matrix.shape[0], dtype=bool)), axis=-1)
    pc_matrix_I, p = row_reduce(pc_matrix_I, ncols=pc_matrix.shape[1])
    return row_reduce(pc_matrix_I[p:, pc_matrix.shape[1]:])[0]

def Get_Generator_and_Parity(code_type, n, k, standard_form):
    path_pc_mat = os.path.join('Codes_DB', f'{code_type}_N{str(n)}_K{str(k)}')
    if code_type in ['POLAR', 'BCH']:
        ParityMatrix = np.loadtxt(path_pc_mat + '.txt')
    elif code_type in ['CCSDS', 'LDPC', 'MACKAY']:
        ParityMatrix = Read_pc_matrixrix_alist(path_pc_mat + '.alist')
    else:
        raise Exception(f'Wrong code {code_type}')
    if standard_form and code_type not in ['CCSDS', 'LDPC', 'MACKAY']:
        ParityMatrix = get_standard_form(ParityMatrix).astype(int)
        GeneratorMatrix = np.concatenate([np.mod(-ParityMatrix[:, min(ParityMatrix.shape):].transpose(), 2), np.eye(k)], 1).astype(int)
    else:
        GeneratorMatrix = get_generator(ParityMatrix)
    assert np.all(np.mod((np.matmul(GeneratorMatrix, ParityMatrix.transpose())), 2) == 0) and np.sum(GeneratorMatrix) > 0
    return GeneratorMatrix.astype(float), ParityMatrix.astype(float)

def bin_to_sign(x):
    return 1 - 2 * x

def sign_to_bin(x):
    return 0.5 * (1 - x)

def EbN0_to_std(EbN0, rate):
    snr = EbN0 + 10. * np.log10(2 * rate)
    return np.sqrt(1. / (10. ** (snr / 10.)))

def BER(x_pred, x_gt):
    return torch.mean((x_pred != x_gt).float()).item()

def FER(x_pred, x_gt):
    return torch.mean(torch.any(x_pred != x_gt, dim=1).float()).item()

def save_results(values, fer_values, ber_values, selected_channel, file_path="simulation_results.txt"):
    if selected_channel == "bsc":
        label = "p"
    else:
        label = "EbN0"
    with open(file_path, "w") as f:
        for val, fer, ber in zip(values, fer_values, ber_values):
            f.write(f"{label}: {val}, FER: {fer:.2e}, BER: {ber:.2e}\n")



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    warnings.filterwarnings("ignore", category=FutureWarning)
    model = torch.load('100qjs', map_location=device).to(device)
    model.eval()

    n, k = 31, 16
    G, H = Get_Generator_and_Parity('BCH', n=31, k=16, standard_form=False)
    pc_matrix = torch.from_numpy(H).float().to(device)
    codeword = torch.zeros((1000, 31), dtype=torch.int).to(device)
    values = []
    
    ebn0_results = []
    fer_results = []
    ber_results = []



    # 사용자로부터 채널 선택
    selected_channel = input("원하는 channel을 입력하시오 (awgn, bsc, rayleigh): ").strip().lower()
    if selected_channel not in ['awgn', 'bsc', 'rayleigh']:
        print("지원하지 않는 채널입니다. 기본적으로 awgn 채널로 진행합니다.")
        selected_channel = 'awgn'

    if selected_channel == 'awgn':
        code_rate = k / n
        Eb_No_dB = list(range(0, 10, 1))
        noise_std = EbN0_to_std(Eb_No_dB, code_rate)
        i = 0
        for SIGMA in noise_std:
            #몇회 돌렸는가가
            count = 0
            #그냥 에러 개수
            error_count = 0
            #ber에 대해 계산할 때 쓸 것임임
            bit_error_count = 0
            # 에러를 1000번까지 검출할때까지 동작작
            while error_count < 100000:
                with torch.no_grad():
                    #코드워드 생성(무작위 난수값)
                    x = codeword
                    # awgn 채널 특성 추가
                    z = torch.randn((1000, 31)).to(device) * SIGMA
                    # BPSK 변조 후 노이즈 추가가
                    y = bin_to_sign(x) + z
                    # 신호의 크기 계산
                    magnitude = torch.abs(y)
                    # 신드롬 계산(지금 y값은 BPSK 변조된 값이니까 다시 변환해줘야함), 정답지이므로 변조된 신호에 대해서 역변조 후 pc_matrix와 곱해줘 신드롬값 계산
                    syndrome = torch.matmul(sign_to_bin(torch.sign(y)).float(), pc_matrix.transpose(0, 1)) % 2
                    # syndrome값을 변환할껀데 이때 모델에서는 역변조 되지않은 값에 대해서 계산할 것이므로 정답지 또한 변조해줘야함함
                    syndrome = bin_to_sign(syndrome)
                    # 예측값 계산
                    z_pred = model(magnitude, syndrome)
                    # 예측값을 다시 이진화
                    x_pred = sign_to_bin(torch.sign(z_pred * torch.sign(y)))
                    # frame error 계산

                    # 프레임 에러는 한 배치에서 하나라도 다르면 에러로 간주
                    batch_frame_errors = torch.sum(torch.any(x_pred != x, dim=1)).item()
                    # 비트 에러는 각 비트에대해서 비교교
                    batch_bit_errors = torch.sum(x_pred != x).item()
                    error_count += batch_frame_errors
                    bit_error_count += batch_bit_errors
                    count += 1000  # 1번 동작 --> 1000개의 codeword가 생성되므로
            fer  = error_count / count             # FER 계산
            ber = bit_error_count / (count * n)    # BER 계산 (코드워드 길이가 31)
            print(f'AWGN - EbN0: {Eb_No_dB[i]}, FER: {fer:.2e}')
            print(f'AWGN - EbN0: {Eb_No_dB[i]}, BER: {ber:.2e}')
            ebn0_results.append(Eb_No_dB[i])
            fer_results.append(fer)
            ber_results.append(ber)
            print(f'AWGN - EbN0: {Eb_No_dB[i]}, error_count: {error_count}')
            print(f'AWGN - EbN0: {Eb_No_dB[i]}, count: {count}')
            print(f'AWGN - EbN0: {Eb_No_dB[i]}, error_rate: {error_count / count}')

            i += 1


    elif selected_channel == 'bsc':
        # BSC 채널 시뮬레이션: p 값의 범위 설정
        p_vals = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]

        for p_val in p_vals:
            count = 0
            error_count = 0
            bit_error_count = 0
            # 에러가 1개라도 검출될 때까지 반복 (Monte Carlo 방식)
            while error_count < 5000:
                with torch.no_grad():
                    x = codeword
                    x_bpsk = bin_to_sign(x)
                    flips = torch.bernoulli(torch.full(x_bpsk.shape, p_val)).to(device)
                    magnitude = torch.abs(y)
                    syndrome = torch.matmul(
                        sign_to_bin(torch.sign(y)).float(), 
                        pc_matrix.transpose(0, 1)
                    ) % 2
                    syndrome = bin_to_sign(syndrome)
                    z_pred = model(magnitude, syndrome)
                    x_pred = sign_to_bin(torch.sign(z_pred * torch.sign(y)))
                    batch_frame_errors = torch.sum(torch.any(x_pred != x, dim=1)).item()
                    batch_bit_errors = torch.sum(x_pred != x).item()
                    error_count += batch_frame_errors
                    bit_error_count += batch_bit_errors
                    count += 500
            fer = error_count / count                     # FER 계산
            ber = bit_error_count / (count * 31)            # BER 계산
            print(f'BSC - p: {p_val}, error_count: {error_count}, count: {count}, FER: {fer}')
            print(f'BSC - p: {p_val}, BER: {ber}')
            # 결과 리스트에 저장 (여기서는 EbN0 대신 p 값을 저장)
            ebn0_results.append(p_val)
            fer_results.append(fer)
            ber_results.append(ber)
        
    elif selected_channel == 'rayleigh':
        code_rate = 0.5
        Eb_No_dB = list(range(2, 8, 1))
        noise_std = EbN0_to_std(Eb_No_dB, code_rate)
        i = 0
        for SIGMA in noise_std:
            count = 0
            error_count = 0
            bit_error_count = 0
            while error_count < 5000:
                with torch.no_grad():
                    x = codeword
                    h = torch.from_numpy(np.random.rayleigh(scale=2, size=x.shape)).float().to(device)
                    z = torch.randn(x.shape).to(device) * SIGMA
                    y = bin_to_sign(x) * h + z
                    magnitude = torch.abs(y)
                    syndrome = torch.matmul(sign_to_bin(torch.sign(y)).float(), pc_matrix.transpose(0, 1)) % 2
                    syndrome = bin_to_sign(syndrome)
                    z_pred = model(magnitude, syndrome)
                    x_pred = sign_to_bin(torch.sign(z_pred * torch.sign(y)))
                    batch_frame_errors = torch.sum(torch.any(x_pred != x, dim=1)).item()
                    batch_bit_errors = torch.sum(x_pred != x).item()  # 추가: 배치 내 비트 에러 계산
                    error_count += batch_frame_errors
                    bit_error_count += batch_bit_errors            # 추가: bit_error_count 업데이트
                    count += 500
                count += 1
            print(f'Rayleigh - EbN0: {Eb_No_dB[i]}, error_count: {error_count}')
            print(f'Rayleigh - EbN0: {Eb_No_dB[i]}, count: {count}')
            print(f'Rayleigh - EbN0: {Eb_No_dB[i]}, error_rate: {error_count / count}')
            ebn0_results.append(Eb_No_dB[i])            
            fer = error_count / count             # FER 계산
            ber = bit_error_count / (count * 31)    # BER 계산 (코드워드 길이가 31)
            print(f'AWGN - EbN0: {Eb_No_dB[i]}, FER: {fer}')
            print(f'AWGN - EbN0: {Eb_No_dB[i]}, BER: {ber}')
            fer_results.append(fer)               # FER 값을 리스트에 추가
            ber_results.append(ber)               # BER 값을 리스트에 추가   

            
            i += 1
    return ebn0_results, fer_results, ber_results, selected_channel

if __name__ == "__main__":
    values, fer_values, ber_values, selected_channel = main()
    save_results(values, fer_values, ber_values, selected_channel, file_path="100번.txt")


