from PIL import Image
import argparse
import torch
import os
import logging
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import warnings
import logging

def Read_pc_matrix_alist(fileName):
    with open(fileName, 'r') as file:
        lines = file.readlines()
        columnNum, rowNum = np.fromstring(
            lines[0].rstrip('\n'), dtype=int, sep=' ')
        H = np.zeros((rowNum, columnNum)).astype(int)
        for column in range(4, 4 + columnNum):
            nonZeroEntries = np.fromstring(
                lines[column].rstrip('\n'), dtype=int, sep=' ')
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
                new_shift = np.arange(ii, min(pc_matrix.shape) - 1).tolist()+[min(pc_matrix.shape) - 1,next_col]
                old_shift = np.arange(ii + 1, min(pc_matrix.shape)).tolist()+[next_col, ii]
                pc_matrix[:, new_shift] = pc_matrix[:, old_shift]
                next_col += 1
            else:
                break
        pc_matrix[[ii, rows_ones[0]], :] = pc_matrix[[rows_ones[0], ii], :]
        other_rows = pc_matrix[:, ii].copy()
        other_rows[ii] = False
        pc_matrix[other_rows] = pc_matrix[other_rows] ^ pc_matrix[ii]
    return pc_matrix.astype(int)

def Get_Generator_and_Parity(code_type, n, k, standard_form):
    path_pc_mat = os.path.join('Codes_DB', f'{code_type}_N{str(n)}_K{str(k)}')
    if code_type in ['POLAR', 'BCH']:
        ParityMatrix = np.loadtxt(path_pc_mat+'.txt')
    elif code_type in ['CCSDS', 'LDPC', 'MACKAY']:
        ParityMatrix = Read_pc_matrix_alist(path_pc_mat+'.alist')
    else:
        raise Exception(f'Wrong code {code_type}')
    if standard_form and code_type not in ['CCSDS', 'LDPC', 'MACKAY']:
        ParityMatrix = get_standard_form(ParityMatrix).astype(int)
        GeneratorMatrix = np.concatenate([np.mod(-ParityMatrix[:, min(ParityMatrix.shape):].transpose(),2),np.eye(k)],1).astype(int)
    else:
        GeneratorMatrix = get_generator(ParityMatrix)
    assert np.all(np.mod((np.matmul(GeneratorMatrix, ParityMatrix.transpose())), 2) == 0) and np.sum(GeneratorMatrix) > 0
    return GeneratorMatrix.astype(float), ParityMatrix.astype(float)

def bin_to_sign(x):
    return 1 - 2 * x

def sign_to_bin(x):
    return 0.5 * (1 - x)

def EbN0_to_std(EbN0, rate):
    snr =  EbN0 + 10. * np.log10(2 * rate)
    return np.sqrt(1. / (10. ** (snr / 10.)))

def BER(x_pred, x_gt):
    return torch.mean((x_pred != x_gt).float()).item()

def FER(x_pred, x_gt):
    return torch.mean(torch.any(x_pred != x_gt, dim=1).float()).item()

def row_reduce(mat, ncols=None):
    assert mat.ndim == 2
    ncols = mat.shape[1] if ncols is None else ncols
    mat_row_reduced = mat.copy()
    p = 0
    for j in range(ncols):
        idxs = p + np.nonzero(mat_row_reduced[p:,j])[0]
        if idxs.size == 0:
            continue
        mat_row_reduced[[p,idxs[0]],:] = mat_row_reduced[[idxs[0],p],:]
        idxs = np.nonzero(mat_row_reduced[:,j])[0].tolist()
        idxs.remove(p)
        mat_row_reduced[idxs,:] = mat_row_reduced[idxs,:] ^ mat_row_reduced[p,:]
        p += 1
        if p == mat_row_reduced.shape[0]:
            break
    return mat_row_reduced, p

def get_generator(pc_matrix_):
    assert pc_matrix_.ndim == 2
    pc_matrix = pc_matrix_.copy().astype(bool).transpose()
    pc_matrix_I = np.concatenate((pc_matrix, np.eye(pc_matrix.shape[0], dtype=bool)), axis=-1)
    pc_matrix_I, p = row_reduce(pc_matrix_I, ncols=pc_matrix.shape[1])
    return row_reduce(pc_matrix_I[p:,pc_matrix.shape[1]:])[0]


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load('test',map_location=device, weights_only=False).to(device)
    #model 평가 모드
    model.eval()

    n, k = 31, 16
    G, H = Get_Generator_and_Parity('BCH', n = 31, k = 16, standard_form=False)
    pc_matrix = torch.from_numpy(H).float().to(device)
    codeword = torch.ones(1000, 31, dtype=int).to(device)

    code_rate = k / n
    Eb_No_dB = list(range(0, 10, 1))
    noise_std = EbN0_to_std(Eb_No_dB, code_rate)

    bit_error_count = 0
    i = 0
    for SIGMA in noise_std:
        #셈 설정
        frame_count = 0
        bit_count = 0
        bit_error_count = 0
        fer = 0
        ber = 0
        frame_error_count = 0
        while bit_error_count < 10000:
            with torch.no_grad():
                x = codeword.clone()
                z = torch.randn((1000, 31)).to(device) * SIGMA
                #channel특성(노이즈) 추가
                y = bin_to_sign(x) + z
                magnitude = torch.abs(y)
                # syndrome = torch.matmul(sign_to_bin(torch.sign(y)).float(), pc_matrix.transpose(0, 1)) % 2
                syndrome = torch.matmul(sign_to_bin(torch.sign(y)).float(), pc_matrix.transpose(0, 1)) % 2
                syndrome = bin_to_sign(syndrome)
                z_pred = model(magnitude.to(device), syndrome.to(device))
                x_pred = sign_to_bin(torch.sign(z_pred * torch.sign(y).to(device)))
                bit_error_count += torch.sum(x_pred != x).item()
                frame_error_count += torch.sum(torch.any(x_pred != x, dim=1)).item()
                #맞춘갯수
                
                frame_count += 1000
                bit_count += 1000 * n

        print(f'EbN0:{Eb_No_dB[i]}, bit_error_count:{bit_error_count}')
        print(f'EbN0:{Eb_No_dB[i]}, frame_error_count:{frame_error_count}')
        print(f'EbN0:{Eb_No_dB[i]}, count:{frame_count}')
        # print(f'EbN0:{Eb_No_dB[i]}, bit_error_rate:{bit_error_count/bit_count}')  
        # print(f'EbN0:{Eb_No_dB[i]}, frame_error_rate:{frame_error_count/frame_count}')
        print(f'EbN0:{Eb_No_dB[i]}, bit_error_rate:{(bit_error_count/bit_count):.2e}')
        print(f'EbN0:{Eb_No_dB[i]}, frame_error_rate:{(frame_error_count/frame_count):.2e}')

        i += 1
    return frame_count, bit_error_count

if __name__ == '__main__':
    count, error_count = main()
        


