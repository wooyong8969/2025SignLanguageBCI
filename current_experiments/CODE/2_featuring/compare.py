import numpy as np

# ---------- 1-5 세션 ---------- #
rwa_1_5        = np.load(r'current_experiments\DATA\processed\experiment_001\experiment_001(1-5)_cleaned.npy')
dwt_time_1_5   = np.load('1_5_dwt_time.npy')
dwt_freq_1_5   = np.load('1_5_dwt_freq.npy')
csp_1_5        = np.load('1_5_csp.npy')
# sel_1_5        = np.load('6_8_X_train_sel.npy')
# lda_1_5        = np.load('6_8_X_train_lda.npy')

# ---------- 1-8 세션 ---------- #
rwa_1_8        = np.load(r'current_experiments\DATA\processed\experiment_001\experiment_001(1-8)_cleaned.npy')[:rwa_1_5.shape[0]]
dwt_time_1_8   = np.load('1_8_dwt_time.npy')[:dwt_time_1_5.shape[0]]
dwt_freq_1_8   = np.load('1_8_dwt_freq.npy')[:dwt_freq_1_5.shape[0]]
csp_1_8        = np.load('1_8_csp.npy')[:csp_1_5.shape[0]]
# sel_1_8        = np.load('1_8_X_train_sel.npy')[:sel_1_5.shape[0]]
# lda_1_8        = np.load('1_8_X_train_lda.npy')[:lda_1_5.shape[0]]

# ---------- 결과 비교 ---------- #
print("raw data 일치?:      ", np.allclose(rwa_1_5, rwa_1_8))
print("DWT time 일치?:      ", np.allclose(dwt_time_1_5, dwt_time_1_8))
print("DWT freq 일치?:      ", np.allclose(dwt_freq_1_5, dwt_freq_1_8))
print("CSP feature 일치?:   ", np.allclose(csp_1_5, csp_1_8))
# print("Feature sel 일치?:   ", np.allclose(sel_1_5, sel_1_8))
# print("LDA 일치?:           ", np.allclose(lda_1_5, lda_1_8))
