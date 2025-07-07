import numpy as np

# ---------- 1-5 세션 ---------- #
dwt_time_1_5   = np.load('1_5_dwt_time.npy')
dwt_freq_1_5   = np.load('1_5_dwt_freq.npy')
csp_1_5        = np.load('1_5_csp.npy')
lda_1_5        = np.load('1_5_lda.npy')

# ---------- 1-8 세션 ---------- #
dwt_time_1_8   = np.load('1_8_dwt_time.npy')[:dwt_time_1_5.shape[0]]
dwt_freq_1_8   = np.load('1_8_dwt_freq.npy')[:dwt_freq_1_5.shape[0]]
csp_1_8        = np.load('1_8_csp.npy')[:csp_1_5.shape[0]]
lda_1_8        = np.load('1_8_lda.npy')[:lda_1_5.shape[0]]

# ---------- 결과 비교 ---------- #
print("DWT time 일치?:      ", np.allclose(dwt_time_1_5, dwt_time_1_8))
print("DWT freq 일치?:      ", np.allclose(dwt_freq_1_5, dwt_freq_1_8))
print("CSP feature 일치?:   ", np.allclose(csp_1_5, csp_1_8))
print("LDA 일치?:           ", np.allclose(lda_1_5, lda_1_8))
