import heapq, json, pickle
import numpy as np
import pandas as pd
from readrawdata import readdata
from wilcoxonsnp import gen_snp

def rank_snp(snp_info, k=25):
    """
        input: a snp array with idx and score
        output: top k snp
    """
    heapq.heapify(snp_info)
    snp_list = []
    for i in range(k):
        if len(snp_info) == 0:
            return
        n_score, snp, p_val= heapq.heappop(snp_info)
        print (-n_score, snp, p_val)
        snp_list.append(snp)
    return np.array(snp_list)

# X, y, n_snp = readdata("/Users/zhang/Documents/CSIC/I529/team_CRY/Data/call_method_32.b",
#                         "/Users/zhang/Documents/CSIC/I529/team_CRY/Data/phenotype.csv",
#                         seed=0)

# with open("snp_data.p", "wb") as f:
#     pickle.dump((X, y, n_snp), f)

# start = time.time()
# snp, snp_info = gen_snp(X, y, n_snp, k=200, r=30, p=0.05)
# print ("the running time is "+str(time.time()-start))
# print (len(snp))
# print (snp)


# with open("snp_info.p", "wb") as f:
#     pickle.dump((snp, snp_info), f)

# with open("snp_list.txt", "w") as f:
#     json.dump(snp, f)

with open("snp_info.p", "rb") as f:
    SNP, snp_info = pickle.load(f)


snp_list = rank_snp(snp_info)-1
data = pd.read_csv("/Users/zhang/Documents/CSIC/I529/team_CRY/Data/snp_chromos_position.csv", sep=",")
for snp_i in snp_list:
    print (data.iloc[snp_i,1:3].to_string())


