import sys
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
table = pd.read_csv(sys.argv[1], sep='\t')
if 'fe_js' not in table:
	nt = table[['clf','feats','lex','MWE_avg','total_acc','ambig_acc','unambig_acc','unk_acc']]
else:
	nt = table[['clf','feats','lex','MWE_avg','total_acc','ambig_acc','unambig_acc','unk_acc','fe_js']]
print(nt)

