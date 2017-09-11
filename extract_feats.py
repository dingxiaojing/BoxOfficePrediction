import numpy as np

money=[]
search=[]
# features
movie_feats=[]
fp=open('D:/spark/movie/data.txt')
for line in fp.readlines():
    strlist=line.split('')
    feat_vec=[float(feat.strip()) for feat in strlist[8:18]]
    year,month,day=[int(strlist[7][:10].split('-')[i]) for i in range(3)]
    if year==2013:
        feat_vec+=[1.0,0.0,0.0]
    elif year==2014:
        feat_vec+=[0.0,1.0,0.0]
    else:
        feat_vec+=[0.0,0.0,1.0]
    feat_vec+=[int(f) for f in strlist[35:-2]]
    influence_vec=[int(f) for f in strlist[35:-2]]
    if influence_vec and feat_vec[-1]!=-1:
        movie_feats.append(feat_vec)
        money.append(int(strlist[-1].strip()[3:])/100.0)
        search.append(np.array([influence_vec]).sum()/100.0)
fp.close()

movie_feats=np.array(movie_feats)
money=np.array([money])
search=np.array([search])
movie_feats=np.concatenate((movie_feats,search.T),axis=1)
# Normalization
for i in range(movie_feats.shape[1]):
    m1=movie_feats[:,i].mean()
    m2=movie_feats[:,i].std()
    movie_feats[:,i]=(movie_feats[:,i]-m1)/m2
# Save to the file
data=np.concatenate((movie_feats,money.T),axis=1)
np.savetxt("D:/spark/movie/result3.txt",data,fmt="%.8f")


