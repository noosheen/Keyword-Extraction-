from sklearn.feature_extraction.text import TfidfTransformer
import joblib
import re

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn):
    
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        fname = feature_names[idx]
        
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results

def pre_process(text):
    
    #remove tags
    text=re.sub("</?.*?>"," <> ",text)
    
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    
    return text




doc=open('SiteData.txt','r',encoding='utf-8').read()
doc=pre_process(doc)
vec=joblib.load('KeyWordvectorizer0.pkl.bz2')
wcv=joblib.load('KeyWordvectors0.pkl.bz2')
s1=open("stopwords.txt",'r',encoding='utf-8')
st=s1.readlines()

tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(wcv)
feature_names=vec.get_feature_names()

#generate tf-idf for the given document
tf_idf_vector=tfidf_transformer.transform(vec.transform([doc]))

#sort the tf-idf vectors by descending order of scores
sorted_items=sort_coo(tf_idf_vector.tocoo())

#extract only the top n
keywords=extract_topn_from_vector(feature_names,sorted_items,50)
