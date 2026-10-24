"""Models fit ONLY on the caller's training-window interactions and metadata.

Prediction candidates contain IDs only. Future schedule metadata is never used
to fit content features. Repeats are allowed consistently across all methods.
"""
import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from als import ImplicitALS

SEED=2026
ATTRIBUTES=['tv_show_category','tv_show_genre_1','tv_show_genre_2',
            'tv_show_genre_3','year_of_production','director','actors']

def content_tokens(row):
    """Field-qualified attribute tokens; no item ID or future data feature."""
    tokens=[]
    for field in ATTRIBUTES:
        value=row.get(field)
        if pd.isna(value) or str(value).strip() in ('','0','nan','None'):
            continue
        # Preserve full actor/director names; split common list delimiters only.
        values=str(value).lower().replace(';',',').split(',')
        for value in values:
            value='_'.join(value.strip().split())
            if value:
                tokens.append(field+'='+value)
    return tokens

class TrainingBundle:
    def __init__(self,interactions,metadata,users):
        # interactions represent qualified distinct airings, aggregated by user-item.
        self.users=list(users)
        self.uindex={u:i for i,u in enumerate(self.users)}
        self.items=sorted(set(interactions.tv_show_id)-{0})
        if not self.items:
            raise ValueError('No training interactions')
        self.iindex={x:i for i,x in enumerate(self.items)}
        counts=interactions.groupby(['user_id','tv_show_id']).size().rename('count').reset_index()
        counts=counts[counts.user_id.isin(self.uindex)]
        self.counts=csr_matrix((counts['count'].astype(np.float32),
             (counts.user_id.map(self.uindex),counts.tv_show_id.map(self.iindex))),
             shape=(len(self.users),len(self.items)))
        self.weights=self.counts.copy()
        self.weights.data=np.log1p(self.weights.data)
        self.popularity=np.asarray((self.counts>0).sum(axis=0)).ravel()
        self.metadata=metadata.drop_duplicates('tv_show_id').set_index('tv_show_id')
        self.cold=np.asarray(self.counts.sum(axis=1)).ravel()==0

    def candidate_map(self,candidates):
        c=np.array(sorted(set(map(int,candidates))-{0}),dtype=np.int64)
        if len(c)<5:
            raise ValueError('Fewer than five candidates')
        ci=np.array([self.iindex.get(int(x),-1) for x in c])
        pop=np.array([self.popularity[i] if i>=0 else 0 for i in ci])
        return c,ci,pop

    def rank(self,scores,candidates):
        """Positive scores first; fill missing/zero/negative slots with popularity.

        Exact ties use training popularity then integer ID. Identical fallback
        for every model; unseen items have popularity=0 then ID tie-break.
        """
        c,_,pop=self.candidate_map(candidates)
        if scores.shape != (len(self.users),len(c)):
            raise ValueError('Unexpected score matrix shape')
        fallback=np.lexsort((c,-pop))
        rankings={}
        for ui,user in enumerate(self.users):
            s=scores[ui]
            if self.cold[ui]:
                indices=fallback[:5]
            else:
                order=np.lexsort((c,-pop,-np.nan_to_num(s,nan=-np.inf)))
                chosen=order[np.isfinite(s[order]) & (s[order]>0)][:5].tolist()
                for j in fallback:
                    if len(chosen)==5:
                        break
                    if int(j) not in chosen:
                        chosen.append(int(j))
                indices=chosen
            rankings[user]=[int(c[i]) for i in indices]
        return rankings

    def popularity_scores(self,candidates):
        c,ci,pop=self.candidate_map(candidates)
        return np.broadcast_to(pop,(len(self.users),len(c))).copy()

    def content_scores(self,candidates):
        # Vocabulary/IDF fit on train-only item attributes, never hold-out metadata.
        docs=[content_tokens(row) for _,row in self.metadata.iterrows()]
        vectorizer=TfidfVectorizer(analyzer=lambda x:x,norm='l2',dtype=np.float32)
        try:
            vectorizer.fit(docs)
            features=vectorizer.transform([content_tokens(self.metadata.loc[x]) if x in self.metadata.index else [] for x in self.items])
        except ValueError as error:
            if 'empty vocabulary' not in str(error):
                raise
            return np.zeros((len(self.users),len(self.candidate_map(candidates)[0])),np.float32)
        profiles=normalize(self.weights@features)
        c,ci,_=self.candidate_map(candidates)
        scores=np.zeros((len(self.users),len(c)),np.float32)
        candidate_features=vectorizer.transform([content_tokens(self.metadata.loc[int(x)]) if int(x) in self.metadata.index else [] for x in c])
        scores=(profiles@candidate_features.T).toarray()
        self.content_vectorizer=vectorizer
        return scores

    def als_scores(self,candidates,factors=32,regularization=.1,alpha=20,iterations=20):
        # implicit expects OBSERVED confidence values, missing entries mean c=1.
        confidence=self.weights.copy()
        confidence.data=1+alpha*confidence.data
        model=ImplicitALS(factors=factors,regularization=regularization,
             iterations=iterations,random_state=SEED,num_threads=2)
        model.fit(confidence)
        c,ci,_=self.candidate_map(candidates)
        scores=np.zeros((len(self.users),len(c)),np.float32)
        known=ci>=0
        scores[:,known]=model.user_factors@model.item_factors[ci[known]].T
        self.als_model=model
        return scores

def reciprocal_rank_fusion(rankings_a,rankings_b,weight=.5,offset=60):
    """Fuse full candidate rankings supplied by caller; evaluate hybrid separately."""
    if set(rankings_a)!=set(rankings_b):
        raise ValueError('Cohort mismatch')
    result={}
    for user in rankings_a:
        score={}
        for ranking,w in [(rankings_a[user],weight),(rankings_b[user],1-weight)]:
            for rank,item in enumerate(ranking,1):
                score[item]=score.get(item,0)+w/(offset+rank)
        result[user]=sorted(score,key=lambda x:(-score[x],x))[:5]
    return result
