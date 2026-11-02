"""Clearly marked SYNTHETIC fixture: for software tests, never for real findings."""
import numpy as np
import pandas as pd


def synthetic_fixture(n=600, seed=42):
    rng = np.random.default_rng(seed)
    job = rng.choice(['admin.','student','technician','management'], n)
    ed = rng.choice(['high.school','university.degree','basic.9y'], n)
    contact = rng.choice(['cellular','telephone'], n)
    previous = rng.poisson(.35, n)
    euribor = rng.normal(3.8, .9, n)
    signal = -1.9 + .65*(previous>0) + .5*(contact=='cellular') - .35*(euribor-3.8)
    # Deliberately non-perfect classifier; synthetic drift in final period.
    signal += (np.arange(n) > int(n*.8)) * -.5
    prob = 1/(1+np.exp(-signal))
    y = np.where(rng.random(n) < prob, 'yes','no')
    df = pd.DataFrame({
       'age': rng.integers(19,78,n), 'job':job,
       'marital':rng.choice(['single','married','divorced'],n),
       'education':ed,'default':rng.choice(['no','unknown'],n),
       'housing':rng.choice(['yes','no'],n),'loan':rng.choice(['yes','no'],n),
       'contact':contact,'month':rng.choice(['may','jun','jul','aug'],n),
       'day_of_week':rng.choice(['mon','tue','wed','thu','fri'],n),
       'duration':rng.integers(1,1000,n), 'campaign':rng.integers(1,6,n),
       'pdays':rng.choice([999,999,999,3,6],n), 'previous':previous,
       'poutcome':rng.choice(['nonexistent','failure','success'],n),
       'emp.var.rate':rng.normal(.3,1.8,n),
       'cons.price.idx':rng.normal(93.5,.4,n),
       'cons.conf.idx':rng.normal(-40,5,n),
       'euribor3m':euribor, 'nr.employed':rng.normal(5150,50,n), 'y':y
    })
    return df
