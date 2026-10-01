import numpy as np, scipy.stats as st
# t-test example
m1,s1,m2,s2,n=12.4,7.0,8.1,7.2,30
sp=np.sqrt(((n-1)*s1**2+(n-1)*s2**2)/(2*n-2)); se=sp*np.sqrt(2/n); t=(m1-m2)/se; df=2*n-2
p=2*st.t.sf(t,df); tc=st.t.ppf(.975,df)
print('sp',sp,'se',se,'t',t,'p',p,'tcrit',tc,'CI',(m1-m2)-tc*se,(m1-m2)+tc*se)
# power
sig=7; d=4; se2=sig*np.sqrt(2/30); ncp=d/se2
print('power n30 (z approx)', st.norm.sf(1.96-ncp)+st.norm.cdf(-1.96-ncp), 'ncp',ncp)
print('power n30 (t exact)', st.nct.sf(st.t.ppf(.975,58),58,ncp)+st.nct.cdf(-st.t.ppf(.975,58),58,ncp))
nreq=2*(st.norm.ppf(.975)+st.norm.ppf(.8))**2*sig**2/d**2; print('n per group',nreq, nreq/0.9)
# big data
se3=15*np.sqrt(2/250000); z=0.3/se3; print('big z',z,'p',2*st.norm.sf(z))
# FWER
for k in [1,2,3,5,10,20]: print(k, 1-0.95**k)
# multiple testing
p=np.array([0.004,0.012,0.021,0.038,0.260]); m=len(p)
bon=np.minimum(p*m,1)
o=np.argsort(p); ps=p[o]
holm=np.maximum.accumulate(np.minimum(ps*(m-np.arange(m)),1))
bh=np.minimum.accumulate((ps*m/np.arange(1,m+1))[::-1])[::-1]
print('bon',bon,'holm',holm,'bh',bh)
# t crit
for d in [2,5,10,30,60,120]: print('tcrit',d,st.t.ppf(.975,d))
print('chi1',st.chi2.ppf(.95,1),'chi2',st.chi2.ppf(.95,2),'chi4',st.chi2.ppf(.95,4),'F2_57',st.f.ppf(.95,2,57),'F3_60',st.f.ppf(.95,3,60))
# SMD table1
def smdc(a,b,c,d): return (a-c)/np.sqrt((b**2+d**2)/2)
def smdp(p1,p2): return (p1-p2)/np.sqrt((p1*(1-p1)+p2*(1-p2))/2)
print('age',smdc(58.3,10.2,63.1,11.4),'hba1c',smdc(8.1,1.3,7.6,1.2),'female',smdp(.40,.47))
print('htn',smdp(.618,.702),'metformin',smdp(.853,.796),'insulin',smdp(.061,.098),'statin', smdp(.672,.655))
# logistic output z
b,se=0.642,0.211; z=b/se; print('z',z,'p',2*st.norm.sf(z),'wald',z*z,'OR',np.exp(b), np.exp(b-1.96*se), np.exp(b+1.96*se))
# lognormal pop
mu,sg=3,0.9; mean=np.exp(mu+sg**2/2); sd=mean*np.sqrt(np.exp(sg**2)-1); print('pop mean',mean,'median',np.exp(mu),'sd',sd,'se30',sd/np.sqrt(30),'se5',sd/np.sqrt(5))
# rank example
A=np.array([1,2,3,3,4,5,47]); B=np.array([6,7,8,9,10,11,12])  # ch01 사 절 표와 같은 자료
print('meanA',A.mean(),'meanB',B.mean(),'medA',np.median(A),'medB',np.median(B))
print('t',st.ttest_ind(A,B), 'welch', st.ttest_ind(A,B,equal_var=False))
print('MW',st.mannwhitneyu(A,B,alternative='two-sided',method='exact'), st.mannwhitneyu(A,B,alternative='two-sided'))
allv=np.concatenate([A,B]); r=st.rankdata(allv); print('ranks A',r[:7],'sum',r[:7].sum(),'ranks B',r[7:],'sum',r[7:].sum())
# 2/pi
print('2/pi',2/np.pi,'3/pi',3/np.pi)

# ---------------------------------------------------------------- 파이썬 출력 상자 (마 절)
# statsmodels 로지스틱 회귀 vs SAS PROC LOGISTIC. 가상 2×2 표(30일 재입원 × 흡연):
#   흡연자 229명 중 재입원 53명, 비흡연자 424명 중 재입원 58명 -> beta = 0.642, SE = 0.211
# Run: source /home/claude/pylibs/env.sh && python3 gen/nums_ch01.py
try:
    import pandas as pd, statsmodels.formula.api as smf
except ImportError:
    print("statsmodels not found: run with  source /home/claude/pylibs/env.sh")
else:
    lg = pd.DataFrame({"smoker": [1] * 229 + [0] * 424,
                       "readmit": [1] * 53 + [0] * 176 + [1] * 58 + [0] * 366})
    fit = smf.logit("readmit ~ smoker", data=lg).fit(disp=0)
    print(fit.summary())
    print(repr(fit.pvalues["smoker"]))
    b, se = fit.params["smoker"], fit.bse["smoker"]
    print("beta", b, "se", se, "z", b / se, "Wald chi2 (SAS)", (b / se) ** 2, "p", st.chi2.sf((b / se) ** 2, 1))
    print("OR", np.exp(b), "CI", np.exp(fit.conf_int().loc["smoker"].values), "2x2 OR", (53 * 366) / (176 * 58))

# ---------------------------------------------------------------- 2026-09-30 초보자용 개편에서 새로 들어간 숫자 (마·바·사 절)
# 마 절 'SD와 SE' 논문 상자, 쉽게 말하면 상자
print("SE drug", 7.0 / np.sqrt(30), "SE placebo", 7.2 / np.sqrt(30), "sqrt30", np.sqrt(30),
      "SE n300", 7.0 / np.sqrt(300), "mean±2SD", 12.4 - 2 * 7.0, 12.4 + 2 * 7.0)
# 바 절 비용 논문 상자와 수식으로 보기: 위젯 모집단 = lognormal(mu=3, sigma=0.9), 단위 만원
mu, sg = 3, 0.9
pm = np.exp(mu + sg**2 / 2); psd = pm * np.sqrt(np.exp(sg**2) - 1)
skew = (np.exp(sg**2) + 2) * np.sqrt(np.exp(sg**2) - 1)      # lognormal 왜도 = 4.745
print("cost pop mean (원)", pm * 1e4, "SD", psd * 1e4, "median", np.exp(mu) * 1e4)
print("SE n=1200 (원)", 336000 / np.sqrt(1200), "mean-2SD", 301000 - 2 * 336000)
print("skewness", skew, "skew of mean n=30/100/1200", [skew / np.sqrt(n) for n in (30, 100, 1200)])
# 사 절 표: IQR (numpy 기본 선형보간)
print("IQR A", np.percentile(A, [25, 75]), "IQR B", np.percentile(B, [25, 75]))
# 라 절 논문 상자 p-value
print("F(2,87)=4.12 p", st.f.sf(4.12, 2, 87), "chi2(1)=5.31 p", st.chi2.sf(5.31, 1))
