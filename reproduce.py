"""Exact prediction-stability experiments: python reproduce.py."""
from fractions import Fraction as F
from bisect import bisect_left
from collections import Counter
from itertools import product
from pathlib import Path
import json
import io
from certify import certify, uniform, write_csv, display_bound

ROOT=Path(__file__).resolve().parent

# Petal-width feature and class index for all 150 records in the corrected
# scikit-learn Iris dataset, encoded in tenths (50 records per class).
# Source: R. A. Fisher, Iris, UCI ML Repository, DOI 10.24432/C56C76.
# License: Creative Commons Attribution 4.0 International (CC BY 4.0).
IRIS_TENTHS=(
    '0202020202040302020102020101020404030303020402050202040202020204010202020201020203030206040302020202',
    '1415151315131610131410151014131415101511181315121314141715101110121615161513131312141210131213131113',
    '2519211822211718182520192120242318222315232020182118181821161920221514232418182124231923252319202318',
)


def iris_data():
    rows=[]
    for label,encoded in enumerate(IRIS_TENTHS):
        assert len(encoded)==100
        for i in range(0,len(encoded),2):
            width=F(int(encoded[i:i+2]),10)
            rows.append(((width-F(1,10))/F(12,5),label))
    return rows


class Step:
    def __init__(self, cuts, labels, e0=None, e1=None):
        self.cuts=list(map(F,cuts))
        self.labels=list(labels)
        assert len(self.labels)==len(self.cuts)+1
        assert all(0<t<1 for t in self.cuts)
        assert all(a<b for a,b in zip(self.cuts,self.cuts[1:]))
        self.e0=self.labels[0] if e0 is None else e0
        self.e1=self.labels[-1] if e1 is None else e1

    def value(self,x):
        if x==0:return self.e0
        if x==1:return self.e1
        return self.labels[bisect_left(self.cuts,x)]

    def oracle(self,a,b):
        vals=self.labels[bisect_left(self.cuts,a):bisect_left(self.cuts,b)+1]
        if a==0:vals=vals+[self.e0]
        if b==1:vals=vals+[self.e1]
        return F(min(vals)),F(max(vals))

    def integral(self):
        edges=[F(0)]+self.cuts+[F(1)]
        return sum(F(c)*(b-a) for a,b,c in zip(edges,edges[1:],self.labels))

    def canonical(self):
        cuts,labels=[],[self.labels[0]]
        for t,c in zip(self.cuts,self.labels[1:]):
            if c!=labels[-1]:cuts.append(t);labels.append(c)
        return Step(cuts,labels,self.e0,self.e1)

    def boundaries(self):
        c=self.canonical()
        return c.cuts+([F(0)] if c.e0!=c.labels[0] else [])+([F(1)] if c.e1!=c.labels[-1] else [])

    def occupancy(self,m):
        N=2**m
        return len({min(int(t*N),N-1) for t in self.boundaries()})

    def uniform_cells(self,eps):
        for m in range(100):
            if F(self.occupancy(m),2**(m+1))<=eps:return 2**m
        raise RuntimeError('Depth limit exceeded')


def response(tree,x,delta):
    a,b=x-delta,x+delta
    assert 0<=a<b<=1
    y=tree.value(x)
    cuts=[(t-a)/(2*delta) for t in tree.cuts if a<t<b]
    edges=[F(0)]+cuts+[F(1)]
    labels=[int(tree.value(a+2*delta*(u+v)/2)!=y) for u,v in zip(edges,edges[1:])]
    g=Step(cuts,labels,int(tree.value(a)!=y),int(tree.value(b)!=y)).canonical()
    raw=[F(0)]+tree.cuts+[F(1)]
    ref=sum(max(F(0),min(v,b)-max(u,a)) for u,v,c in zip(raw,raw[1:],tree.labels) if c!=y)/(2*delta)
    assert ref==g.integral()
    return g,ref


def disagreement(first,second):
    cuts=sorted(set(first.cuts+second.cuts))
    edges=[F(0)]+cuts+[F(1)]
    g=Step(cuts,[int(first.value((a+b)/2)!=second.value((a+b)/2))
                 for a,b in zip(edges,edges[1:])],
           int(first.value(F(0))!=second.value(F(0))),
           int(first.value(F(1))!=second.value(F(1)))).canonical()
    a=[F(0)]+first.cuts+[F(1)]
    b=[F(0)]+second.cuts+[F(1)]
    ref=sum(max(F(0),min(v,z)-max(u,w)) for u,v,c in zip(a,a[1:],first.labels)
            for w,z,d in zip(b,b[1:],second.labels) if c!=d)
    assert ref==g.integral()
    return g,ref


def train(data,max_depth=4,min_leaf=4):
    def impurity(rows):
        counts=Counter(y for x,y in rows)
        return F(len(rows))-sum(F(n*n,len(rows)) for n in counts.values())
    def fit(rows,depth):
        counts=Counter(y for x,y in rows)
        label=min(counts,key=lambda y:(-counts[y],y))
        if depth>=max_depth or len(counts)==1:return ('leaf',label)
        xs=sorted(set(x for x,y in rows))
        best,gain=None,F(0)
        for a,b in zip(xs,xs[1:]):
            t=(a+b)/2
            left=[r for r in rows if r[0]<=t]
            right=[r for r in rows if r[0]>t]
            if min(len(left),len(right))<min_leaf:continue
            g=impurity(rows)-impurity(left)-impurity(right)
            if g>gain:best=(t,left,right);gain=g
        if best is None:return ('leaf',label)
        t,left,right=best
        return ('node',t,fit(left,depth+1),fit(right,depth+1))
    nested=fit(data,0)
    def flatten(node):
        if node[0]=='leaf':return [],[node[1]]
        _,t,left,right=node
        lc,ll=flatten(left);rc,rl=flatten(right)
        return lc+[t]+rc,ll+rl
    cuts,labels=flatten(nested)
    return Step(cuts,labels),nested


class Bands:
    def __init__(self,bands,labels):
        self.bands=[tuple(map(F,band)) for band in bands]
        self.labels=list(labels)
        assert len(labels)==len(bands)+1
        assert all(0<l<=r<1 for l,r in self.bands)
        assert all(r<l for (_,r),(l,_) in zip(self.bands,self.bands[1:]))

    def active(self,y):
        c=[int(v!=y) for v in self.labels]
        bands,labels=[],[c[0]]
        for band,v in zip(self.bands,c[1:]):
            if v!=labels[-1]:bands.append(band);labels.append(v)
        return Bands(bands,labels)

    def oracle(self,a,b):
        if any(a<=r and l<=b for l,r in self.bands):return F(0),F(1)
        k=sum(r<a for l,r in self.bands)
        return (F(self.labels[k]),)*2

    def freeze(self,a,b):
        return any(l<=a and b<=r for l,r in self.bands)

    def hull(self):
        lower=upper=F(self.labels[-1])
        for i,(l,r) in enumerate(self.bands,1):
            a=self.labels[i]-self.labels[i-1]
            lower-=max(a*l,a*r)
            upper-=min(a*l,a*r)
        return lower,upper

    def components(self,q):
        for j,(l,r) in enumerate(self.bands):
            if l<=q<=r:
                upper=self.bands.copy();upper[j]=(q,r)
                out=[(Bands(upper,self.labels),self.labels[j],'>=')]
                if l<q:
                    lower=self.bands.copy();lower[j]=(l,q)
                    out.append((Bands(lower,self.labels),self.labels[j+1],'<'))
                return out
        k=sum(r<q for l,r in self.bands)
        return [(self,self.labels[k],'all')]


def raw_fixed_probability(labels,theta,y):
    edges=[F(0)]+list(theta)+[F(1)]
    return sum(b-a for a,b,c in zip(edges,edges[1:],labels) if c!=y)


def verify_family(family,q=F(1,2)):
    components=family.components(q)
    all_extrema=[]
    for box,y,side in components:
        active=box.active(y)
        vals=[raw_fixed_probability(box.labels,theta,y) for theta in product(*box.bands)]
        assert (min(vals),max(vals))==active.hull()
        all_extrema.extend(vals)
    # Independent nominal-class branches from threshold samples and one-sided limits.
    candidates=[]
    crossing=[i for i,(l,r) in enumerate(family.bands) if l<=q<=r]
    grids=[sorted(set([l,r]+([q] if l<=q<=r else []))) for l,r in family.bands]
    for theta in product(*grids):
        tree=Step(theta,family.labels)
        candidates.append(raw_fixed_probability(family.labels,theta,tree.value(q)))
        if crossing:
            j=crossing[0]
            if theta[j]==q and family.bands[j][0]<q:
                candidates.append(raw_fixed_probability(family.labels,theta,family.labels[j+1]))
    assert (min(all_extrema),max(all_extrema))==(min(candidates),max(candidates))
    return min(all_extrema),max(all_extrema)


def save_figure(fig, path):
    buffer = io.BytesIO()
    fig.savefig(buffer, format='pdf')
    data = buffer.getvalue()
    assert data.startswith(b'%PDF') and b'%%EOF' in data[-64:]
    temporary = path.with_suffix('.tmp')
    temporary.write_bytes(data)
    temporary.replace(path)


def main():
    (ROOT/'data').mkdir(exist_ok=True)
    (ROOT/'figures').mkdir(exist_ok=True)
    data=iris_data()
    iris,nested=train(data)
    model={'cuts':list(map(str,iris.cuts)),'labels':iris.labels,'nested':nested}
    (ROOT/'data/iris_model.json').write_text(json.dumps(model,default=str,indent=2)+'\n')
    print('Iris:',model,flush=True)
    iris_g,iris_p=response(iris,F(2,3),F(1,8))
    assert iris_p==F(5,12)
    sep=Step([F(k,13) for k in range(1,13)],[k%2 for k in range(13)])
    clustered=Step([F(37,100)+F(k,100000) for k in range(12)],[k%2 for k in range(13)])
    multi=Step([F(1,5),F(2,5),F(3,5),F(4,5)],[0,1,2,1,0])
    mg,mp=response(multi,F(1,2),F(1,2))
    dg,dp=disagreement(Step([F(1,3),F(2,3)],[0,1,0]),Step([F(2,5),F(3,4)],[0,1,0]))
    ordinary=[('Single threshold',Step([F(7,10)],[0,1]),F(3,10)),
              ('Narrow band',Step([F(3699,10000),F(3701,10000)],[0,1,0]),F(1,5000)),
              ('12 separated',sep,sep.integral()),('12 clustered',clustered,clustered.integral()),
              ('Multiclass cancellation',mg,mp),('Iris boundary',iris_g,iris_p),
              ('Disagreement',dg,dp),('Endpoint',Step([], [0],e0=1),F(0)),
              ('Constant',Step([],[0]),F(0))]
    rows=[]
    for name,g,ref in ordinary:
        for power in range(4,19,2):
            eps=F(1,2**power)
            r=certify(g.oracle,eps)
            assert r['lower']<=ref<=r['upper']
            rows.append(dict(model=name,tolerance=eps,reference=ref,**r,uniform_cells=g.uniform_cells(eps)))
            if power==14:print(name, r['cells'],r['queries'],g.uniform_cells(eps),
                              display_bound(r['lower']),display_bound(r['upper'],upper=True),flush=True)
    write_csv(ROOT/'data/ordinary.csv',rows)
    families=[('Single band',Bands([(F(1,6),F(1,3))],[1,0]),range(5,16,2)),
              ('Four bands',Bands([(F(i,5)-F(1,1000),F(i,5)+F(1,1000)) for i in range(1,5)],
                                  [0,1,0,1,0]),range(6,17,2))]
    uncertain=[]
    for name,family,powers in families:
        lower,upper=family.hull();rho=(upper-lower)/2
        for power in powers:
            eta=F(1,2**power)
            for method,freeze in [('ordinary',None),('frozen',family.freeze)]:
                r=certify(family.oracle,rho+eta,freeze)
                assert r['lower']<=lower<=upper<=r['upper']
                uncertain.append(dict(model=name,method=method,excess=eta,exact_lower=lower,
                                      exact_upper=upper,**r))
                if power==list(powers)[-1]:print(name,method,r['queries'],flush=True)
    write_csv(ROOT/'data/uncertain.csv',uncertain)
    nominal=[('Binary crossing',Bands([(F(2,5),F(3,5))],[0,1])),
             ('Multiclass endpoint',Bands([(F(1,5),F(1,4)),(F(9,20),F(1,2))],[0,1,2])),
             ('Multiclass crossing',Bands([(F(1,10),F(3,20)),(F(2,5),F(3,5)),(F(3,4),F(4,5))],[0,1,2,1]))]
    joint,component_rows=[],[]
    for name,family in nominal:
        low,high=verify_family(family)
        for power in range(6,17,2):
            eta=F(1,2**power);parts=[]
            for h,(box,y,side) in enumerate(family.components(F(1,2))):
                active=box.active(y);l,u=active.hull()
                r=certify(active.oracle,(u-l)/2+eta/2,active.freeze)
                assert r['lower']<=l<=u<=r['upper']
                parts.append(r)
                component_rows.append(dict(model=name,component=h,side=side,excess=eta,**r))
            A=min(r['lower'] for r in parts);B=max(r['upper'] for r in parts)
            assert A<=low<=high<=B and (B-A)/2<=(high-low)/2+eta
            joint.append(dict(model=name,excess=eta,exact_lower=low,exact_upper=high,
                              lower=A,upper=B,radius=(B-A)/2,queries=sum(r['queries'] for r in parts)))
            if power==14:print(name,display_bound(A,6),display_bound(B,6,True),joint[-1]['queries'],flush=True)
    write_csv(ROOT/'data/nominal.csv',joint)
    write_csv(ROOT/'data/components.csv',component_rows)
    checks={'ordinary_certificates':len(rows),'fixed_reference_certificates':len(uncertain),
            'joint_certificates':len(joint),'component_certificates':len(component_rows),'range_checks':0,
            'occupancy_checks':0,'conditional_corner_checks':len(nominal)}
    for _,g,ref in ordinary:
        for m in range(8):
            N=2**m;ambiguous=0
            for j in range(N):
                a,b=F(j,N),F(j+1,N)
                pts=sorted(set([a,b]+[t for t in g.cuts if a<t<b]))
                vals={g.value(v) for v in pts}
                vals.update(g.value((u+v)/2) for u,v in zip(pts,pts[1:]))
                assert g.oracle(a,b)==(min(vals),max(vals))
                ambiguous+=min(vals)!=max(vals)
                checks['range_checks']+=1
            assert ambiguous==g.occupancy(m)
            checks['occupancy_checks']+=1
    # Degenerate bands and nominal points at either band endpoint.
    for band in [(F(2,5),F(1,2)),(F(1,2),F(3,5)),(F(1,2),F(1,2))]:
        verify_family(Bands([band],[0,1]))
        checks['conditional_corner_checks']+=1
    (ROOT/'data/validation.json').write_text(json.dumps(checks,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'serif','font.size':9,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(7,2.6),layout='constrained')
    for g,label,color,style in [(sep,'12 separated','#245775','-'),(clustered,'12 clustered','#9a4e2f','--')]:
        axes[0].plot(range(23),[g.occupancy(m) for m in range(23)],style,label=label,color=color)
    axes[0].set(xlabel='Dyadic depth',ylabel='Ambiguous cells',yticks=[0,4,8,12]);axes[0].legend(frameon=False)
    edges=[F(0)]+iris.cuts+[F(1)]
    for a,b,y in zip(edges,edges[1:],iris.labels):
        axes[1].plot([float(a),float(b)],[y,y],color='#245775',lw=1.7)
    axes[1].axvspan(float(F(2,3)-F(1,8)),float(F(2,3)+F(1,8)),alpha=.15,color='#9a4e2f',label='Noise interval')
    axes[1].axvline(2/3,color='#9a4e2f',linestyle='--',label='Nominal input')
    axes[1].set(xlabel='Normalized petal width',ylabel='Predicted class',yticks=[0,1,2]);axes[1].legend(frameon=False,fontsize=8)
    save_figure(fig, ROOT/'figures/boundary_geometry.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(7,2.7),layout='constrained')
    for method,color,style in [('ordinary','#9a4e2f','o-'),('frozen','#245775','s-')]:
        rr=[r for r in uncertain if r['model']=='Single band' and r['method']==method]
        axes[0].loglog([float(r['excess']) for r in rr],[r['queries'] for r in rr],style,
                       color=color,markersize=3,label=method.capitalize())
    axes[0].invert_xaxis();axes[0].set(xlabel=r'Excess budget $\eta$',ylabel='Range queries');axes[0].legend(frameon=False)
    rr=[r for r in uncertain if r['model']=='Four bands' and r['method']=='frozen']
    eta=[float(r['excess']) for r in rr]
    axes[1].axhspan(.396,.404,color='#245775',alpha=.13,label='Exact model range')
    axes[1].semilogx(eta,[float(r['lower']) for r in rr],'o-',color='#245775',markersize=3,label='Certified endpoints')
    axes[1].semilogx(eta,[float(r['upper']) for r in rr],'o-',color='#245775',markersize=3)
    axes[1].invert_xaxis();axes[1].set(xlabel=r'Excess budget $\eta$',ylabel='Probability');axes[1].legend(frameon=False,fontsize=8)
    save_figure(fig, ROOT/'figures/uncertainty.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(7,2.6),layout='constrained')
    axes[0].plot([.4,.5,.6],[.4,.5,.4],color='#245775')
    axes[0].axhspan(.4,.5,color='#245775',alpha=.1)
    axes[0].set(xlabel=r'Threshold $\theta$',ylabel='Prediction-change probability')
    axes[1].plot([.45,.5],[.45,.5],color='#245775')
    axes[1].plot(.5,.5,'o',mfc='white',mec='#245775',label='Limit from below')
    axes[1].plot([.5,.5],[.7,.75],color='#9a4e2f',lw=3,label=r'Range at $\theta_2=1/2$')
    axes[1].set(xlabel=r'Threshold $\theta_2$',ylabel='Prediction-change probability',ylim=(.43,.77),xlim=(.448,.503))
    axes[1].legend(frameon=False,fontsize=8)
    save_figure(fig, ROOT/'figures/nominal.pdf');plt.close(fig)
    print('Validation:',checks)


if __name__=='__main__':
    main()
