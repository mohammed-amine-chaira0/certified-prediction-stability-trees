from fractions import Fraction as F
import heapq


def certify(oracle, tolerance, freeze=None, keep_leaves=False):
    tolerance = F(tolerance)
    if tolerance <= 0:
        raise ValueError('The tolerance must be positive.')
    lo, hi = map(F, oracle(F(0), F(1)))
    A, B, count = lo, hi, 1
    heap, leaves = [], {}

    def add(a, b, l, u, depth):
        if keep_leaves:
            leaves[a] = (b, l, u, depth)
        e = (b-a)*(u-l)/2
        if e and not (freeze and freeze(a, b)):
            heapq.heappush(heap, (-e, depth, a, b, l, u))

    add(F(0), F(1), lo, hi, 0)
    splits = 0
    while (B-A)/2 > tolerance:
        if not heap:
            raise ValueError('Requested radius is below the frozen certificate.')
        _, depth, a, b, l, u = heapq.heappop(heap)
        if keep_leaves:
            del leaves[a]
        mid = (a+b)/2
        A -= (b-a)*l
        B -= (b-a)*u
        for x, y in [(a, mid), (mid, b)]:
            raw_l, raw_u = map(F, oracle(x, y))
            cl, cu = max(l, raw_l), min(u, raw_u)
            assert cl <= cu
            A += (y-x)*cl
            B += (y-x)*cu
            add(x, y, cl, cu, depth+1)
        count += 2
        splits += 1
    assert count == 2*(splits+1)-1
    return dict(lower=A, upper=B, radius=(B-A)/2,
                cells=splits+1, queries=count, leaves=leaves)


def uniform(oracle, depth, regular=None, background=False):
    """Full-grid sums, aggregating regular dyadic blocks exactly."""
    h = F(1, 2**depth)
    calls = 0

    def visit(a, b, level, inherited=None):
        nonlocal calls
        l, u = map(F, oracle(a, b))
        calls += 1
        if inherited:
            l, u = max(l, inherited[0]), min(u, inherited[1])
        if level == depth or l == u:
            return (b-a)*l, (b-a)*u
        if regular and regular(a, b):
            if background:
                n = 2**(depth-level)
                c = l-a*a/8
                lower = c*(b-a) + h*(n*a*a+a*h*n*(n-1)
                                      +h*h*n*(n-1)*(2*n-1)/6)/8
                return lower, lower+h*(b*b-a*a)/8
            return (b-a)*l, (b-a)*u
        mid = (a+b)/2
        la, ua = visit(a, mid, level+1, (l, u))
        lb, ub = visit(mid, b, level+1, (l, u))
        return la+lb, ua+ub

    A, B = visit(F(0), F(1), 0)
    return dict(lower=A, upper=B, radius=(B-A)/2,
                cells=2**depth, queries=calls)


def write_csv(path, rows):
    import csv
    rows = [{k: str(v) for k, v in r.items() if k != 'leaves'} for r in rows]
    if rows:
        with open(path, 'w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def display_bound(x, places=7, upper=False):
    scale = 10**places
    n = -((-x.numerator*scale)//x.denominator) if upper else x.numerator*scale//x.denominator
    sign = '-' if n < 0 else ''
    n = abs(n)
    return f'{sign}{n//scale}.{n%scale:0{places}d}'
