"""The five combinatorial conditions on the two printed face lists.

A triangulation here is a finite vertex set V together with a set F of
positively ordered triples subject to

  (a) the triples consist of distinct vertices and are pairwise distinct as sets,
  (b) every edge lies in exactly two faces, traversed once in each direction by
      their cycles,
  (c) the faces at a vertex v, each written (v, x, y), give a successor map
      x -> y that is a single cycle on the neighbours of v,
  (d) |F| = 2|V|,
  (e) the list is connected: any two faces are joined by a chain of faces in
      which consecutive ones share an edge.

Condition (e) is a hypothesis, not a sanity check: it makes the dual graph
connected, which is what licenses the spanning tree of the development, and the
surface is connected because every vertex lies in a face, which is what lets the
Euler characteristic decide the topological type.  That the list is a torus is a
conclusion of the five conditions, not a hypothesis.

Checked here on the two face lists T and T' of data.py, together with the
derived edge set against the frozen one.  Integer combinatorics throughout; no
coordinate is read.
"""
import sys

from data import FACES_SQUARE, EDGES_SQUARE, FACES_HEX, EDGES_HEX

CHECKS = []


def check(ok, label, value=''):
    """One aligned PASS/FAIL line, collected for the closing summary."""
    CHECKS.append(bool(ok))
    print(f'  {"PASS" if ok else "FAIL"}  {label:<46}{value}'.rstrip())
    return ok


def run(name, faces, frozen_edges, vertices=range(8)):
    print(f'\n{name}')
    V = sorted(vertices)
    n0 = len(CHECKS)

    # (a) distinct vertices in each face, faces pairwise distinct as sets
    sets = [frozenset(f) for f in faces]
    check(all(len(set(f)) == 3 for f in faces) and len(set(sets)) == len(faces),
          '(a) triples distinct, faces distinct as sets', f'{len(faces)} faces')

    # (b) each edge in exactly two faces, once in each direction
    directed = {}
    for f in faces:
        for k in range(3):
            d = (f[k], f[(k + 1) % 3])
            directed[d] = directed.get(d, 0) + 1
    undirected = {}
    for (a, b) in directed:
        e = (min(a, b), max(a, b))
        undirected[e] = undirected.get(e, 0) + 1
    check(all(c == 1 for c in directed.values())
          and all(c == 2 for c in undirected.values()),
          '(b) each edge in two faces, once each way', f'|E| = {len(undirected)}')

    # (c) the faces at each vertex form a single cycle on its neighbours
    degrees = {}
    ok_cycles = True
    for v in V:
        succ = {}
        for f in faces:
            if v not in f:
                continue
            i = f.index(v)
            x, y = f[(i + 1) % 3], f[(i + 2) % 3]
            if x in succ:
                ok_cycles = False
            succ[x] = y
        if not succ:
            ok_cycles = False
            continue
        start = next(iter(succ))
        seen, cur = [start], succ[start]
        while cur != start:
            if cur in seen or cur not in succ:
                ok_cycles = False
                break
            seen.append(cur)
            cur = succ[cur]
        if len(seen) != len(succ):
            ok_cycles = False
        degrees[v] = len(succ)
    check(ok_cycles, '(c) every vertex star a single cycle',
          f'degrees {tuple(degrees[v] for v in V)}')

    # (d) |F| = 2|V|
    check(len(faces) == 2 * len(V), '(d) |F| = 2|V|',
          f'{len(faces)} = 2 * {len(V)}')

    # the derived edge set against the frozen one
    check(sorted(undirected) == sorted(frozen_edges),
          'derived edge set equals the frozen list',
          f'{len(frozen_edges)} edges')

    # (e) dual graph connected, and every vertex in a face
    dual = {i: set() for i in range(len(faces))}
    by_edge = {}
    for i, f in enumerate(faces):
        for k in range(3):
            e = (min(f[k], f[(k + 1) % 3]), max(f[k], f[(k + 1) % 3]))
            by_edge.setdefault(e, []).append(i)
    for e, fs in by_edge.items():
        for i in fs:
            for j in fs:
                if i != j:
                    dual[i].add(j)
    comp_f, frontier = {0}, [0]
    while frontier:
        i = frontier.pop()
        for j in dual[i] - comp_f:
            comp_f.add(j)
            frontier.append(j)
    covered = set().union(*[set(f) for f in faces]) == set(V)
    check(len(comp_f) == len(faces) and covered,
          '(e) dual graph connected, all vertices used',
          f'{len(comp_f)} faces reached')

    return all(CHECKS[n0:])


def main():
    print('combinatorics   the five list conditions, both face lists')
    run('T, the square triangulation', FACES_SQUARE, EDGES_SQUARE)
    run("T', the hexagonal triangulation", FACES_HEX, EDGES_HEX)
    return all(CHECKS), len(CHECKS)


if __name__ == '__main__':
    sys.exit(0 if main()[0] else 1)
