"""
Transmission Expansion Planning (TEP) with New-Bus Subnetworks
=============================================
The available Methods in this version are:
    - "VI"       : LSPC+Shortest-path big-M, PFCH+Separation cuts
    - "RandomVI" : LSPC+Shortest-path big-M, Randomly generated cuts
    - "SP"       : Shortest-path big-M, no cuts
    - "NVI"      : Naive (sum-of-all) big-M, no cuts
    - "LSPC"     : LSPC+Shortest-path big-M, no cuts

Usage
-----
Configure `tasks`, `method`, and instance settings in `main()`, then run.

Dependencies: numpy, pandas, gurobipy, pathlib
"""

import os
import copy
import heapq
import math
import random
import time
import collections
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

# ---------------------------------------------------------------------------
#%% Paths
# ---------------------------------------------------------------------------
In  = Path("")
Out = Path("output")
Out.mkdir(parents=True, exist_ok=True)


# ===========================================================================
#%% DATA
# ===========================================================================

def read_data(task):
    """Parse teh MATPOWER-style .dat file into four dictionaries."""
    bus_data, gend_data, branch_data, candidate_data = {}, {}, {}, {}

    with open(In / task["dataset"], 'r') as fh:
        lines = fh.read().splitlines()

    current_param = None
    column_order  = None

    for line in lines:
        if line.startswith("param:"):
            current_param = line.split()[1].strip()
            column_order  = line.split()[2:]
            continue

        if not line.strip():
            continue

        fields = line.split()
        if current_param and column_order and len(fields) >= 6:
            key       = int(fields[0])
            data_dict = {
                col: float(val) if "." in val else int(val)
                for col, val in zip(column_order, fields[1:])
            }
            if   current_param == "bus_num":   bus_data[key]       = data_dict
            elif current_param == "GEND:":     gend_data[key]      = data_dict
            elif current_param == "BRANCH:":   branch_data[key]    = data_dict
            elif current_param == "candidate": candidate_data[key] = data_dict

    return bus_data, branch_data, gend_data, candidate_data



# ===========================================================================
#%% LP RELAXATION
# ===========================================================================

def branch2cand(branch, branch_data, candidate_data):
    """Return the candidate_line id whose endpoints match branch_data[branch]."""
    fr = branch_data[branch]['branch_fbus']
    to = branch_data[branch]['branch_tbus']
    for bid, binfo in candidate_data.items():
        if (binfo['branch_fbus'] == fr and binfo['branch_tbus'] == to) or \
           (binfo['branch_fbus'] == to and binfo['branch_tbus'] == fr):
            return bid
    raise KeyError(f"No candidate found for branch {branch}")


def LR(cutpool, bus_data, branch_data, gend_data, candidate_data, dist, Sbase, wf):
    """
    Solve the LP relaxation of the DC-TEP.

    Returns
    -------
    p_res   : {cand_id: total_flow}
    p0_res  : {branch_id: flow}
    LMP     : {bus_id: shadow_price}
    sw      : {(cand_id, k): y_value}
    thetas  : {bus_id: angle}
    """
    modellp = gp.Model("relaxed")
    modellp.Params.OutputFlag = 0

    g, p0, p, theta, y = {}, {}, {}, {}, {}

    for gen_id, gen_info in gend_data.items():
        if gen_info["genD_status"]:
            g[gen_id] = modellp.addVar(
                vtype=GRB.CONTINUOUS, lb=0, ub=gen_info["genD_Pmax"],
                name=f"g_{gen_id}"
            )

    for branch_id, branch_info in candidate_data.items():
        for k in range(1, branch_info["max_lines"] + 1):
            y[branch_id, k] = modellp.addVar(
                vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"y_{branch_id}_{k}"
            )
            p[branch_id, k] = modellp.addVar(
                vtype=GRB.CONTINUOUS, lb=-float('inf'), ub=float('inf'),
                name=f"P_{branch_id}_{k}"
            )

    for branch_id, branch_info in branch_data.items():
        if branch_info['branch_status']:
            p0[branch_id] = modellp.addVar(
                vtype=GRB.CONTINUOUS,
                lb=-branch_info["Fmax"] * Sbase,
                ub= branch_info["Fmax"] * Sbase,
                name=f"P0_{branch_id}"
            )

    for bus in bus_data:
        theta[bus] = modellp.addVar(
            vtype=GRB.CONTINUOUS, lb=-float('inf'), ub=float('inf'),
            name=f'theta_{bus}'
        )

    # Objective  {IMPORTANT: Only generators with status=1 on the dataset are considered!}
    modellp.setObjective(
        gp.quicksum(
            g[gen_id] * wf * gen_info["CG/MWh"]
            for gen_id, gen_info in gend_data.items() if gen_info["genD_status"]
        ) +
        gp.quicksum(
            y[cand_id, k] * cand_info["IC"]
            for cand_id, cand_info in candidate_data.items()
            for k in range(1, cand_info["max_lines"] + 1)
        ),
        sense=GRB.MINIMIZE
    )

    # Nodal balance constraints (named so we can retrieve dual variables)
    nodal_constr = {}
    for bus_id, bus_info in bus_data.items():
        Pd = bus_info["bus_Pd"]
        out0 = gp.LinExpr(); in0 = gp.LinExpr()
        out  = gp.LinExpr(); inn = gp.LinExpr()

        for branch_id, branch_info in branch_data.items():
            if branch_info['branch_status']:
                if   branch_info["branch_fbus"] == bus_id: out0.add(p0[branch_id])
                elif branch_info["branch_tbus"] == bus_id: in0.add(p0[branch_id])

        for branch_id, branch_info in candidate_data.items():
            for k in range(1, branch_info["max_lines"] + 1):
                if   branch_info["branch_fbus"] == bus_id: out.add(p[branch_id, k])
                elif branch_info["branch_tbus"] == bus_id: inn.add(p[branch_id, k])

        balance = (-out - out0 + inn + in0) + gp.quicksum(
            g[gen_id]
            for gen_id, gen_info in gend_data.items()
            if gen_info["genD_bus"] == bus_id and gen_info["genD_status"]
        )
        nodal_constr[bus_id] = modellp.addConstr(balance == Pd, name=f"balance[{bus_id}]")

    # DC power-flow constraints - existing lines
    for branch_id, branch_info in branch_data.items():
        if branch_info['branch_status']:
            i, j = branch_info["branch_fbus"], branch_info["branch_tbus"]
            modellp.addConstr(p0[branch_id] * branch_info["branch_x"] == (theta[i] - theta[j]))

    # DC power-flow constraints - candidate lines (big-M)
    for branch_id, branch_info in candidate_data.items():
        i, j   = branch_info["branch_fbus"], branch_info["branch_tbus"]
        X_ij   = branch_info["branch_x"]
        M_ij   = dist[(i, j)]
        f_min  = -branch_info["Fmax"] * Sbase
        f_max  =  branch_info["Fmax"] * Sbase
        for k in range(1, branch_info["max_lines"] + 1):
            P_ijk = p[branch_id, k]
            modellp.addConstr(P_ijk >= f_min * y[branch_id, k])
            modellp.addConstr(P_ijk <= f_max * y[branch_id, k])
            modellp.addConstr(P_ijk * X_ij - theta[i] + theta[j] <=  M_ij * (1 - y[branch_id, k]))
            modellp.addConstr(P_ijk * X_ij - theta[i] + theta[j] >= -M_ij * (1 - y[branch_id, k]))

    # Add valid inequalities from the cut pool (Only the cut pool constrcuted using the separation algorithm)
    if cutpool:
        for cut_info in cutpool.values():
            if cut_info["GAMMA"] <= 0:
                continue
            m_node = cut_info['pair'][0]
            n_node = cut_info['pair'][1]
            underho = cut_info['und_rho']
            bigm    = cut_info['bigM']

            underhos = [[]]
            for idx in range(1, len(underho)):
                lids = line_map((underho[idx - 1], underho[idx]), branch_data)
                if len(lids) == 1:
                    for path in underhos:
                        path.append(lids[0])
                else:
                    new_underhos = [path + [line] for path in underhos for line in lids]
                    underhos = new_underhos

            for path1 in underhos:
                T1    = 0
                sumy1 = gp.LinExpr()
                cr1   = 0
                for edge in path1:
                    cr1 += branch_data[edge]['Fmax'] * branch_data[edge]["branch_x"] * Sbase
                    if branch_data[edge]['branch_status'] == 0:
                        eid = branch2cand(edge, branch_data, candidate_data)
                        sumy1.add(y[eid, 1])
                        T1 += 1
                Delta = bigm - cr1
                modellp.addConstr(theta[n_node] - theta[m_node] >= -(cr1 + (T1 - sumy1) * Delta))
                modellp.addConstr(theta[n_node] - theta[m_node] <=  (cr1 + (T1 - sumy1) * Delta))

    modellp.update()
    modellp.optimize()

    # Retrieve results
    LMP = {
        bus_id: modellp.getConstrByName(f"balance[{bus_id}]").Pi
        for bus_id in bus_data
    }

    p_res = {}
    for (cand_id, k), var in p.items():
        p_res.setdefault(cand_id, 0)
        if y[cand_id, 1].X > 0.0001:
            p_res[cand_id] += var.X

    p0_res = {bid: var.X for bid, var in p0.items() if var.X != 0}
    p0_res = {bid: 0 for bid in p0} | p0_res  # ensure all keys present

    sw     = {key: var.X for key, var in y.items()}
    thetas = {bus_id: theta[bus_id].X for bus_id in bus_data}

    return p_res, p0_res, LMP, sw, thetas


# ===========================================================================
#%% GRAPH FUNCTIONS (big-M computation helpers)
# ===========================================================================

def adj_list(branch_data, Sbase):
    """Build adjacency lists for all lines and existing lines only."""
    adj_all   = defaultdict(lambda: defaultdict(float))
    adj_exist = defaultdict(lambda: defaultdict(float))
    for info in branch_data.values():
        u, v = info['branch_fbus'], info['branch_tbus']
        w = info['branch_x'] * info['Fmax'] * Sbase
        adj_all[u][v] += w
        adj_all[v][u] += w
        if info['branch_status']:
            adj_exist[u][v] += w
            adj_exist[v][u] += w
    adj_all   = {u: list(vw.items()) for u, vw in adj_all.items()}
    adj_exist = {u: list(vw.items()) for u, vw in adj_exist.items()}
    return adj_all, adj_exist


def edge_list(branch_data, Sbase):
    """Return edge lists as [[u, v, w], ...] for existing and all lines."""
    all_edges      = defaultdict(float)
    existing_edges = defaultdict(float)
    for info in branch_data.values():
        u, v = info['branch_fbus'], info['branch_tbus']
        w = info['branch_x'] * info['Fmax'] * Sbase
        all_edges[(u, v)] += w
        if info['branch_status']:
            existing_edges[(u, v)] += w
    edge_list0 = [[u, v, w] for (u, v), w in existing_edges.items()]
    edge_list1 = [[u, v, w] for (u, v), w in all_edges.items()]
    return edge_list0, edge_list1


def adj_neighbors(adjacency_list, node):
    return [nbr for nbr, _ in adjacency_list[node]]


def is_connected(adj_ls, i, j):
    if i not in adj_ls or j not in adj_ls:
        return False
    visited = set()
    queue   = deque([i])
    while queue:
        node = queue.popleft()
        if node == j:
            return True
        if node not in visited:
            visited.add(node)
            queue.extend(nbr for nbr, _ in adj_ls[node] if nbr not in visited)
    return False


def has_edge(edgelist, i, j):
    return (i, j) in edgelist


def edge_weight(adjacency_list, i, j):
    if i in adjacency_list:
        for nbr, w in adjacency_list[i]:
            if nbr == j:
                return w
    return float('inf')


def dijkstra_adj(adjacency_list, source, num_vertices):
    """Dijkstra from *source*; returns (dist_dict, path_dict)."""
    vertices = list(range(1, num_vertices + 1))
    d    = {v: float('inf') for v in vertices};  d[source] = 0
    pred = {v: None         for v in vertices}
    heap = [(0, source)]
    vis  = set()
    while heap:
        dist_u, u = heapq.heappop(heap)
        if u in vis:
            continue
        vis.add(u)
        for nbr, w in adjacency_list.get(u, []):
            if nbr not in vis:
                nd = d[u] + w
                if nd < d[nbr]:
                    d[nbr]    = nd
                    pred[nbr] = u
                    heapq.heappush(heap, (nd, nbr))
    paths = {}
    for v in vertices:
        if d[v] < float('inf'):
            path, cur = [], v
            while cur is not None:
                path.append(cur); cur = pred[cur]
            path.reverse()
        else:
            path = []
        paths[v] = path
    return d, paths


# ===========================================================================
#%% BIG-M COMPUTATION
# ===========================================================================

def bounder_lspc(pair, adj_existing_lines, adj_all_lines,
                 edge_existing_lines, edge_all_lines, num_vertices, naive):
    """
    LSPC-based tight big-M bound for a bus pair (s, t).
    Falls back to *naive* if no valid bound is found.
    """
    s, t = pair
    start_node, end_node = s, t

    sn = set(adj_neighbors(adj_all_lines, start_node)); sn.discard(end_node)
    en = set(adj_neighbors(adj_all_lines, end_node));   en.discard(start_node)
    ng = [(i, j) for i in sn for j in en]

    nodes_bar = set()
    for u, v, _ in edge_all_lines:
        if u not in (start_node, end_node): nodes_bar.add(u)
        if v not in (start_node, end_node): nodes_bar.add(v)
    adjacency_bar = {node: [] for node in nodes_bar}
    for u, v, w in edge_all_lines:
        if u not in (start_node, end_node) and v not in (start_node, end_node):
            adjacency_bar[u].append((v, w))
            adjacency_bar[v].append((u, w))

    ng = [(i, j) for (i, j) in ng if is_connected(adjacency_bar, i, j)]

    remv, repl = set(), []
    for (i, j) in ng:
        if has_edge(edge_existing_lines, start_node, i):
            cand = (start_node, j)
            if cand not in ng and cand not in repl: repl.append(cand)
            remv.add((i, j))
    ng = [p for p in ng if p not in remv] + repl

    remv, repl = set(), []
    for (i, j) in ng:
        if has_edge(edge_existing_lines, end_node, j):
            cand = (i, end_node)
            if cand not in ng and cand not in repl: repl.append(cand)
            remv.add((i, j))
    ng = [p for p in ng if p not in remv] + repl

    tmp, seen_set = [], set()
    for (i, j) in ng:
        if i != start_node and is_connected(adj_existing_lines, start_node, j):
            cand = (start_node, j)
        elif j != end_node and is_connected(adj_existing_lines, end_node, i):
            cand = (i, end_node)
        else:
            cand = (i, j)
        if cand not in seen_set:
            tmp.append(cand); seen_set.add(cand)
    ng = tmp

    has_st = any(
        (u == start_node and v == end_node) or (u == end_node and v == start_node)
        for (u, v, _) in edge_all_lines
    )

    cn = sum(1 for (i, j) in ng if is_connected(adj_existing_lines, i, j)) if ng else 0
    if cn == len(ng):
        if has_st and (start_node, end_node) not in ng and (end_node, start_node) not in ng:
            ng.append((start_node, end_node))

        dj_cache = {}
        def dist_from(src):
            if src not in dj_cache:
                dj_cache[src] = dijkstra_adj(adj_existing_lines, src, num_vertices)[0]
            return dj_cache[src]

        lsp = 0.0
        for (i, j) in ng:
            if   i == start_node and j == end_node:
                lsp = max(lsp, edge_weight(adj_all_lines, start_node, end_node))
            elif i == start_node:
                lsp = max(lsp, dist_from(start_node)[j] + edge_weight(adj_all_lines, j, end_node))
            elif j == end_node:
                lsp = max(lsp, edge_weight(adj_all_lines, start_node, i) + dist_from(end_node)[i])
            else:
                lsp = max(lsp, edge_weight(adj_all_lines, start_node, i)
                               + dist_from(i)[j]
                               + edge_weight(adj_all_lines, j, end_node))
        return lsp

    return naive


def Only_needed_bounds(candidate_data, adj_existing_lines, adj_all_lines,
                       edge_existing_lines, edge_all_lines, num_vertices, naive):
    """Compute LSPC big-M bounds only for candidate line endpoints."""
    dist_list = {}
    for info in candidate_data.values():
        s, t  = info['branch_fbus'], info['branch_tbus']
        ds, _ = dijkstra_adj(adj_existing_lines, s, num_vertices)
        dd    = ds[t]
        if dd == float('inf'):
            dd = bounder_lspc(
                (s, t), adj_existing_lines, adj_all_lines,
                edge_existing_lines, edge_all_lines, num_vertices, naive
            )
        val = min(dd, naive)
        dist_list[(s, t)] = val
        dist_list[(t, s)] = val
    return dist_list


def Only_Sp(candidate_data, adj_existing_lines, num_vertices, naive):
    """Compute shortest-path big-M bounds only for candidate line endpoints."""
    dist_list = {}
    for info in candidate_data.values():
        s, t  = info['branch_fbus'], info['branch_tbus']
        ds, _ = dijkstra_adj(adj_existing_lines, s, num_vertices)
        val   = min(ds[t], naive)
        dist_list[(s, t)] = val
        dist_list[(t, s)] = val
    return dist_list


# ===========================================================================
#%% CYCLE CANDIDATE GENERATION
# ===========================================================================

def normalize_cycle(cycle_edges):
    """Return a canonical representation of an undirected cycle."""
    if not cycle_edges:
        return cycle_edges
    nodes = [cycle_edges[0][0]] + [e[1] for e in cycle_edges]
    if nodes[0] == nodes[-1]:
        nodes = nodes[:-1]
    n         = len(nodes)
    rotations = [nodes[i:] + nodes[:i] for i in range(n)]
    rev       = list(reversed(nodes))
    rotations.extend(rev[i:] + rev[:i] for i in range(n))
    best      = min(rotations)
    return [tuple(sorted((best[i], best[(i + 1) % n]))) for i in range(n)]


def build_edge_weights(branch_data, candidate_data, p_res, p0_res, lmp_by_bus,
                       alpha, gamma, delta, gap_normalize, crc_normalize, eps=1e-6):
    """
    Compute per-edge weights for cycle-finding Dijkstra.

        Weight_e = 1 / (eps + alpha*util_e + gamma*gap_e + delta*crc_e)

    Smaller weight  =>  edge is more attractive for cycle generation and separation.
    """
    lengths = {}

    # Cache undirected pair -> representative candidate_id
    cache = getattr(build_edge_weights, "_cand_pair_cache", None)
    ckey  = id(candidate_data)
    if cache is None or cache["key"] != ckey or cache["size"] != len(candidate_data):
        cand_pair_to_id = {}
        for cid, cinfo in candidate_data.items():
            cu, cv = cinfo['branch_fbus'], cinfo['branch_tbus']
            e = (cu, cv) if cu < cv else (cv, cu)
            if e not in cand_pair_to_id:
                cand_pair_to_id[e] = cid
        build_edge_weights._cand_pair_cache = {
            "key": ckey, "size": len(candidate_data), "map": cand_pair_to_id
        }
    else:
        cand_pair_to_id = cache["map"]

    if (alpha + gamma + delta) == 0:
        for info in branch_data.values():
            u, v = info['branch_fbus'], info['branch_tbus']
            lengths[(u, v) if u < v else (v, u)] = 1.0
        return lengths

    # LMP gap normalization prep
    use_gap  = (gamma != 0) and bool(lmp_by_bus)
    lam_mad  = None
    gmode    = (gap_normalize or "").lower()
    if use_gap:
        if gmode == "mad":
            vals = list(lmp_by_bus.values())
            if vals:
                m = np.median(vals)
                lam_mad = np.median(np.abs(np.array(vals, float) - m)) or 1.0
            else:
                lam_mad = 1.0
        

    # CRC normalization prep
    use_crc  = (delta != 0)
    rent_map = None
    rent_scale = None
    cmode    = (crc_normalize or "").lower()
    if use_crc:
        rent_map = {}
        for bid, info in branch_data.items():
            u, v = info['branch_fbus'], info['branch_tbus']
            e    = (u, v) if u < v else (v, u)
            gap  = abs(lmp_by_bus.get(u, 0.0) - lmp_by_bus.get(v, 0.0))
            flow = (float(p0_res.get(bid, 0.0)) if info['branch_status']
                    else float(p_res.get(cand_pair_to_id.get(e), 0.0)))
            rent_map[e] = rent_map.get(e, 0.0) + abs(gap * flow)
        rents = np.fromiter(rent_map.values(), float) if rent_map else np.array([])
        if cmode == "mad" and rents.size: rent_scale = np.median(np.abs(rents - np.median(rents))) or 1.0

    # Main loop
    for branchid, info in branch_data.items():
        u, v = info['branch_fbus'], info['branch_tbus']
        e    = (u, v) if u < v else (v, u)

        F    = info.get('Fmax', 0.0) or 0.0
        F    = F if F > 0.0 else 1e-6
        if info['branch_status']:
            util = abs(float(p0_res.get(branchid, 0.0)) / F)
        else:
            cid  = cand_pair_to_id.get(e)
            util = abs(float(p_res.get(cid, 0.0)) if cid is not None else 0.0) / F

        if use_gap and u in lmp_by_bus and v in lmp_by_bus:
            raw_gap = abs(lmp_by_bus[u] - lmp_by_bus[v])
            gap = raw_gap / (lam_mad + 1.0)
            
        else:
            gap = 0.0

        if use_crc:
            r = rent_map.get(e, 0.0)
            if   cmode =="mad": crc = r / (rent_scale + 1e-12) if r > 0 else 0.0
        else:
            crc = 0.0

        lengths[e] = 1.0 / (eps + alpha * util + gamma * gap + delta * crc)

    return lengths


def dijkstra_path(graph, lengths, s, t, banned_pair=None):
    """Weighted shortest path; optionally ban one undirected edge."""
    if s == t:
        return [s]
    dist = {s: 0.0};  pred = {s: None};  pq = [(0.0, s)]
    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]: continue
        if u == t:       break
        for v in graph.get(u, ()):
            e  = (u, v) if u < v else (v, u)
            if banned_pair is not None and e == banned_pair: continue
            nd = d + lengths.get(e, 1.0)
            if v not in dist or nd < dist[v]:
                dist[v] = nd; pred[v] = u
                heapq.heappush(pq, (nd, v))
    if t not in dist:
        return None
    path, cur = [], t
    while cur is not None:
        path.append(cur); cur = pred[cur]
    path.reverse()
    return path


def shortest_cycle_basis(branch_data, candidate_data, p_res, p0_res, LMP,
                         weight_configs=((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 1.0)),
                         gap_normalize="mad", crc_normalize="mad"):
    """
    Generate candidate cycles using weighted Dijkstra.

    Each entry in *weight_configs* is (alpha, gamma, delta) controlling
    utilization, LMP-gap, and congestion-cost edge weights respectively.
    """
    # Candidate-only graph (status == False)
    graph_cand = {}
    for info in branch_data.values():
        if not info['branch_status']:
            u, v = info['branch_fbus'], info['branch_tbus']
            if u != v:
                graph_cand.setdefault(u, set()).add(v)
                graph_cand.setdefault(v, set()).add(u)
    cands = [(u, v) for u in graph_cand for v in graph_cand[u] if u < v]

    # Full graph (existing + candidates)
    graph = {}
    for info in branch_data.values():
        u, v = info['branch_fbus'], info['branch_tbus']
        if u != v:
            graph.setdefault(u, set()).add(v)
            graph.setdefault(v, set()).add(u)

    lengths_cache = [
        build_edge_weights(
            branch_data, candidate_data, p_res, p0_res, LMP,
            alpha=a, gamma=g, delta=d,
            gap_normalize=gap_normalize, crc_normalize=crc_normalize
        )
        for (a, g, d) in weight_configs
    ]

    seen, cycles = set(), []

    def _run_pass(lengths):
        for u, v in cands:
            path = dijkstra_path(graph, lengths, u, v, banned_pair=(u, v))
            if path is None: continue
            ce   = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
            norm = normalize_cycle(ce)
            key  = tuple(norm)
            if key not in seen:
                seen.add(key); cycles.append(norm)

    for L in lengths_cache:
        _run_pass(L)

    return cycles


# ---------------------------------------------------------------------------
#%% Random cycle generation (used by RandomVI)
# ---------------------------------------------------------------------------

def bfs_shortest_path_random(graph, s, t, banned_pair=None, rng=None,
                              randomize_neighbors=True):
    """BFS shortest path with optional randomised neighbour order.
       The randomised neighbour is only necessary when multiple shortest paths 
       exist for the endpoints of a candidate line, where in this case
       BFS will pick one at random.
    """
    if s == t:
        return [s]
    if rng is None:
        rng = random.Random()
    q    = collections.deque([s])
    pred = {s: None}
    while q:
        u = q.popleft()
        if u == t: break
        nbrs = list(graph.get(u, ()))
        if randomize_neighbors:
            rng.shuffle(nbrs)
        for v in nbrs:
            e = (u, v) if u < v else (v, u)
            if banned_pair is not None and e == banned_pair: continue
            if v in pred: continue
            pred[v] = u
            q.append(v)
    if t not in pred:
        return None
    path, cur = [], t
    while cur is not None:
        path.append(cur); cur = pred[cur]
    path.reverse()
    return path


def shortest_cycle_basis_random_bfs(branch_data, seed=0,
                                    randomize_candidates=True,
                                    randomize_neighbors=True,
                                    max_cycles=None):
    """
    Generate candidate cycles using randomised BFS (no LP information needed).
    Used by the RandomVI method. The randomness in this function is only about 
    randomizing the order in which candidate lines are considered for cycle generaion.
    If you have any early termination, then order matters because the pool becomes
    "the first K unique cycles we happened to see." Otherwise, it's unnecessary!
    """
    graph_cand = {}
    for info in branch_data.values():
        if not info['branch_status']:
            u, v = info['branch_fbus'], info['branch_tbus']
            if u != v:
                graph_cand.setdefault(u, set()).add(v)
                graph_cand.setdefault(v, set()).add(u)
    cands = [(u, v) for u in graph_cand for v in graph_cand[u] if u < v]

    graph = {}
    for info in branch_data.values():
        u, v = info['branch_fbus'], info['branch_tbus']
        if u != v:
            graph.setdefault(u, set()).add(v)
            graph.setdefault(v, set()).add(u)

    rng = random.Random(seed)
    if randomize_candidates:
        rng.shuffle(cands)

    seen, cycles = set(), []
    for (u, v) in cands:
        path = bfs_shortest_path_random(graph, u, v, banned_pair=(u, v),
                                        rng=rng,
                                        randomize_neighbors=randomize_neighbors)
        if path is None or len(path) < 2:
            continue
        ce   = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        ce.append((path[-1], path[0]))
        norm = normalize_cycle(ce)
        key  = tuple(norm)
        if key not in seen:
            seen.add(key); cycles.append(norm)
            if max_cycles is not None and len(cycles) >= max_cycles:
                break
    return cycles


# ===========================================================================
#%% LINE / CANDIDATE INDEX HELPERS
# ===========================================================================

def _ensure_line_index(branch_data):
    cache = getattr(line_map, "_cache", None)
    key   = id(branch_data)
    if cache is None or cache["key"] != key or cache["size"] != len(branch_data):
        idx = defaultdict(list)
        for bid, info in branch_data.items():
            u, v = info['branch_fbus'], info['branch_tbus']
            idx[(u, v) if u < v else (v, u)].append(bid)
        line_map._cache = {"key": key, "size": len(branch_data), "idx": idx}
    return line_map._cache["idx"]


def _ensure_cand_index(candidate_data):
    cache = getattr(cand_map, "_cache", None)
    key   = id(candidate_data)
    if cache is None or cache["key"] != key or cache["size"] != len(candidate_data):
        idx = defaultdict(list)
        for cid, info in candidate_data.items():
            u, v = info['branch_fbus'], info['branch_tbus']
            idx[(u, v) if u < v else (v, u)].append(cid)
        cand_map._cache = {"key": key, "size": len(candidate_data), "idx": idx}
    return cand_map._cache["idx"]


def line_map(branch, branch_data):
    u, v = branch
    return _ensure_line_index(branch_data).get((u, v) if u < v else (v, u), [])


def cand_map(branch, candidate_data):
    u, v = branch
    return _ensure_cand_index(candidate_data).get((u, v) if u < v else (v, u), [])


# ===========================================================================
#%% SEPARATION (Cut Pool Construction for the method "VI")
# ===========================================================================

def get_cycle_nodes(edges):
    """Reconstruct ordered node list [v0, ..., vk, v0] from cycle edge set."""
    nbr = {}
    for a, b in edges:
        nbr.setdefault(a, []).append(b)
        nbr.setdefault(b, []).append(a)
    start = edges[0][0]
    prev  = start
    curr  = nbr[start][0]
    cycle = [start]
    while curr != start:
        cycle.append(curr)
        n0, n1 = nbr[curr][0], nbr[curr][1]
        prev, curr = curr, (n0 if n0 != prev else n1)
    cycle.append(start)
    return cycle


def separator(M_mat, find_all, sw, thetas, cycleset, it, cutset, violval,
              branch_data, candidate_data, dist, Sbase,
              adj_existing_lines, adj_all_lines, edge_existing_lines, edge_all_lines,
              num_vertices, naive):
    """
    Scan *cycleset* for violated PVIs

    Returns
    -------
    ncuts     : dict of newly discovered cuts
    temptance : updated big-M cache matrix
    """
    # O(1) edge-to-id indices
    cand_idx = defaultdict(list)
    for cid, info in candidate_data.items():
        u, v = info['branch_fbus'], info['branch_tbus']
        cand_idx[(u, v) if u < v else (v, u)].append(cid)

    line_idx = defaultdict(list)
    for bid, info in branch_data.items():
        u, v = info['branch_fbus'], info['branch_tbus']
        line_idx[(u, v) if u < v else (v, u)].append(bid)

    def cand_map_fast(a, b):
        return cand_idx.get((a, b) if a < b else (b, a), [])

    def line_map_fast(a, b):
        return line_idx.get((a, b) if a < b else (b, a), [])

    numerator = 0
    ncuts     = {}
    pairset   = {}
    temptance = copy.deepcopy(M_mat)

    for loops in cycleset:
        viol = violval          # MaxL: reset per cycle
        numerator += 1

        cycs   = get_cycle_nodes(loops)
        cycles = [cycs]
        for i in range(1, len(cycs) - 1):
            clock = [cycs[i]] + cycs[i + 1:] + cycs[1:i] + [cycs[i]]
            cycles.append(clock)
        cycles.extend(c[::-1] for c in list(cycles))

        for mdir in cycles:
            ccr   = [0];  xhats = [0];  cands = []

            for i in range(1, len(mdir)):
                a, b  = mdir[i - 1], mdir[i]
                lids  = cand_map_fast(a, b)
                if lids:
                    cr   = max(candidate_data[j]['branch_x'] * candidate_data[j]['Fmax'] * Sbase for j in lids)
                    xhat = max(sw[j, 1] for j in lids)
                    cands.append(True)
                else:
                    lids = line_map_fast(a, b)
                    cr   = min(branch_data[j]['branch_x'] * branch_data[j]['Fmax'] * Sbase for j in lids)
                    xhat = 1.0
                    cands.append(False)
                ccr.append(ccr[i - 1] + cr)
                xhats.append(xhats[i - 1] + xhat)

            if it == 0 and not find_all:
                if mdir[0] in adj_existing_lines:
                    distant, _ = dijkstra_adj(adj_existing_lines, mdir[0], num_vertices)
                    for col, val in distant.items():
                        temptance[mdir[0], col] = min(naive, val, temptance[mdir[0], col])
                        temptance[col, mdir[0]] = temptance[mdir[0], col]

                for i in range(1, len(mdir) - 1):
                    deltheta = abs(thetas[mdir[i]] - thetas[mdir[0]])
                    if ccr[i] < deltheta - 0.01 and any(cands[:i]) and ccr[i] != ccr[-1]:
                        pair = (mdir[0], mdir[i])
                        if temptance[mdir[0], mdir[i]] == naive:
                            if pair in dist:
                                m = dist[pair]
                            else:
                                m = bounder_lspc(pair, adj_existing_lines, adj_all_lines,
                                                 edge_existing_lines, edge_all_lines, num_vertices, naive)
                            temptance[mdir[0], mdir[i]] = m
                            temptance[mdir[i], mdir[0]] = m
                        else:
                            m = temptance[mdir[0], mdir[i]]

                        if ccr[i] < m - 0.1:
                            und_gamma  = i - xhats[i]
                            PVIS_rhs   = ccr[i] + (m - ccr[i]) * und_gamma
                            vl         = deltheta - PVIS_rhs
                            sorted_pair = tuple(sorted([mdir[0], mdir[i]]))
                            if vl > viol:
                                pairset[sorted_pair] = vl
                                rhoshort = mdir[:i + 1]
                                cut_key  = (sorted_pair, tuple(sorted(rhoshort)), 0)
                                if cut_key not in cutset:
                                    cutset.add(cut_key)
                                    viol = vl
                                    ncuts[f"{it:02}{numerator}"] = {
                                        "GAMMA":   vl,
                                        "pair":    (mdir[0], mdir[i]),
                                        "und_rho": rhoshort,
                                        "bigM":    m
                                    }
            else:
                for i in range(1, len(mdir) - 1):
                    deltheta = abs(thetas[mdir[i]] - thetas[mdir[0]])
                    if ccr[i] < deltheta - 0.01 and any(cands[:i]) and ccr[i] != ccr[-1]:
                        if temptance[mdir[0], mdir[i]] != naive:
                            m = temptance[mdir[0], mdir[i]]
                        else:
                            if mdir[0] in adj_existing_lines:
                                distant, _ = dijkstra_adj(adj_existing_lines, mdir[0], num_vertices)
                                for col, val in distant.items():
                                    temptance[mdir[0], col] = min(naive, val, temptance[mdir[0], col])
                                    temptance[col, mdir[0]] = temptance[mdir[0], col]
                            pair = (mdir[0], mdir[i])
                            if temptance[mdir[0], mdir[i]] == naive:
                                if pair in dist:
                                    m = dist[pair]
                                else:
                                    m = bounder_lspc(pair, adj_existing_lines, adj_all_lines,
                                                     edge_existing_lines, edge_all_lines, num_vertices, naive)
                                temptance[mdir[0], mdir[i]] = m
                                temptance[mdir[i], mdir[0]] = m
                            else:
                                m = temptance[mdir[0], mdir[i]]

                        if ccr[i] < m - 1e-1:
                            und_gamma  = i - xhats[i]
                            PVIS_rhs   = ccr[i] + (m - ccr[i]) * und_gamma
                            vl         = deltheta - PVIS_rhs
                            sorted_pair = tuple(sorted([mdir[0], mdir[i]]))
                            if vl > viol:
                                pairset[sorted_pair] = vl
                                rhoshort = mdir[:i + 1]
                                cut_key  = (sorted_pair, tuple(sorted(rhoshort)), 0)
                                if cut_key not in cutset:
                                    cutset.add(cut_key)
                                    viol = vl
                                    ncuts[f"{it:02}{numerator}"] = {
                                        "GAMMA":   vl,
                                        "pair":    (mdir[0], mdir[i]),
                                        "und_rho": rhoshort,
                                        "bigM":    m
                                    }

    return ncuts, temptance


def random_cut_gen(cycleset, branch_data, candidate_data, dist, Sbase, it,
               num_vertices, naive, adj_existing_lines, adj_all_lines,
               edge_existing_lines, edge_all_lines, seed=0):
    """
    Generate cuts from randomly selected paths from each cycle in *cycleset*.
    Used exclusively by the RandomVI method.
    """
    cand_idx = defaultdict(list)
    for cid, info in candidate_data.items():
        u, v = info['branch_fbus'], info['branch_tbus']
        cand_idx[(u, v) if u < v else (v, u)].append(cid)

    line_idx = defaultdict(list)
    for bid, info in branch_data.items():
        u, v = info['branch_fbus'], info['branch_tbus']
        line_idx[(u, v) if u < v else (v, u)].append(bid)

    def cand_map_fast(a, b):
        return cand_idx.get((a, b) if a < b else (b, a), [])

    def line_map_fast(a, b):
        return line_idx.get((a, b) if a < b else (b, a), [])

    cuts      = {}
    numerator = 0

    for loops in cycleset:
        numerator += 1
        cycs   = get_cycle_nodes(loops)
        cycles = [cycs]
        for i in range(1, len(cycs) - 1):
            clock = [cycs[i]] + cycs[i + 1:] + cycs[1:i] + [cycs[i]]
            cycles.append(clock)
        cycles.extend(c[::-1] for c in list(cycles))

        flag = False
        while not flag:
            rnd = random.Random(seed)
            rnd.shuffle(cycles)
            rand_cycle = cycles[0]
            Le     = len(rand_cycle)
            slicer = random.randint(1, Le - 1)
            path   = []
            for i in range(1, slicer):
                path.append((rand_cycle[i - 1], rand_cycle[i]))
                path.append((rand_cycle[i], rand_cycle[i - 1]))
            flag = bool(set(path) & cand_idx.keys())

        ccr = [0]
        for i in range(1, slicer):
            a, b = rand_cycle[i - 1], rand_cycle[i]
            lids = cand_map_fast(a, b)
            if lids:
                cr = max(candidate_data[j]['branch_x'] * candidate_data[j]['Fmax'] * Sbase for j in lids)
            else:
                lids = line_map_fast(a, b)
                cr   = min(branch_data[j]['branch_x'] * branch_data[j]['Fmax'] * Sbase for j in lids)
            ccr.append(ccr[i - 1] + cr)

        pair = (rand_cycle[0], rand_cycle[slicer - 1])
        m    = float('inf')
        if rand_cycle[0] in adj_existing_lines:
            distant, _ = dijkstra_adj(adj_existing_lines, rand_cycle[0], num_vertices)
            m = distant[rand_cycle[slicer - 1]]
        if m == float('inf'):
            m = bounder_lspc(pair, adj_existing_lines, adj_all_lines,
                             edge_existing_lines, edge_all_lines, num_vertices, naive)

        if ccr[-1] < m:
            cuts[f"{it:02}{numerator}"] = {
                "cr":   ccr[-1],
                "pair": pair,
                "path": rand_cycle[:slicer],
                "bigM": m
            }

    return cuts


# ===========================================================================
#%% MILP SOLVER
# ===========================================================================

def MILP_solver(method, bus_data, branch_data, candidate_data, gend_data,
               cutpool, dist, Sbase, timelimit, opt_gap, wf):
    """
    Build and solve the DC-TEP MILP with Gurobi.

    *cutpool* may contain VI cuts (from VI / RandomVI methods) or be
    empty (SP / NVI / LSPC).

    Returns
    -------
    optval : optimal objective value, or 0 (time limit), or 'infsble'
    gap    : MILP gap at termination 
    """
    model = gp.Model("DCOTS")
    model.Params.TimeLimit = timelimit
    model.setParam('MIPGap', opt_gap)

    g, p0, p, theta, y = {}, {}, {}, {}, {}

    for gen_id, gen_info in gend_data.items():
        if gen_info["genD_status"]:
            g[gen_id] = model.addVar(
                vtype=GRB.CONTINUOUS, lb=0, ub=gen_info["genD_Pmax"],
                name=f"g_{gen_id}"
            )

    for branch_id, branch_info in candidate_data.items():
        for k in range(1, branch_info["max_lines"] + 1):
            y[branch_id, k] = model.addVar(vtype=GRB.BINARY, name=f"y_{branch_id}_{k}")
            p[branch_id, k] = model.addVar(
                vtype=GRB.CONTINUOUS, lb=-float('inf'), ub=float('inf'),
                name=f"P_{branch_id}_{k}"
            )

    for branch_id, branch_info in branch_data.items():
        if branch_info['branch_status']:
            p0[branch_id] = model.addVar(
                vtype=GRB.CONTINUOUS,
                lb=-branch_info["Fmax"] * Sbase,
                ub= branch_info["Fmax"] * Sbase,
                name=f"P0_{branch_id}"
            )

    for bus in bus_data:
        theta[bus] = model.addVar(
            vtype=GRB.CONTINUOUS, lb=-float('inf'), ub=float('inf'),
            name=f'theta_{bus}'
        )

    # Objective
    model.setObjective(
        gp.quicksum(
            g[gen_id] * wf * gen_info["CG/MWh"]
            for gen_id, gen_info in gend_data.items() if gen_info["genD_status"]
        ) +
        gp.quicksum(
            y[cand_id, k] * cand_info["IC"]
            for cand_id, cand_info in candidate_data.items()
            for k in range(1, cand_info["max_lines"] + 1)
        ),
        sense=GRB.MINIMIZE
    )

    # Nodal balance
    for bus_id, bus_info in bus_data.items():
        Pd = bus_info["bus_Pd"]
        out0 = gp.LinExpr(); in0 = gp.LinExpr()
        out  = gp.LinExpr(); inn = gp.LinExpr()
        for branch_id, branch_info in branch_data.items():
            if branch_info['branch_status']:
                if   branch_info["branch_fbus"] == bus_id: out0.add(p0[branch_id])
                elif branch_info["branch_tbus"] == bus_id: in0.add(p0[branch_id])
        for branch_id, branch_info in candidate_data.items():
            for k in range(1, branch_info["max_lines"] + 1):
                if   branch_info["branch_fbus"] == bus_id: out.add(p[branch_id, k])
                elif branch_info["branch_tbus"] == bus_id: inn.add(p[branch_id, k])
        model.addConstr(
            (-out - out0 + inn + in0) + gp.quicksum(
                g[gen_id] for gen_id, gen_info in gend_data.items()
                if gen_info["genD_bus"] == bus_id and gen_info["genD_status"]
            ) == Pd
        )

    # DC flow - existing
    for branch_id, branch_info in branch_data.items():
        if branch_info['branch_status']:
            i, j = branch_info["branch_fbus"], branch_info["branch_tbus"]
            model.addConstr(p0[branch_id] * branch_info["branch_x"] == (theta[i] - theta[j]))

    # DC flow - candidates (big-M)
    for branch_id, branch_info in candidate_data.items():
        i, j  = branch_info["branch_fbus"], branch_info["branch_tbus"]
        X_ij  = branch_info["branch_x"]
        M_ij  = dist if method == "NVI" else dist[(i, j)]
        f_min = -branch_info["Fmax"] * Sbase
        f_max =  branch_info["Fmax"] * Sbase
        for k in range(1, branch_info["max_lines"] + 1):
            P_ijk = p[branch_id, k]
            model.addConstr(P_ijk >= f_min * y[branch_id, k])
            model.addConstr(P_ijk <= f_max * y[branch_id, k])
            model.addConstr(P_ijk * X_ij - theta[i] + theta[j] <=  M_ij * (1 - y[branch_id, k]))
            model.addConstr(P_ijk * X_ij - theta[i] + theta[j] >= -M_ij * (1 - y[branch_id, k]))

    # Only needed if multiple candidate lines are considered within expansion corridors
    for branch_id, branch_info in candidate_data.items():
        for k in range(1, branch_info["max_lines"]):
            model.addConstr(y[branch_id, k] >= y[branch_id, k + 1])

    # Valid inequalities from cut pool
    if cutpool:
        for cut_info in cutpool.values():
            # RandomVI cuts 
            if "path" in cut_info:
                # RandomVI cut
                m_node = cut_info['pair'][0]
                n_node = cut_info['pair'][1]
                route  = cut_info['path']
                bigm   = cut_info['bigM']
                underhos = [[]]
                for idx in range(1, len(route)):
                    lids = line_map((route[idx - 1], route[idx]), branch_data)
                    if len(lids) == 1:
                        for path in underhos: path.append(lids[0])
                    else:
                        underhos = [path + [line] for path in underhos for line in lids]
                for path1 in underhos:
                    T1 = 0; sumy1 = gp.LinExpr(); cr1 = 0
                    for edge in path1:
                        cr1 += branch_data[edge]['Fmax'] * branch_data[edge]["branch_x"] * Sbase
                        if branch_data[edge]['branch_status'] == 0:
                            eid = branch2cand(edge, branch_data, candidate_data)
                            sumy1.add(y[eid, 1]); T1 += 1
                    Delta = bigm - cr1
                    model.addConstr(theta[n_node] - theta[m_node] >= -(cr1 + (T1 - sumy1) * Delta))
                    model.addConstr(theta[n_node] - theta[m_node] <=  (cr1 + (T1 - sumy1) * Delta))
            else:
                # VI cuts
                if cut_info.get("GAMMA", 0) <= 0:
                    continue
                m_node  = cut_info['pair'][0]
                n_node  = cut_info['pair'][1]
                underho = cut_info['und_rho']
                bigm    = cut_info['bigM']
                underhos = [[]]
                for idx in range(1, len(underho)):
                    lids = line_map((underho[idx - 1], underho[idx]), branch_data)
                    if len(lids) == 1:
                        for path in underhos: path.append(lids[0])
                    else:
                        underhos = [path + [line] for path in underhos for line in lids]
                for path1 in underhos:
                    T1 = 0; sumy1 = gp.LinExpr(); cr1 = 0
                    for edge in path1:
                        cr1 += branch_data[edge]['Fmax'] * branch_data[edge]["branch_x"] * Sbase
                        if branch_data[edge]['branch_status'] == 0:
                            eid = branch2cand(edge, branch_data, candidate_data)
                            sumy1.add(y[eid, 1]); T1 += 1
                    Delta = bigm - cr1
                    model.addConstr(theta[n_node] - theta[m_node] >= -(cr1 + (T1 - sumy1) * Delta))
                    model.addConstr(theta[n_node] - theta[m_node] <=  (cr1 + (T1 - sumy1) * Delta))

    model.update()
    model.optimize()

    if model.Status == GRB.OPTIMAL:
        return model.objVal, 0
    elif model.Status == GRB.TIME_LIMIT:
        return 0, (model.MIPGap if model.SolCount > 0 else "No_inc")
    else:
        return 'infsble', 'infsble'


# ===========================================================================
#%% CUTTING-PLANE LOOP
# ===========================================================================

def cutting_plane(method, find_all, bus_data, branch_data, candidate_data, gend_data,
                  violation_threshold, iteration, dist, Sbase, ins, wf, BigM_time,
                  adj_existing_lines, adj_all_lines, edge_existing_lines, edge_all_lines,
                  num_vertices, naive):
    """
    Run the cutting-plane pre-processing phase and then solve the MILP.

    Methods
    -----------------
    VI       : LP-guided Cycle generation using PFCH + Separation
    RandomVI : BFS cycle generation + random_cut_gen
    SP / NVI / LSPC : No cutting-plane iterations; proceed directly to MILP
    """
    key     = f"{ins}-{violation_threshold}-{iteration}"
    cutpool = {}
    prev    = 0
    cutset  = set()
    res     = {}

    start_time = time.time()
    
    if method == "VI":
        for it in range(iteration):
            p_res, p0_res, LMP, sw, thetas = LR(
                cutpool, bus_data, branch_data, gend_data, candidate_data, dist, Sbase, wf
            )
            candidate_cycles = shortest_cycle_basis(
                branch_data, candidate_data, p_res, p0_res, LMP
            )
            print(f"  [VI] {len(candidate_cycles)} cycles")
            if it == 0:
                Ms = np.full((num_vertices + 1, num_vertices + 1), naive)
            pool, Ms = separator(
                Ms, find_all, sw, thetas, candidate_cycles, it, cutset,
                violation_threshold, branch_data, candidate_data, dist, Sbase,
                adj_existing_lines, adj_all_lines, edge_existing_lines,
                edge_all_lines, num_vertices, naive
            )
    
            cutpool.update(pool)
            print(f"  Total cuts so far: {len(cutpool)}")
            if len(cutpool) == prev:
                break
            prev = len(cutpool)
    
        prep_time = round(time.time() - start_time, 5)
        res[key]  = {
            "num_cuts":     len(cutpool),
            "num_cycles":   len(candidate_cycles),
            "iter":         it,
            "preprocessing": prep_time + BigM_time,
            "MIP_time":      0,
            "MIP_gap":       0,
            "OptSol":        0
        }
    
    elif method == "RandomVI":
        it=0
        candidate_cycles = shortest_cycle_basis_random_bfs(
            branch_data, seed=0,
            randomize_candidates=True,
            randomize_neighbors=True
        )
        print(f"  [RandomVI] {len(candidate_cycles)} cycles")
        pool = random_cut_gen(
            candidate_cycles, branch_data, candidate_data, dist,
            Sbase, it, num_vertices, naive,
            adj_existing_lines, adj_all_lines,
            edge_existing_lines, edge_all_lines, seed=0
        )
    
        cutpool.update(pool)
        print(f"  Total cuts so far: {len(cutpool)}")
        
        prep_time = round(time.time() - start_time, 5)
        res[key]  = {
            "num_cuts":     len(cutpool),
            "num_cycles":   len(candidate_cycles),
            "iter":         it,
            "preprocessing": prep_time + BigM_time,
            "MIP_time":      0,
            "MIP_gap":       0,
            "OptSol":        0
        }
    
  
    else:
        res[key] = {
            "preprocessing": BigM_time,
            "MIP_time":  0,
            "MIP_gap":   0,
            "OptSol":    0
        }

    timelimit = 7200
    opt_gap   = 0.001
    start_time = time.time()
    optval, MIP_gap = MILP_solver(
        method, bus_data, branch_data, candidate_data, gend_data,
        cutpool, dist, Sbase, timelimit, opt_gap, wf
    )
    res[key]["Optsol"]   = optval
    res[key]["MIP_gap"]  = MIP_gap
    res[key]["MIP_time"] = round(time.time() - start_time, 5)

    return res


# ===========================================================================
#%% RUN ORCHESTRATION
# ===========================================================================

def run(method, find_all, task, bus_data, branch_data, candidate_data, gend_data,
        loads, instance_set, Sbase, wf, naive):
    """
    Orchestrate big-M pre-computation and per-instance solving.

    Parameters
    ----------
    method       : "VI" | "RandomVI" | "SP" | "NVI" | "LSPC"
    find_all     : if True, run full LSPC for all pairs (not recommended)
    loads        : numpy array of shape (n_instances, n_buses), or None
    instance_set : iterable of load scenario indices, or None (use base loads)
    """
    results = {}
    violation_thresholds = [1e-6]
    iters                = [5]

    adj_all_lines, adj_existing_lines = adj_list(branch_data, 1)
    edge_existing_lines, edge_all_lines = edge_list(branch_data, Sbase)
    num_vertices = len(
        set(adj_all_lines) | {v for edges in adj_all_lines.values() for v, _ in edges}
    )

    # --- Big-M pre-computation ---
    print(f"Computing big-M bounds (method={method}) ...")
    start_time = time.time()
    if find_all:
        raise NotImplementedError("Full LSPC (find_all=True) not exposed in this interface.")
    elif method in ("VI", "RandomVI", "LSPC"):
        dist_mat = Only_needed_bounds(
            candidate_data, adj_existing_lines, adj_all_lines,
            edge_existing_lines, edge_all_lines, num_vertices, naive
        )
    elif method == "SP":
        dist_mat = Only_Sp(candidate_data, adj_existing_lines, num_vertices, naive)
    elif method == "NVI":
        dist_mat = naive
    else:
        raise ValueError(f"Unknown method: {method}")
    BigM_time = round(time.time() - start_time, 5)
    print(f"Big-M done in {BigM_time}s")

    def _solve_instance(instance, label):
        """Run cutting_plane for a single load scenario."""
        if instance is not None:
            for bus_id, bus_info in bus_data.items():
                bus_info['bus_Pd'] = loads[instance, bus_id - 1]

        if method in ("VI", "RandomVI"):
            for viol in violation_thresholds:
                for iteration in iters:
                    result = cutting_plane(
                        method, find_all, bus_data, branch_data, candidate_data,
                        gend_data, viol, iteration, dist_mat, Sbase, label, wf,
                        BigM_time, adj_existing_lines, adj_all_lines,
                        edge_existing_lines, edge_all_lines, num_vertices, naive
                    )
                    results.update(result)
        else:
            # SP / NVI / LSPC: no cutting-plane loop
            key     = f"{label}"
            cutpool = {}
            res     = {key: {"preprocessing": BigM_time, "MIP_time": 0, "MIP_gap": 0, "OptSol": 0}}
            timelimit = 7200; opt_gap = 0.001
            start_t = time.time()
            optval, MIP_gap = MILP_solver(
                method, bus_data, branch_data, candidate_data, gend_data,
                cutpool, dist_mat, Sbase, timelimit, opt_gap, wf
            )
            res[key]["Optsol"]   = optval
            res[key]["MIP_gap"]  = MIP_gap
            res[key]["MIP_time"] = round(time.time() - start_t, 5)
            results.update(res)

    # --- Instance loop ---
    if instance_set is not None:
        print(f"Running {len(list(instance_set))} instances ...")
        for instance in instance_set:
            print(f"  Instance {instance}")
            _solve_instance(instance, instance)
            df = pd.DataFrame.from_dict(results, orient='index')
            df.to_csv(
                Out / f"{task}_{method}_{list(instance_set)}.txt",
                sep='\t', float_format='%.4f', header=False
            )
    else:
        print("Running on base load scenario ...")
        _solve_instance(None, 'root')
        df = pd.DataFrame.from_dict(results, orient='index')
        df.to_csv(
            Out / f"{task}_{method}_root.txt",
            sep='\t', float_format='%.4f', header=False
        )

    return results


# ===========================================================================
#%% MAIN
# ===========================================================================

def main():
    """
    Entry point. Edit the settings below to run different experiments.

    Testcases
    ---------
    "6717" : 6717-bus system  (Sbase=1,   wf=0.05)
    
    Methods
    -------
    "VI"       : LSPC+Shortest-path big-M, PFCH+Separation cuts
    "RandomVI" : LSPC+Shortest-path big-M, Randomly generated cuts
    "SP"       : Shortest-path big-M, no cuts
    "NVI"      : Naive (sum-of-all) big-M, no cuts
    "LSPC"     : LSPC+Shortest-path big-M, no cuts
    """
    testcases = {
        "6717": {"dataset": "6717bus_v1.4.dat",  "load_update": "6717bus_loads_v1.1.txt", "gen_update": None}
    }

    # ---- User settings ----
    tasks        = ["6717"]
    method       = "VI"             # "VI" | "RandomVI" | "SP" | "NVI" | "LSPC"
    instance_set = range(35)          # set to None to use base loads
    find_all     = False
    # -----------------------

    for task in tasks:
        Sbase = 1
        wf    = 0.05   # Scaling factor to make investment costs and generation costs comparable

        bus_data, branch_data, gend_data, candidate_data = read_data(testcases[task])

        naive = round(
            sum(info['branch_x'] * info['Fmax'] * Sbase for info in branch_data.values()),
            1
        )

        loads = (                      # The load at each bus for each instance
            np.loadtxt(In / testcases[task]["load_update"])
            if testcases[task]["load_update"] is not None
            else None
        )

        run(method, find_all, task, bus_data, branch_data, candidate_data,
            gend_data, loads, instance_set, Sbase, wf, naive)


if __name__ == "__main__":
    main()