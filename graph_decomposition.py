import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter, defaultdict

# =========================================================
# Helpers: graph + conditions
# =========================================================

def complete_graph_minus_perfect_matching(n: int) -> nx.Graph:
    """
    Return K_n if n is odd; if n is even return K_n - I
    where I is the standard 1-factor {(i, i+n/2) : i < n/2} on labels 0..n-1.
    """
    G = nx.complete_graph(n)
    if n % 2 == 0:
        half = n // 2
        to_remove = [(i, (i + half) % n) for i in range(half)]
        G.remove_edges_from(to_remove)
    return G

def alspach_edge_target(n: int) -> int:
    # Number of edges in K_n (odd) or K_n - I (even)
    return n * (n - 1) // 2 if n % 2 else n * (n - 2) // 2

def check_equal_m_conditions(n: int, m: int) -> tuple[bool, str]:
    if not (3 <= m <= n):
        return False, "Duljina ciklusa m mora biti u intervalu [3, n]."
    total_edges = alspach_edge_target(n)
    if total_edges % m != 0:
        return False, f"Broj bridova ({total_edges}) nije djeljiv s m={m}."
    return True, "Uvjeti zadovoljeni."

# =========================================================
# Cycles: enumeration and utilities
# =========================================================

def canonical_cycle(cycle):
    """
    Normalize undirected simple cycle (list of nodes, last not repeated):
    - rotate so the smallest node is first
    - choose direction (original or reversed) lexicographically smaller
    """
    c = list(cycle)
    if len(c) > 1 and c[0] == c[-1]:
        c = c[:-1]
    m = min(c)
    idx = c.index(m)
    c1 = c[idx:] + c[:idx]
    c2 = [c1[0]] + list(reversed(c1[1:]))
    return tuple(c1) if tuple(c1) <= tuple(c2) else tuple(c2)

def enumerate_simple_cycles_of_length(G: nx.Graph, m: int) -> list[tuple]:
    """
    Enumerate *undirected* simple cycles of exact length m via backtracking.
    Pruning: enforce start < second to break symmetries.
    """
    nodes = list(G.nodes())
    nbrs = {u: set(G.neighbors(u)) for u in nodes}
    cycles_set = set()

    def dfs(start, current, visited):
        if len(current) == m:
            if start in nbrs[current[-1]]:
                cycles_set.add(canonical_cycle(current))
            return
        last = current[-1]
        for w in nbrs[last]:
            if w == start:  # close only when length==m
                continue
            if w in visited:
                continue
            if len(current) == 1 and not (start < w):
                continue
            current.append(w)
            visited.add(w)
            dfs(start, current, visited)
            visited.remove(w)
            current.pop()

    for s in nodes:
        dfs(s, [s], {s})

    return sorted(cycles_set)

def edges_of_cycle(cyc: tuple) -> tuple[tuple]:
    return tuple(tuple(sorted((cyc[i], cyc[(i+1) % len(cyc)]))) for i in range(len(cyc)))

def format_cycle(cyc) -> str:
    return "(" + " - ".join(str(v) for v in cyc) + ")"

# =========================================================
# Exact cover for equal m-cycles (returns catalogs for UI)
# =========================================================

def exact_cover_equal_m(G: nx.Graph, m: int, max_nodes: int = 200000):
    """
    Select exactly T cycles of length m so that all edges are covered exactly once,
    where T = |E(G)| / m.
    Backtracking with 'most constrained edge' heuristic.

    Returns:
        ok: bool
        chosen: list[tuple]               # selected m-cycles
        all_cycles: list[tuple]           # all candidate m-cycles
        cyc_edges: dict[cycle -> edges]   # edges per candidate cycle
        edge2cycles: dict[edge -> list[c]]# candidates per edge
    """
    E = set(tuple(sorted(e)) for e in G.edges())
    target_edges = len(E)
    T = target_edges // m

    # Precompute candidate m-cycles and their edges
    all_cycles = enumerate_simple_cycles_of_length(G, m)
    cyc_edges = {cyc: edges_of_cycle(cyc) for cyc in all_cycles}

    # Edge -> candidate cycles that contain it
    edge2cycles = defaultdict(list)
    for cyc in all_cycles:
        for e in cyc_edges[cyc]:
            if e in E:
                edge2cycles[e].append(cyc)

    # Early check: every edge must be in at least one m-cycle
    for e in E:
        if not edge2cycles[e]:
            return False, [], all_cycles, cyc_edges, edge2cycles

    chosen = []
    covered = set()
    used_cycle = set()
    nodes_expanded = 0

    def pick_most_constrained_edge():
        best_e, best_cnt = None, None
        for e in E:
            if e in covered:
                continue
            cnt = 0
            for c in edge2cycles[e]:
                if c in used_cycle:
                    continue
                # cycle must be fully unused
                es = cyc_edges[c]
                if all((ed not in covered) for ed in es):
                    cnt += 1
            if cnt == 0:
                return e, 0
            if best_cnt is None or cnt < best_cnt:
                best_e, best_cnt = e, cnt
                if best_cnt == 1:
                    break
        return best_e, best_cnt or 0

    def backtrack():
        nonlocal nodes_expanded
        nodes_expanded += 1
        if nodes_expanded > max_nodes:
            return False

        if len(covered) == target_edges and len(chosen) == T:
            return True
        if len(chosen) > T:
            return False

        e, cnt = pick_most_constrained_edge()
        if cnt == 0:
            return False

        for cyc in edge2cycles[e]:
            if cyc in used_cycle:
                continue
            es = cyc_edges[cyc]
            if any((x in covered) for x in es):
                continue

            # choose
            used_cycle.add(cyc)
            chosen.append(cyc)
            for x in es:
                covered.add(x)

            if backtrack():
                return True

            # undo
            for x in es:
                covered.remove(x)
            chosen.pop()
            used_cycle.remove(cyc)

        return False

    ok = backtrack()
    return ok, chosen, all_cycles, cyc_edges, edge2cycles

# =========================================================
# Visualization
# =========================================================

def draw_graph(G, pos=None, title=None):
    if pos is None:
        pos = nx.spring_layout(G, seed=42)  # fixed layout (seed removed from UI)
    fig, ax = plt.subplots()
    nx.draw_networkx_nodes(G, pos, node_size=600, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
    nx.draw_networkx_edges(G, pos, width=1.5, ax=ax)
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    return fig, pos

def draw_cycles_colored(G, cycles, pos, title=None, idx_map=None):
    """
    Each cycle gets a distinct color; if idx_map given, color index follows cycle IDs.
    """
    fig, ax = plt.subplots()
    nx.draw_networkx_nodes(G, pos, node_size=600, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)

    # plenty of distinct colors
    cmap = plt.cm.get_cmap("tab20", max(20, len(cycles)))
    for cyc in cycles:
        idx = (idx_map[cyc] - 1) if idx_map else list(cycles).index(cyc)
        color = cmap(idx % cmap.N)
        es = list(edges_of_cycle(cyc))
        nx.draw_networkx_edges(G, pos, edgelist=es, width=3, edge_color=[color], ax=ax)

    ax.set_axis_off()
    if title:
        ax.set_title(title)
    return fig

# =========================================================
# Streamlit UI
# =========================================================

st.set_page_config(page_title="Dekompozicija grafa", layout="wide")
st.title("Dekompozicija potpunog grafa na cikluse jednake duljine")

with st.sidebar:
    st.header("Ulazni parametri")
    n = st.number_input("Broj vrhova n", min_value=3, max_value=24, value=9, step=1)
    m = st.number_input(
        "Duljina ciklusa m",
        min_value=3, max_value=24, value=3, step=1,
        help="Aplikacija dekomponira Kₙ (ako je n neparan) ili Kₙ − I (ako je n paran) u cikluse duljine m, ako je to moguće."
    )
    max_nodes = st.number_input(
        "Limit koraka pretraživanja",
        min_value=1000, max_value=2_000_000, value=200_000, step=1000,
        help="Veća vrijednost može pomoći kod težih instanci, ali usporava pretragu."
    )
    run = st.button("Pokreni dekompoziciju")

st.markdown(
    """
**Napomena:**  
- Ako je *n* **neparan**, radi se s **Kₙ**; ako je **paran**, radi se s **Kₙ − I**.  
- Ovdje se traži **isključivo dekompozicija na m-cikluse**.  
"""
)

left, right = st.columns(2)

# Build and show initial graph
G = complete_graph_minus_perfect_matching(n)
with left:
    fig0, pos = draw_graph(G, title=f"Inicijalni graf ({'Kₙ' if n%2 else 'Kₙ − I'}) za n={n}")
    st.pyplot(fig0, clear_figure=True)

if run:
    ok_cond, msg = check_equal_m_conditions(n, int(m))
    if not ok_cond:
        st.error(f"Uvjet nije zadovoljen: {msg}")
    else:
        st.success(f"Alspach uvjet (slučaj m-ciklusa): {msg}")

        with st.status("Tražim m-ciklusnu dekompoziciju…", expanded=True) as status:
            st.write("• Generiram sve kandidate za m-cikluse…")
            st.write("• Pokrećem pretraživanje egzaktnog pokrivanja s heuristikom najviše ograničenog brida…")
            ok, cycles, all_cycles, cyc_edges, edge2cycles = exact_cover_equal_m(
                G, int(m), max_nodes=int(max_nodes)
            )
            if ok:
                status.update(label="Dekompozicija pronađena ✔️", state="complete")
            else:
                status.update(label="Nije pronađeno u zadanom limitu ❌", state="error")

        # ---- Popis kandidata po bridovima (uvijek koristan za analizu) ----
        st.subheader("Popis generiranih m-ciklusa po bridovima (kandidati)")
        with st.expander("Prikaži popis po bridovima", expanded=False):
            edge_list = sorted(tuple(sorted(e)) for e in G.edges())
            idx_map_all = {cyc: i for i, cyc in enumerate(all_cycles, start=1)}
            for e in edge_list:
                cand = edge2cycles.get(tuple(sorted(e)), [])
                if not cand:
                    st.markdown(f"- **Brid {e}**: (nema kandidata)")
                    continue
                lines = [f"**C{idx_map_all[cyc]}** {format_cycle(cyc)}" for cyc in cand]
                st.markdown(f"- **Brid {e}** pokrivaju: " + "; ".join(lines))

            # Tablični prikaz + export
            rows = []
            for e in edge_list:
                for cyc in edge2cycles.get(tuple(sorted(e)), []):
                    rows.append({
                        "Brid": e,
                        "CiklusID": idx_map_all[cyc],
                        "Ciklus": format_cycle(cyc)
                    })
            if rows:
                df = pd.DataFrame(rows).sort_values(["Brid", "CiklusID"])
                st.dataframe(df, use_container_width=True)
                csv = df.to_csv(index=False) #.encode("utf-8")
                st.download_button(
                    "Preuzmi popis kandidata (CSV)",
                    data=csv,
                    file_name="m_ciklusi_po_bridovima_kandidati.csv",
                    mime="text/csv"
                )

        # ---- Ako je nađena dekompozicija: vizual + sažetak + export ----
        if ok:
            with right:
                # indeksiraj odabrane cikluse zasebno (radi boja i legendi)
                idx_map_selected = {cyc: i for i, cyc in enumerate(cycles, start=1)}
                fig1 = draw_cycles_colored(
                    G, cycles, pos,
                    title=f"Dekompozicija na {len(cycles)} ciklusa duljine m={int(m)}",
                    idx_map=idx_map_selected
                )
                st.pyplot(fig1, clear_figure=True)

            # Sažetak i provjera
            used_edges = set()
            for cyc in cycles:
                used_edges.update(edges_of_cycle(cyc))

            all_edges = set(tuple(sorted(e)) for e in G.edges())
            missing = all_edges - used_edges
            extra = used_edges - all_edges

            st.subheader("Sažetak dekompozicije")
            length_counts = Counter(len(c) for c in cycles)
            df = pd.DataFrame({
                "Duljina ciklusa": list(length_counts.keys()),
                "Broj ciklusa": list(length_counts.values())
            })

            col1, col2, col3 = st.columns([1, 1, 1])

            with col1:
                st.table(df)

            # Legenda ciklusa (ID -> čvorovi)
            with st.expander("Legenda ciklusa (odabrani u dekompoziciji)", expanded=False):
                for cyc in cycles:
                    cid = idx_map_selected[cyc]
                    st.markdown(f"- **C{cid}** {format_cycle(cyc)}")

            if not missing and not extra:
                st.success("Provjera: svi bridovi su pokriveni točno jednom.")
            else:
                st.error(f"Provjera nije prošla (missing={len(missing)}, extra={len(extra)}).")

            # Export odabrane dekompozicije
            rows_sel = []
            for cyc in cycles:
                cid = idx_map_selected[cyc]
                rows_sel.append({
                    "CiklusID": cid,
                    "Ciklus": format_cycle(cyc),
                    "Duljina": len(cyc)
                })
            df_sel = pd.DataFrame(rows_sel).sort_values("CiklusID")
            st.dataframe(df_sel, use_container_width=True)
            csv_sel = df_sel.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Preuzmi odabranu dekompoziciju (CSV)",
                data=csv_sel,
                file_name="dekompozicija_m_ciklusi.csv",
                mime="text/csv"
            )

        else:
            st.info(
                "Pokušaj s manjim n, drugim m, ili povećaj limit pretraživanja. "
                "Za veće grafove preporučam specijalizirane konstrukcije (cikličke/1-rotacijske, Walecki za Hamilton cikluse, Hilton–Johnson za 4/6/8)."
            )
