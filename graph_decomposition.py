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
        # remove 1-factor edges (i, i+n/2)
        half = n // 2
        for i in range(half):
            G.remove_edge(i, i + half)
    return G


def alspach_edge_target(n: int) -> int:
    """
    Number of edges in the graph we are decomposing:
    - n odd: K_n
    - n even: K_n - I
    """
    if n % 2 == 1:
        return n * (n - 1) // 2
    else:
        return n * (n - 2) // 2


def check_equal_m_conditions(n: int, m: int):
    """
    Check feasibility conditions for a decomposition of K_n (if n odd),
    or K_n - I (if n even), into cycles all of the same length m.

    Returns (ok: bool, message: str).
    """
    if n < 3:
        return False, "Broj vrhova n mora biti barem 3."
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
            # simple symmetry-breaking: enforce start < second vertex
            if len(current) == 1 and start > w:
                continue
            dfs(start, current + [w], visited | {w})

    for v in nodes:
        dfs(v, [v], {v})

    return sorted(cycles_set)


def edges_of_cycle(cycle) -> set[tuple]:
    """Return set of edges (as sorted tuples) of the given cycle."""
    c = list(cycle)
    if len(c) > 1 and c[0] == c[-1]:
        c = c[:-1]
    edges = set()
    for i in range(len(c)):
        u = c[i]
        v = c[(i + 1) % len(c)]
        edges.add(tuple(sorted((u, v))))
    return edges


def format_cycle(cycle) -> str:
    """Format a cycle tuple as (v0 – v1 – ... – v_{k-1} – v0)."""
    c = list(cycle)
    if len(c) > 1 and c[0] == c[-1]:
        c = c[:-1]
    return "(" + " - ".join(map(str, c)) + f" - {c[0]})"

# =========================================================
# Exact cover for equal-m-cycles
# =========================================================

def exact_cover_equal_m(G: nx.Graph, m: int, max_nodes: int = 200_000):
    """
    Try to decompose the edge set of G into edge-disjoint cycles of length m.

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
    used_edges = set()
    nodes_expanded = 0  # for limiting search

    def backtrack():
        nonlocal nodes_expanded, chosen, used_edges
        if len(chosen) == T:
            return len(used_edges) == target_edges

        if nodes_expanded > max_nodes:
            return False
        nodes_expanded += 1

        # pick an uncovered edge with minimal number of candidate cycles
        remaining_edges = [e for e in E if e not in used_edges]
        if not remaining_edges:
            return False

        best_edge = min(remaining_edges, key=lambda e: len(edge2cycles[e]))
        candidates = edge2cycles[best_edge]

        for cyc in candidates:
            es = cyc_edges[cyc]
            if any(e in used_edges for e in es):
                continue

            # choose this cycle
            chosen.append(cyc)
            used_edges.update(es)

            if backtrack():
                return True

            # undo
            chosen.pop()
            for e in es:
                if e not in (edge for cyc2 in chosen for edge in cyc_edges[cyc2]):
                    used_edges.discard(e)

        return False

    ok = backtrack()
    if not ok:
        chosen = []
    return ok, chosen, all_cycles, cyc_edges, edge2cycles

# =========================================================
# Crtanje grafova (UX-friendly)
# =========================================================

FIG_W, FIG_H = 4.0, 3.0  # veličina figura u inčima (širina, visina)


def draw_graph(G, pos=None, title=None):
    """
    Crta inicijalni graf s nešto manjim čvorovima / fontom kako bi dvije slike stale
    u ekran bez prevelikog scrolla.
    """
    if pos is None:
        pos = nx.spring_layout(G, seed=42)  # fiksni layout radi konzistentnosti
    fig, ax = plt.subplots()
    nx.draw_networkx_nodes(G, pos, node_size=450, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    nx.draw_networkx_edges(G, pos, width=1.3, ax=ax)
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    return fig, pos


def draw_cycles_colored(G, cycles, pos, title=None, idx_map=None):
    """
    Crta graf s obojanim m-ciklusima. Svaki ciklus dobije svoju boju.
    Ako je zadan idx_map, indeks boje slijedi ID ciklusa (C1, C2, ...).
    """
    fig, ax = plt.subplots()
    nx.draw_networkx_nodes(G, pos, node_size=450, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

    # dovoljno različitih boja
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

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Dekompozicija grafa", layout="wide")
st.title("Dekompozicija potpunog grafa na cikluse jednake duljine")

# ------------------------ SIDEBAR ------------------------
with st.sidebar:
    st.header("Ulazni parametri")
    n = st.number_input("Broj vrhova n", min_value=3, max_value=24, value=9, step=1)
    m = st.number_input(
        "Duljina ciklusa m",
        min_value=3, max_value=24, value=3, step=1,
        help="Ako je n neparan, dekomponira se Kₙ; ako je n paran, radi se s Kₙ − I."
    )
    max_nodes = st.number_input(
        "Limit koraka pretraživanja",
        min_value=1000, max_value=2_000_000, value=200_000, step=1000,
        help="Veći limit pomaže kod težih instanci, ali usporava pretragu."
    )

    run = st.button("Pokreni dekompoziciju")

st.markdown(
    """
**Napomena:**  
- Ako je *n* **neparan**, radi se s **Kₙ**; ako je **paran**, radi se s **Kₙ − I**.  
- Traži se **isključivo dekompozicija na m-cikluse**.  
"""
)

# Kontejner za poruke / status – IZNAD tabova
status_container = st.container()

# Inicijalni graf (računa se odmah)
G = complete_graph_minus_perfect_matching(int(n))
fig0, pos = draw_graph(
    G,
    title=f"Inicijalni graf ({'Kₙ' if n % 2 else 'Kₙ − I'}) za n={int(n)}"
)

# Varijable za rezultat
ok = False
ok_cond = None
cycles = []
all_cycles = []
cyc_edges = {}
edge2cycles = {}
idx_map_selected = {}

# ----------------------- POKRETANJE ALGORITMA -----------------------
if run:
    with status_container:
        ok_cond, msg = check_equal_m_conditions(int(n), int(m))
        if not ok_cond:
            st.error(f"Uvjet nije zadovoljen: {msg}")
        else:
            st.success(f"Alspach uvjet (slučaj m-ciklusa): {msg}")

            with st.status("Tražim m-ciklusnu dekompoziciju…", expanded=True) as status:
                st.write("• Generiram kandidate za m-cikluse…")
                st.write("• Pokrećem pretraživanje egzaktnog pokrivanja...")

                ok, cycles, all_cycles, cyc_edges, edge2cycles = exact_cover_equal_m(
                    G, int(m), max_nodes=int(max_nodes)
                )

                if ok:
                    status.update(
                        label="Dekompozicija pronađena ✔️ – vidi tab **Dekompozicija**.",
                        state="complete",
                    )
                    idx_map_selected = {cyc: i for i, cyc in enumerate(cycles, start=1)}
                else:
                    status.update(
                        label="U zadanom limitu nije pronađena m-ciklusna dekompozicija ❌",
                        state="error",
                    )

# mali hint u sidebaru kad je rješenje pronađeno
with st.sidebar:
    if ok:
        st.success("Dekompozicija pronađena.\nOtvori tab **Dekompozicija** za prikaz.")
    elif ok_cond and not ok and run:
        st.info("❌ Nije nađena dekompozicija za zadane parametre.")

# -------------------------- GLAVNI TABOVI --------------------------
tab_init, tab_dec = st.tabs(["Inicijalni graf", "Dekompozicija"])

# --- TAB 1: inicijalni graf ---
with tab_init:
    st.subheader("Inicijalni graf")
    st.pyplot(fig0, width=700, clear_figure=True)

# --- TAB 2: dekompozicija + detalji ---
with tab_dec:
    st.subheader("Dekompozicija grafa")

    if not run:
        st.info("Pokreni algoritam kako bi se prikazala dekompozicija.")
    elif ok_cond is False:
        st.info("Uvjet dekompozicije nije zadovoljen – nema dekompozicije za zadane parametre.")
    elif ok_cond and not ok:
        st.info(
            "U zadanom limitu pretraživanja nije pronađena m-ciklusna dekompozicija. "
            "Pokušaj s drugim parametrima ili većim limitom."
        )
    else:
        # imamo rješenje → lijevo graf, desno sažetak
        col_graf, col_det = st.columns([2, 1])

        with col_graf:
            fig1 = draw_cycles_colored(
                G,
                cycles,
                pos,
                title=f"Dekompozicija na {len(cycles)} ciklusa duljine m={int(m)}",
                idx_map=idx_map_selected,
            )
            st.pyplot(fig1, width=700, clear_figure=True)

        with col_det:
            st.markdown("### Sažetak dekompozicije")

            # osnovna provjera bridova
            used_edges = set()
            for cyc in cycles:
                used_edges.update(edges_of_cycle(cyc))

            all_edges = {tuple(sorted(e)) for e in G.edges()}
            missing = all_edges - used_edges
            extra = used_edges - all_edges

            if not missing and not extra:
                st.success("Svi bridovi grafa pokriveni su točno jednom.")
            else:
                if missing:
                    st.error(f"Nepokriveni bridovi: {sorted(missing)}")
                if extra:
                    st.error(f"Viškovi (bridovi koji nisu u grafu): {sorted(extra)}")

            # KRATKI SAŽETAK CIKLUSA (mala tablica, ne širi puno UI)
            length_counts = Counter(len(c) for c in cycles)

            st.markdown("**Kratki pregled ciklusa:**")
            for duljina, broj in sorted(length_counts.items()):
                st.markdown(f"- broj ciklusa duljine **{duljina}**: {broj}")

            # DETALJNA TABLICA – u expanderu s fiksnom visinom (scroll unutar tablice)
            rows_sel = []
            for cyc in cycles:
                cid = idx_map_selected[cyc]
                rows_sel.append({
                    "CiklusID": cid,
                    "Ciklus": format_cycle(cyc),
                    "Duljina": len(cyc)
                })
            df_sel = pd.DataFrame(rows_sel).sort_values("CiklusID")

            with st.expander("Detaljna tablica ciklusa", expanded=False):
                st.dataframe(
                    df_sel,
                    use_container_width=True,
                    height=200
                )

            # with st.expander("Legenda ciklusa", expanded=False):
            #     for cyc in cycles:
            #         cid = idx_map_selected[cyc]
            #         st.markdown(f"- **C{cid}** {format_cycle(cyc)}")

            with st.expander("Export (CSV)", expanded=False):
                rows_export = []
                for cyc in cycles:
                    cid = idx_map_selected[cyc]
                    rows_export.append(
                        {
                            "CiklusID": cid,
                            "Ciklus": format_cycle(cyc),
                            "Bridovi": [e for e in edges_of_cycle(cyc)],
                        }
                    )
                df_export = pd.DataFrame(rows_export)
                csv_sel = df_export.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Preuzmi dekompoziciju (CSV)",
                    data=csv_sel,
                    file_name="dekompozicija_m_ciklusi.csv",
                    mime="text/csv",
                )
