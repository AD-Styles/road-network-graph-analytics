# ============================================================
# NVIDIA DLI "Fundamentals of Accelerated Data Science" 학습 기반
# GPU 가속 그래프 분석 파이프라인 재현 및 확장 구현
#
# 분석 흐름
#   Section 1  : 환경 설정
#   Section 2  : 합성 도로망 데이터 생성
#   Section 3  : 데이터 전처리 (ID 정규화 · 속도 가중치)
#   Section 4  : 그래프 기본 분석 (차수 분포 · 연결성)
#   Section 5  : 거리 기반 SSSP 분석
#   Section 6  : 이동 시간 기반 SSSP 분석
#   Section 7  : 중심성 지표 5종 비교
#   Section 8  : 종합 시각화 저장
# ============================================================

import os, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import seaborn as sns

PLOTS_DIR = 'plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

plt.rcParams['figure.dpi'] = 120
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style('darkgrid')

# 영국 도로 유형별 속도 제한 (mph)
# 출처: https://www.rac.co.uk/drive/advice/legal/speed-limits/
SPEED_LIMITS = {
    'Motorway':                     70,
    'A Road':                       60,
    'B Road':                       60,
    'Local Road':                   30,
    'Local Access Road':            30,
    'Minor Road':                   30,
    'Restricted Local Access Road': 30,
    'Secondary Access Road':        30,
}
MPH_TO_MS = 1609.34 / 3600  # 1 mph → m/s

print('=' * 60)
print('  Road Network Graph Analytics Pipeline')
print('  (NVIDIA DLI Accelerated Data Science Portfolio)')
print('=' * 60)


# ─────────────────────────────────────────────────────────────
# Section 2.  합성 도로망 데이터 생성
#   실제 GB 도로망 CSV 구조(road_nodes / road_edges)를 그대로 재현.
#   원본과 동일하게 src_id 앞에 '#' 접두사가 붙은 상태로 생성하여
#   Section 3 의 전처리 과정을 검증할 수 있도록 함.
# ─────────────────────────────────────────────────────────────
def generate_road_network(grid_size: int = 35, seed: int = 42):
    rng = np.random.default_rng(seed)
    road_type_keys = list(SPEED_LIMITS.keys())
    road_type_probs = [0.08, 0.14, 0.14, 0.20, 0.12, 0.16, 0.09, 0.07]

    # ── 노드 ──────────────────────────────────────────────────
    node_ids, eastings, northings, node_types = [], [], [], []
    for row in range(grid_size):
        for col in range(grid_size):
            nid   = f'osgb{row:04d}{col:04d}'
            east  = 100_000 + col * 2_800 + int(rng.integers(-400, 400))
            north = 200_000 + row * 2_800 + int(rng.integers(-400, 400))
            ntype = rng.choice(
                ['junction', 'roundabout', 'endpoint'],
                p=[0.60, 0.10, 0.30]
            )
            node_ids.append(nid)
            eastings.append(east)
            northings.append(north)
            node_types.append(ntype)

    road_nodes = pd.DataFrame({
        'node_id': node_ids,
        'east':    eastings,
        'north':   northings,
        'type':    node_types,
    })

    # ── 엣지 ──────────────────────────────────────────────────
    src_list, dst_list, length_list, rtype_list, form_list = [], [], [], [], []

    def add_edge(s, d, base_len):
        actual_len = base_len * (1 + rng.uniform(-0.12, 0.12))
        rtype = rng.choice(road_type_keys, p=road_type_probs)
        form  = rng.choice(
            ['Single Carriageway', 'Dual Carriageway', 'Slip Road'],
            p=[0.70, 0.24, 0.06]
        )
        # 원본 파일과 동일하게 '#' 접두사 부착
        src_list.append('#' + s)
        dst_list.append('#' + d)
        length_list.append(round(float(actual_len), 1))
        rtype_list.append(rtype)
        form_list.append(form)

    for r in range(grid_size):
        for c in range(grid_size):
            cur = f'osgb{r:04d}{c:04d}'
            # 수평 연결
            if c + 1 < grid_size:
                add_edge(cur, f'osgb{r:04d}{c+1:04d}', 2_800)
            # 수직 연결
            if r + 1 < grid_size:
                add_edge(cur, f'osgb{r+1:04d}{c:04d}', 2_800)
            # 대각선 연결 (약 15%)
            if rng.random() < 0.15 and r + 1 < grid_size and c + 1 < grid_size:
                add_edge(cur, f'osgb{r+1:04d}{c+1:04d}', 3_960)
            # 역대각선 연결 (약 8%)
            if rng.random() < 0.08 and r + 1 < grid_size and c > 0:
                add_edge(cur, f'osgb{r+1:04d}{c-1:04d}', 3_960)

    road_edges = pd.DataFrame({
        'src_id': src_list,
        'dst_id': dst_list,
        'length': length_list,
        'type':   rtype_list,
        'form':   form_list,
    })

    print(f'\n[데이터 생성] 노드 {len(road_nodes):,}개 | 엣지 {len(road_edges):,}개')
    return road_nodes, road_edges


# ─────────────────────────────────────────────────────────────
# Section 3.  데이터 전처리
#   1) '#' 접두사 제거 → node_id 호환성 확보
#   2) 속도 제한 테이블 병합 → 이동 시간(초) 파생 변수 생성
#   3) 문자열 node_id → 정수 graph_id 매핑
#      (cuGraph는 정수 인덱스를 요구하므로 필수 전처리 단계)
# ─────────────────────────────────────────────────────────────
def preprocess(road_nodes: pd.DataFrame, road_edges: pd.DataFrame):
    print('\n[전처리] ID 정규화 및 속도 가중치 계산...')

    road_edges = road_edges.copy()
    road_edges['src_id'] = road_edges['src_id'].str.lstrip('#')
    road_edges['dst_id'] = road_edges['dst_id'].str.lstrip('#')

    speed_df = pd.DataFrame({
        'type':      list(SPEED_LIMITS.keys()),
        'limit_mph': list(SPEED_LIMITS.values()),
    })
    speed_df['limit_m_s'] = speed_df['limit_mph'] * MPH_TO_MS

    road_edges = road_edges.merge(speed_df[['type', 'limit_mph', 'limit_m_s']], on='type', how='left')
    road_edges['length_s'] = (road_edges['length'] / road_edges['limit_m_s']).round(2)

    # 정수 graph_id 매핑
    unique_nodes = pd.unique(pd.concat([road_edges['src_id'], road_edges['dst_id']]))
    node_map     = {nid: i for i, nid in enumerate(unique_nodes)}
    road_edges['src'] = road_edges['src_id'].map(node_map)
    road_edges['dst'] = road_edges['dst_id'].map(node_map)

    road_nodes = road_nodes[road_nodes['node_id'].isin(node_map)].copy()
    road_nodes['graph_id'] = road_nodes['node_id'].map(node_map)

    print(f'  ✔ ID 정규화 완료 | 등록 노드: {len(node_map):,}개')
    print(f'  ✔ 이동 시간 범위: {road_edges["length_s"].min():.1f}s ~ {road_edges["length_s"].max():.1f}s')
    return road_nodes, road_edges


# ─────────────────────────────────────────────────────────────
# Section 4.  그래프 구성 및 기본 분석
# ─────────────────────────────────────────────────────────────
def build_graph(road_edges: pd.DataFrame, weight_col: str = 'length') -> nx.Graph:
    """
    NetworkX 무방향 그래프 생성.
    실제 환경에서는 cuGraph.Graph + from_cudf_edgelist 사용.
    """
    t0 = time.time()
    G  = nx.from_pandas_edgelist(
        road_edges, source='src', target='dst', edge_attr=weight_col
    )
    elapsed = time.time() - t0
    deg_vals = [d for _, d in G.degree()]
    print(f'\n[그래프 구성] weight="{weight_col}"')
    print(f'  구성 시간: {elapsed:.3f}s')
    print(f'  노드 수  : {G.number_of_nodes():,}')
    print(f'  엣지 수  : {G.number_of_edges():,}')
    print(f'  평균 차수: {np.mean(deg_vals):.2f}  |  최대 차수: {max(deg_vals)}')
    print(f'  홀수 차수 노드(이상치): {sum(1 for d in deg_vals if d % 2 != 0)}개')
    return G


# ─────────────────────────────────────────────────────────────
# Section 5 & 6.  SSSP (Single Source Shortest Path)
# ─────────────────────────────────────────────────────────────
def run_sssp(G: nx.Graph, weight_col: str, label: str = '') -> tuple:
    """
    최고 차수 노드를 출발지로 Dijkstra SSSP 실행.
    cuGraph에서는: cg.sssp(G, source_vertex)
    """
    degrees     = dict(G.degree())
    src_node    = max(degrees, key=degrees.get)
    print(f'\n[SSSP · {label}] 출발 노드: {src_node} (차수 {degrees[src_node]})')

    t0   = time.time()
    dist = nx.single_source_dijkstra_path_length(G, src_node, weight=weight_col)
    elapsed = time.time() - t0

    vals = np.array(list(dist.values()))
    print(f'  ✔ 완료 ({elapsed:.3f}s) | 도달 가능: {len(dist):,}노드')
    print(f'  평균: {vals.mean():,.1f}  |  최대: {vals.max():,.1f}  |  중앙값: {np.median(vals):,.1f}')
    return dist, src_node


# ─────────────────────────────────────────────────────────────
#  Section 7.  중심성 지표 5종
#   nx-cugraph 백엔드 호환 방식으로 구현
#   실제 환경: nx.betweenness_centrality(G, backend='cugraph')
#             또는 nxcg_G = nxcg.from_networkx(G) → Type Dispatch
# ─────────────────────────────────────────────────────────────
def compute_centralities(G: nx.Graph) -> dict:
    print('\n[중심성 분석] 5종 알고리즘 실행...')
    results = {}

    specs = [
        ('degree',      '  Degree Centrality     ',
         lambda: nx.degree_centrality(G)),
        ('betweenness', '  Betweenness Centrality',
         lambda: nx.betweenness_centrality(G, k=min(300, G.number_of_nodes()), seed=42)),
        ('katz',        '  Katz Centrality       ',
         lambda: nx.katz_centrality(G, max_iter=1000, tol=1e-4)),
        ('pagerank',    '  PageRank              ',
         lambda: nx.pagerank(G, max_iter=150, tol=1e-4)),
        ('eigenvector', '  Eigenvector Centrality',
         lambda: nx.eigenvector_centrality(G, max_iter=1000, tol=1e-4)),
    ]

    for key, label, fn in specs:
        print(label, end=' ... ', flush=True)
        t0 = time.time()
        results[key] = fn()
        print(f'({time.time()-t0:.2f}s)')

    print('  ✔ 중심성 분석 완료')
    return results


# ─────────────────────────────────────────────────────────────
# Section 8.  시각화 저장
# ─────────────────────────────────────────────────────────────

# ── Plot 1: 데이터 개요 대시보드 ───────────────────────────────
def plot_data_overview(road_nodes, road_edges, G):
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('Road Network  ·  Data Overview Dashboard', fontsize=16, fontweight='bold', y=1.01)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

    palette = ['#4878CF', '#6ACC65', '#D65F5F', '#B47CC7', '#C4AD66', '#77BEDB', '#EE854A', '#4878CF']

    # (0,0) 노드 유형 분포
    ax = fig.add_subplot(gs[0, 0])
    vc = road_nodes['type'].value_counts()
    bars = ax.bar(vc.index, vc.values, color=palette[:len(vc)], edgecolor='white', linewidth=0.8)
    ax.set_title('Node Type Distribution', fontweight='bold')
    ax.set_ylabel('Count')
    ax.bar_label(bars, fmt='%d', padding=3, fontsize=9)

    # (0,1) 도로 유형별 속도 제한
    ax = fig.add_subplot(gs[0, 1])
    speed_df = pd.DataFrame({'type': list(SPEED_LIMITS.keys()), 'mph': list(SPEED_LIMITS.values())})
    speed_df = speed_df.sort_values('mph', ascending=True)
    h_bars = ax.barh(speed_df['type'], speed_df['mph'],
                     color=[palette[i % len(palette)] for i in range(len(speed_df))],
                     edgecolor='white')
    ax.set_title('Speed Limit by Road Type (mph)', fontweight='bold')
    ax.set_xlabel('mph')
    ax.bar_label(h_bars, fmt='%d', padding=3, fontsize=9)

    # (0,2) 도로 형태 파이 차트
    ax = fig.add_subplot(gs[0, 2])
    form_vc = road_edges['form'].value_counts()
    ax.pie(form_vc.values, labels=form_vc.index, autopct='%1.1f%%',
           colors=['#4878CF', '#D65F5F', '#6ACC65'],
           wedgeprops=dict(width=0.55, edgecolor='white'),
           startangle=90, textprops={'fontsize': 9})
    ax.set_title('Road Form Breakdown', fontweight='bold')

    # (1,0) 차수 분포 히스토그램
    ax = fig.add_subplot(gs[1, 0])
    deg_vals = [d for _, d in G.degree()]
    ax.hist(deg_vals, bins=range(1, max(deg_vals) + 2), color='steelblue',
            edgecolor='white', rwidth=0.85)
    ax.axvline(np.mean(deg_vals), color='red', linestyle='--', linewidth=1.5,
               label=f'Mean: {np.mean(deg_vals):.1f}')
    ax.set_title('Node Degree Distribution', fontweight='bold')
    ax.set_xlabel('Degree')
    ax.set_ylabel('Count')
    ax.legend(fontsize=9)

    # (1,1) 도로 유형별 평균 구간 길이
    ax = fig.add_subplot(gs[1, 1])
    avg_len = road_edges.groupby('type')['length'].mean().sort_values()
    ax.barh(avg_len.index, avg_len.values,
            color=[palette[i % len(palette)] for i in range(len(avg_len))],
            edgecolor='white')
    ax.set_title('Avg. Segment Length by Road Type (m)', fontweight='bold')
    ax.set_xlabel('Length (m)')

    # (1,2) 이동 시간 분포
    ax = fig.add_subplot(gs[1, 2])
    ax.hist(road_edges['length_s'].dropna(), bins=50, color='mediumseagreen',
            edgecolor='white', alpha=0.85)
    ax.set_title('Travel Time Distribution per Segment (s)', fontweight='bold')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Count')
    mean_s = road_edges['length_s'].mean()
    ax.axvline(mean_s, color='red', linestyle='--', linewidth=1.5,
               label=f'Mean: {mean_s:.1f}s')
    ax.legend(fontsize=9)

    out = os.path.join(PLOTS_DIR, '01_data_overview.png')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'  [저장] {out}')


# ── Plot 2: 도로망 공간 레이아웃 ───────────────────────────────
def plot_spatial_network(road_nodes, road_edges, dist_len, dist_time, src_len, src_time):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle('Road Network  ·  Spatial Layout & SSSP Heatmap', fontsize=14, fontweight='bold')

    node_pos = road_nodes.set_index('graph_id')[['east', 'north']].to_dict('index')

    for ax, dist_dict, src, cmap_name, label, unit in [
        (axes[0], dist_len,  src_len,  'plasma', 'Distance-based SSSP',   'm'),
        (axes[1], dist_time, src_time, 'viridis','Travel Time-based SSSP', 's'),
    ]:
        east_v, north_v, val_v = [], [], []
        for nid, d in dist_dict.items():
            if nid in node_pos:
                east_v.append(node_pos[nid]['east'])
                north_v.append(node_pos[nid]['north'])
                val_v.append(d)

        sc = ax.scatter(east_v, north_v, c=val_v, cmap=cmap_name,
                        s=8, alpha=0.80, linewidths=0)
        plt.colorbar(sc, ax=ax, label=f'Shortest Path ({unit})', shrink=0.85)

        if src in node_pos:
            ax.scatter([node_pos[src]['east']], [node_pos[src]['north']],
                       s=200, c='red', marker='*', zorder=5, label=f'Source: {src}')
        ax.set_title(label, fontweight='bold', fontsize=12)
        ax.set_xlabel('Easting (OSGB36)')
        ax.set_ylabel('Northing (OSGB36)')
        ax.legend(fontsize=9)

    out = os.path.join(PLOTS_DIR, '02_sssp_spatial.png')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'  [저장] {out}')


# ── Plot 3: SSSP 거리 분포 비교 ────────────────────────────────
def plot_sssp_distributions(dist_len, dist_time):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle('SSSP  ·  Shortest Path Distribution Comparison', fontsize=13, fontweight='bold')

    pairs = [
        (axes[0], dist_len,  'Distance (m)',   'royalblue', 'Distance-weighted SSSP'),
        (axes[1], dist_time, 'Travel Time (s)','darkorange','Time-weighted SSSP'),
    ]
    for ax, ddict, xlabel, color, title in pairs:
        vals = np.array(list(ddict.values()))
        ax.hist(vals, bins=50, color=color, edgecolor='white', alpha=0.85)
        ax.axvline(vals.mean(),   color='red',   linestyle='--', linewidth=1.5, label=f'Mean:   {vals.mean():,.1f}')
        ax.axvline(np.median(vals), color='lime', linestyle='--', linewidth=1.5, label=f'Median: {np.median(vals):,.1f}')
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Node Count')
        ax.legend(fontsize=9)

    out = os.path.join(PLOTS_DIR, '03_sssp_distribution.png')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'  [저장] {out}')


# ── Plot 4: 중심성 지표 Top-20 비교 ────────────────────────────
def plot_centrality_top20(cent: dict):
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Graph Centrality  ·  Top-20 Nodes Comparison', fontsize=15, fontweight='bold')
    axes = axes.flatten()

    color_map = {
        'degree':      '#4878CF',
        'betweenness': '#D65F5F',
        'katz':        '#6ACC65',
        'pagerank':    '#B47CC7',
        'eigenvector': '#EE854A',
    }
    titles = {
        'degree':      'Degree Centrality\n(직접 연결 수)',
        'betweenness': 'Betweenness Centrality\n(정보 흐름의 교두보)',
        'katz':        'Katz Centrality\n(전역적 영향력)',
        'pagerank':    'PageRank\n(링크 품질 가중 중요도)',
        'eigenvector': 'Eigenvector Centrality\n(이웃 중요도 반영)',
    }

    for i, (key, values) in enumerate(cent.items()):
        ax = axes[i]
        top20 = sorted(values.items(), key=lambda x: x[1], reverse=True)[:20]
        nodes, scores = zip(*top20)
        bars = ax.barh([f'Node {n}' for n in nodes], scores,
                       color=color_map[key], edgecolor='white', linewidth=0.5)
        ax.invert_yaxis()
        ax.set_title(titles[key], fontweight='bold', fontsize=10)
        ax.set_xlabel('Score')
        ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=7)

    # 마지막 패널: 5종 상관관계 히트맵
    ax = axes[5]
    cent_df = pd.DataFrame({k: v for k, v in cent.items()}).dropna()
    corr    = cent_df.corr()
    mask    = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                vmin=-1, vmax=1, linewidths=0.5, ax=ax, mask=mask,
                cbar_kws={'shrink': 0.8})
    ax.set_title('Centrality Correlation Matrix', fontweight='bold', fontsize=10)

    out = os.path.join(PLOTS_DIR, '04_centrality_top20.png')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'  [저장] {out}')


# ── Plot 5: 중심성 지표 공간 분포 ──────────────────────────────
def plot_centrality_spatial(road_nodes, cent: dict):
    keys  = list(cent.keys())
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Graph Centrality  ·  Spatial Distribution on Road Network',
                 fontsize=14, fontweight='bold')
    axes  = axes.flatten()
    cmaps = ['Blues', 'Reds', 'Greens', 'Purples', 'Oranges']

    node_df = road_nodes.set_index('graph_id')

    for i, key in enumerate(keys):
        ax   = axes[i]
        cval = cent[key]
        matched = node_df.index.intersection(list(cval.keys()))
        xs    = node_df.loc[matched, 'east'].values
        ys    = node_df.loc[matched, 'north'].values
        cs    = np.array([cval[n] for n in matched])

        sc = ax.scatter(xs, ys, c=cs, cmap=cmaps[i], s=6, alpha=0.80, linewidths=0)
        plt.colorbar(sc, ax=ax, shrink=0.80)
        ax.set_title(f'{key.capitalize()} Centrality', fontweight='bold', fontsize=11)
        ax.set_xlabel('Easting')
        ax.set_ylabel('Northing')

    # 마지막 패널: 차수 vs PageRank 산점도
    ax   = axes[5]
    G_dummy = nx.Graph()  # 범례용 더미 – 실제 데이터는 cent에서 추출
    deg  = cent['degree']
    pr   = cent['pagerank']
    common = set(deg) & set(pr)
    deg_v  = np.array([deg[n] for n in common])
    pr_v   = np.array([pr[n]  for n in common])
    ax.scatter(deg_v, pr_v, s=10, alpha=0.50, c='steelblue', edgecolors='none')
    ax.set_title('Degree vs PageRank Centrality', fontweight='bold', fontsize=11)
    ax.set_xlabel('Degree Centrality')
    ax.set_ylabel('PageRank')
    z = np.polyfit(deg_v, pr_v, 1)
    p = np.poly1d(z)
    x_line = np.linspace(deg_v.min(), deg_v.max(), 200)
    ax.plot(x_line, p(x_line), 'r--', linewidth=1.5, label='Trend')
    ax.legend(fontsize=9)

    out = os.path.join(PLOTS_DIR, '05_centrality_spatial.png')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'  [저장] {out}')


# ── Plot 6: nx-cugraph 3가지 가속 방식 비교 요약 ──────────────
def plot_backend_methods_summary(cent: dict):
    """
    nx-cugraph 의 3가지 백엔드 활용 방식
    (Environment Variable / Backend Keyword / Type Dispatch)
    을 코드 패턴과 함께 시각적으로 정리한 레퍼런스 플롯.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle('nx-cugraph  ·  Three Backend Dispatch Methods', fontsize=14, fontweight='bold')

    methods = [
        {
            'title': '① Environment Variable',
            'code': (
                "NETWORKX_AUTOMATIC_BACKENDS=cugraph\n"
                "python -m cudf.pandas script.py\n\n"
                "# 코드 변경 없이\n"
                "# 모든 지원 알고리즘 자동 가속"
            ),
            'pros': ['코드 수정 불필요', '기존 스크립트 즉시 적용', '배치 파이프라인에 적합'],
            'cons': ['세밀한 제어 불가', '미지원 API 자동 폴백'],
            'color': '#4878CF',
        },
        {
            'title': '② Backend Keyword Argument',
            'code': (
                "import networkx as nx\n\n"
                "b = nx.betweenness_centrality(\n"
                "      G,\n"
                "      backend='cugraph'\n"
                ")"
            ),
            'pros': ['함수별 명시적 지정', '기존 nx.Graph 재활용', '부분 가속 가능'],
            'cons': ['함수 호출마다 명시 필요', '변환 오버헤드 발생 가능'],
            'color': '#D65F5F',
        },
        {
            'title': '③ Type-Based Dispatching',
            'code': (
                "import nx_cugraph as nxcg\n\n"
                "nxcg_G = nxcg.from_networkx(G)\n"
                "# 변환 1회 발생\n\n"
                "p = nx.pagerank(nxcg_G)\n"
                "# 이후 변환 없이 GPU 실행"
            ),
            'pros': ['변환 최소화(1회)', '반복 호출 시 최고 성능', '명확한 타입 보장'],
            'cons': ['초기 변환 비용 발생', 'nxcg 임포트 필요'],
            'color': '#6ACC65',
        },
    ]

    for ax, m in zip(axes, methods):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # 제목 박스
        ax.add_patch(plt.Rectangle((0, 0.88), 1, 0.12, color=m['color'], alpha=0.85, transform=ax.transAxes))
        ax.text(0.5, 0.94, m['title'], ha='center', va='center',
                fontsize=11, fontweight='bold', color='white', transform=ax.transAxes)

        # 코드 블록
        ax.add_patch(plt.Rectangle((0.02, 0.52), 0.96, 0.34,
                                   color='#1e1e1e', transform=ax.transAxes, zorder=2))
        ax.text(0.05, 0.85, m['code'], ha='left', va='top',
                fontsize=8.5, color='#d4d4d4', fontfamily='monospace',
                transform=ax.transAxes, zorder=3)

        # 장점
        ax.text(0.05, 0.48, '✅ Pros', ha='left', va='top',
                fontsize=9, fontweight='bold', color='#2e7d32', transform=ax.transAxes)
        for j, pro in enumerate(m['pros']):
            ax.text(0.07, 0.42 - j * 0.07, f'· {pro}', ha='left', va='top',
                    fontsize=8.5, color='#333333', transform=ax.transAxes)

        # 단점
        ax.text(0.05, 0.20, '⚠️ Cons', ha='left', va='top',
                fontsize=9, fontweight='bold', color='#c62828', transform=ax.transAxes)
        for j, con in enumerate(m['cons']):
            ax.text(0.07, 0.14 - j * 0.07, f'· {con}', ha='left', va='top',
                    fontsize=8.5, color='#555555', transform=ax.transAxes)

    out = os.path.join(PLOTS_DIR, '06_backend_methods.png')
    plt.savefig(out, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  [저장] {out}')


# ── Plot 7: 핵심 수치 요약 카드 ────────────────────────────────
def plot_summary_card(G, dist_len, dist_time, cent, road_edges):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    fig.patch.set_facecolor('#1a1a2e')

    title_cfg = dict(ha='center', va='center', transform=ax.transAxes,
                     color='white', fontweight='bold')

    ax.text(0.5, 0.95, 'Road Network Graph Analytics  ─  Key Metrics Summary',
            fontsize=15, **title_cfg)

    deg_vals  = [d for _, d in G.degree()]
    d_vals    = np.array(list(dist_len.values()))
    t_vals    = np.array(list(dist_time.values()))

    metrics = [
        ('Nodes',           f"{G.number_of_nodes():,}",         '#4878CF'),
        ('Edges',           f"{G.number_of_edges():,}",         '#6ACC65'),
        ('Avg Degree',      f"{np.mean(deg_vals):.2f}",         '#EE854A'),
        ('Max Degree',      f"{max(deg_vals)}",                 '#D65F5F'),
        ('SSSP Mean (m)',   f"{d_vals.mean():,.0f}",            '#B47CC7'),
        ('SSSP Mean (s)',   f"{t_vals.mean():,.0f}",            '#77BEDB'),
        ('Road Types',      f"{road_edges['type'].nunique()}",  '#C4AD66'),
        ('Avg Speed (mph)', f"{road_edges['limit_mph'].mean():.0f}", '#4878CF'),
    ]

    cols, rows = 4, 2
    w, h = 0.22, 0.28
    for idx, (label, value, color) in enumerate(metrics):
        col = idx % cols
        row = idx // cols
        x   = 0.06 + col * 0.235
        y   = 0.58 - row * 0.35

        rect = plt.Rectangle((x, y), w, h, color=color, alpha=0.20,
                              transform=ax.transAxes, zorder=1)
        ax.add_patch(rect)
        border = plt.Rectangle((x, y), w, h, fill=False,
                                edgecolor=color, linewidth=2,
                                transform=ax.transAxes, zorder=2)
        ax.add_patch(border)

        ax.text(x + w / 2, y + h * 0.62, value,
                ha='center', va='center', transform=ax.transAxes,
                color='white', fontsize=18, fontweight='bold', zorder=3)
        ax.text(x + w / 2, y + h * 0.22, label,
                ha='center', va='center', transform=ax.transAxes,
                color='#aaaaaa', fontsize=9, zorder=3)

    out = os.path.join(PLOTS_DIR, '07_summary_card.png')
    plt.savefig(out, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f'  [저장] {out}')


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Section 2
    road_nodes_raw, road_edges_raw = generate_road_network(grid_size=35, seed=42)

    # Section 3
    road_nodes, road_edges = preprocess(road_nodes_raw, road_edges_raw)

    # Section 4 – 거리 기반 그래프
    G_dist = build_graph(road_edges, weight_col='length')
    # 이동 시간 기반 그래프
    G_time = build_graph(road_edges, weight_col='length_s')

    # Section 5 – 거리 SSSP
    dist_len,  src_len  = run_sssp(G_dist, 'length',   label='거리(m)')
    # Section 6 – 시간 SSSP
    dist_time, src_time = run_sssp(G_time, 'length_s', label='이동시간(s)')

    # Section 7 – 중심성 분석
    centralities = compute_centralities(G_dist)

    # Section 8 – 시각화 저장
    print('\n[시각화] 플롯 저장 중...')
    plot_data_overview(road_nodes, road_edges, G_dist)
    plot_spatial_network(road_nodes, road_edges, dist_len, dist_time, src_len, src_time)
    plot_sssp_distributions(dist_len, dist_time)
    plot_centrality_top20(centralities)
    plot_centrality_spatial(road_nodes, centralities)
    plot_backend_methods_summary(centralities)
    plot_summary_card(G_dist, dist_len, dist_time, centralities, road_edges)

    # 최종 요약
    print('\n' + '=' * 60)
    print('  분석 완료 요약')
    print('=' * 60)

    cent_df = pd.DataFrame(centralities)
    for key in centralities:
        top3 = sorted(centralities[key].items(), key=lambda x: x[1], reverse=True)[:3]
        nodes_str = ', '.join([f'Node {n}({s:.4f})' for n, s in top3])
        print(f'  {key:<14}: Top-3 → {nodes_str}')

    print(f'\n  그래프 노드 수  : {G_dist.number_of_nodes():,}')
    print(f'  그래프 엣지 수  : {G_dist.number_of_edges():,}')
    print(f'  저장된 플롯     : {len(os.listdir(PLOTS_DIR))}개 → ./{PLOTS_DIR}/')
    print('=' * 60)
