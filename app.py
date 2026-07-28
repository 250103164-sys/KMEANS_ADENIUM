import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
import plotly.express as px
import io


REQUIRED_COLUMNS = [
    'NO', 'TANGGAL', 'VARIETAS',
    'JUMLAH TERJUAL', 'HARGA SATUAN', 'JUMLAH HARGA'
]


def bersihkan_rupiah(nilai):
    if pd.isna(nilai):
        return np.nan

    teks = str(nilai).strip()

    teks = teks.replace('Rp', '').replace('RP', '').replace('rp', '').replace(' ', '')

    if teks.lower() in ['', '-', '--', 'nan', 'none', 'null']:
        return np.nan

    if re.fullmatch(r'-?\d{1,3}(\.\d{3})+', teks):
        teks = teks.replace('.', '')
    elif '.' in teks and ',' in teks:
        teks = teks.replace('.', '').replace(',', '.')
    elif ',' in teks:
        teks = teks.replace(',', '.')

    return pd.to_numeric(teks, errors='coerce')


def load_csv_uploaded(uploaded_file):
    uploaded_file.seek(0)
    try:
        df = pd.read_csv(
            uploaded_file,
            encoding='cp1252',
            sep=None,
            engine='python',
            dtype=str
        )
    except Exception:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, dtype=str)
    return df


def main():
    st.set_page_config(page_title="K-Means Clustering Adenium", layout="wide")

    # --- Stylish CSS theme ---
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #e7f6e7 0%, #f4fff4 45%, #f9fff9 100%);
            color: #134f2c;
            font-family: 'Inter', sans-serif;
        }
        .header-title {
            font-size: 2.75rem;
            font-weight: 800;
            color: #0f4825;
            line-height: 1.1;
            margin-bottom: 0.2rem;
        }
        .subtle {
            color: #2f5b3f;
            font-size: 1rem;
            margin-bottom: 1rem;
        }
        .card {
            background: #ffffff;
            padding: 1.4rem;
            border-radius: 20px;
            box-shadow: 0 18px 45px rgba(27, 91, 47, 0.08);
            border: 1px solid rgba(27, 91, 47, 0.12);
            margin-bottom: 1.25rem;
        }
        .card.small {
            padding: 1rem;
        }
        .metric-box {
            background: #e9f6ea;
            padding: 1rem;
            border-radius: 16px;
            border: 1px solid rgba(27, 91, 47, 0.12);
        }
        .metric-title {
            color: #2f5b3f;
            font-size: 0.95rem;
            margin-bottom: 0.35rem;
        }
        .metric-value {
            color: #11401e;
            font-size: 1.55rem;
            font-weight: 700;
        }
        .metric-sub {
            color: #5c7a62;
            font-size: 0.9rem;
        }
        .stSidebar .sidebar-content {
            background: linear-gradient(180deg, #f8fff8 0%, #e9f6ea 100%);
        }
        .sidebar .sidebar-content {
            padding-top: 1rem;
        }
        .sidebar .st-bf {
            color: #11401e;
        }
        .stButton>button {
            background-color: #2f7d4a;
            color: white;
            border-radius: 10px;
            border: none;
            padding: 0.6rem 1rem;
        }
        .stButton>button:hover {
            background-color: #266f3e;
        }
        .stTabs [role="tab"] {
            font-weight: 600;
        }
        .stTabs [role="tab"]:not(.is-selected) {
            color: #365e41;
        }
        .stTabs [role="tab"].is-selected {
            color: #0f4825;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="header-title">K-Means Clustering — Penjualan Adenium</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtle">Unggah dataset, bersihkan, eksplorasi, dan jalankan clustering secara interaktif.</div>', unsafe_allow_html=True)

    st.sidebar.header("Upload & Pengaturan")
    uploaded_file = st.sidebar.file_uploader("Unggah CSV dataset (cp1252 rekomendasi)", type=["csv"])
    hapus_duplikat = st.sidebar.checkbox("Hapus duplikat (default: True)", value=True)
    harga_dalam_ribuan = st.sidebar.checkbox("Harga dalam ribuan (kalikan 1000)", value=False)
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **Panduan singkat**
        - Unggah dataset CSV
        - Gunakan kolom numerik yang valid
        - Pilih K untuk clustering
        - Lihat hasil di tab Results
        """
    )

    st.sidebar.markdown("---")
    menu_utama = st.sidebar.radio(
        "Menu Utama",
        ["Data", "Preprocessing", "EDA", "Clustering", "Results"],
        index=0,
    )

    if uploaded_file is None:
        st.info("Unggah file CSV untuk memulai.")
        return

    df_raw = load_csv_uploaded(uploaded_file)

    if menu_utama == "Data":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Preview dataset (awal)")
        st.caption(f"Ukuran dataset: {df_raw.shape}")
        st.dataframe(df_raw.head())
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # Bersihkan nama kolom
    df_raw.columns = (
        df_raw.columns.astype(str)
        .str.replace('\n', ' ', regex=False)
        .str.replace('\r', ' ', regex=False)
        .str.replace('\xa0', ' ', regex=False)
        .str.strip()
        .str.upper()
        .str.replace(r'\s+', ' ', regex=True)
    )

    missing = [c for c in REQUIRED_COLUMNS if c not in df_raw.columns]
    if missing:
        st.error(f"Kolom yang diperlukan tidak ditemukan: {missing}")
        st.stop()

    data = df_raw[REQUIRED_COLUMNS].copy()

    simbol_kosong = ['', ' ', '-', '--', 'NA', 'N/A', 'NAN', 'NULL', 'NONE']
    data = data.replace(simbol_kosong, np.nan)

    data['TANGGAL'] = pd.to_datetime(data['TANGGAL'], dayfirst=True, errors='coerce')
    data['TAHUN PENJUALAN'] = data['TANGGAL'].dt.year.astype('Int64')

    data['JUMLAH TERJUAL'] = (
        data['JUMLAH TERJUAL'].astype('string').str.strip().str.replace(',', '.', regex=False)
    )
    data['JUMLAH TERJUAL'] = pd.to_numeric(data['JUMLAH TERJUAL'], errors='coerce')

    data['HARGA SATUAN'] = data['HARGA SATUAN'].apply(bersihkan_rupiah)
    data['JUMLAH HARGA'] = data['JUMLAH HARGA'].apply(bersihkan_rupiah)

    if harga_dalam_ribuan:
        kondisi_harga = data['HARGA SATUAN'].notna() & (data['HARGA SATUAN'] > 0) & (data['HARGA SATUAN'] < 1000)
        data.loc[kondisi_harga, 'HARGA SATUAN'] = data.loc[kondisi_harga, 'HARGA SATUAN'] * 1000

        kondisi_total = data['JUMLAH HARGA'].notna() & (data['JUMLAH HARGA'] > 0) & (data['JUMLAH HARGA'] < 1000)
        data.loc[kondisi_total, 'JUMLAH HARGA'] = data.loc[kondisi_total, 'JUMLAH HARGA'] * 1000

    kondisi_valid = data['JUMLAH HARGA'].notna() & data['JUMLAH TERJUAL'].notna() & (data['JUMLAH TERJUAL'] > 0)
    data.loc[kondisi_valid, 'HARGA SATUAN'] = data.loc[kondisi_valid, 'JUMLAH HARGA'] / data.loc[kondisi_valid, 'JUMLAH TERJUAL']

    data['JUMLAH HARGA HITUNG'] = data['JUMLAH TERJUAL'] * data['HARGA SATUAN']
    data['JUMLAH HARGA'] = data['JUMLAH HARGA HITUNG']

    kolom_pemeriksaan_duplikat = ['TANGGAL', 'VARIETAS', 'JUMLAH TERJUAL', 'HARGA SATUAN', 'JUMLAH HARGA']
    if hapus_duplikat:
        data = data.drop_duplicates(subset=kolom_pemeriksaan_duplikat, keep='first').copy()

    kolom_wajib = ['TANGGAL', 'VARIETAS', 'JUMLAH TERJUAL', 'HARGA SATUAN', 'JUMLAH HARGA']
    jumlah_sebelum = len(data)
    data = data.dropna(subset=kolom_wajib).copy()
    data = data[(data['JUMLAH TERJUAL'] > 0) & (data['HARGA SATUAN'] > 0) & (data['JUMLAH HARGA'] > 0)].copy()
    data = data.reset_index(drop=True)

    if menu_utama == "Preprocessing":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Data setelah pembersihan")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.markdown('<div class="metric-title">Total Baris Setelah Bersih</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{len(data):,}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.markdown('<div class="metric-title">Baris Dihapus</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{jumlah_sebelum - len(data):,}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        st.write('---')
        st.dataframe(data.head())
        st.markdown('</div>', unsafe_allow_html=True)

    # Normalisasi nama varietas
    data['VARIETAS'] = (
        data['VARIETAS'].astype('string').str.strip().str.upper().str.replace(r'\s+', ' ', regex=True)
    )

    perbaikan_varietas = {'ARABICUM': 'ARABIKUM', 'AKAR  MAS': 'AKAR MAS', 'BLACK  DRAGON': 'BLACK DRAGON'}
    data['VARIETAS'] = data['VARIETAS'].replace(perbaikan_varietas)

    # Agregasi per varietas
    data_agregasi = (
        data.groupby('VARIETAS', as_index=False)
        .agg(
            TOTAL_KUANTITAS_TERJUAL=('JUMLAH TERJUAL', 'sum'),
            JUMLAH_TRANSAKSI=('VARIETAS', 'size'),
            TOTAL_NILAI_PENJUALAN=('JUMLAH HARGA', 'sum')
        )
    )

    data_agregasi['RATA_RATA_HARGA'] = data_agregasi['TOTAL_NILAI_PENJUALAN'] / data_agregasi['TOTAL_KUANTITAS_TERJUAL']

    data_agregasi = data_agregasi[[
        'VARIETAS', 'RATA_RATA_HARGA', 'TOTAL_KUANTITAS_TERJUAL', 'JUMLAH_TRANSAKSI'
    ]].copy()

    data_agregasi['RATA_RATA_HARGA'] = data_agregasi['RATA_RATA_HARGA'].round().astype('Int64')
    data_agregasi['TOTAL_KUANTITAS_TERJUAL'] = data_agregasi['TOTAL_KUANTITAS_TERJUAL'].round().astype('Int64')
    data_agregasi['JUMLAH_TRANSAKSI'] = data_agregasi['JUMLAH_TRANSAKSI'].astype('Int64')

    data_agregasi = data_agregasi.sort_values('TOTAL_KUANTITAS_TERJUAL', ascending=False).reset_index(drop=True)

    if menu_utama == "EDA":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Data agregasi per VARIETAS")
        st.write(f"Jumlah varietas: {len(data_agregasi)}")
        st.dataframe(data_agregasi.head(200))

        fitur_clustering = ['RATA_RATA_HARGA', 'TOTAL_KUANTITAS_TERJUAL', 'JUMLAH_TRANSAKSI']

        st.write("### Statistik deskriptif fitur clustering")
        statistik_deskriptif = pd.DataFrame({
            'JUMLAH DATA': data_agregasi[fitur_clustering].count(),
            'MINIMUM': data_agregasi[fitur_clustering].min(),
            'MAKSIMUM': data_agregasi[fitur_clustering].max(),
            'RATA-RATA': data_agregasi[fitur_clustering].mean(),
            'MEDIAN': data_agregasi[fitur_clustering].median(),
            'STANDAR DEVIASI': data_agregasi[fitur_clustering].std()
        })
        st.dataframe(statistik_deskriptif.round(2))

        # Features summary cards
        col_a, col_b, col_c = st.columns(3)
        summary = {
            'RATA_RATA_HARGA': 'Average Price',
            'TOTAL_KUANTITAS_TERJUAL': 'Total Quantity',
            'JUMLAH_TRANSAKSI': 'Total Transactions'
        }
        for widget_col, name in zip([col_a, col_b, col_c], fitur_clustering):
            with widget_col:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-title">{summary[name]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{data_agregasi[name].mean():,.0f}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-sub">Mean dari {name.replace("_", " ").title()}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        st.write('---')
        st.write("### Distribusi fitur clustering")
        for kolom in fitur_clustering:
            fig = px.histogram(data_agregasi, x=kolom, nbins=20, title=f'Distribusi {kolom}')
            st.plotly_chart(fig, use_container_width=True)

        fig2 = px.box(data_agregasi, x='RATA_RATA_HARGA', points='all', title='Boxplot RATA_RATA_HARGA')
        st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.box(data_agregasi, x='TOTAL_KUANTITAS_TERJUAL', points='all', title='Boxplot TOTAL_KUANTITAS_TERJUAL')
        st.plotly_chart(fig3, use_container_width=True)

        fig4 = px.box(data_agregasi, x='JUMLAH_TRANSAKSI', points='all', title='Boxplot JUMLAH_TRANSAKSI')
        st.plotly_chart(fig4, use_container_width=True)

        # Scatter pairs
        st.write("### Scatter pairs")
        pasangan_variabel = [
            ('RATA_RATA_HARGA', 'TOTAL_KUANTITAS_TERJUAL'),
            ('RATA_RATA_HARGA', 'JUMLAH_TRANSAKSI'),
            ('TOTAL_KUANTITAS_TERJUAL', 'JUMLAH_TRANSAKSI')
        ]
        for xcol, ycol in pasangan_variabel:
            fig = px.scatter(data_agregasi, x=xcol, y=ycol, title=f'{xcol} vs {ycol}', hover_name='VARIETAS')
            st.plotly_chart(fig, use_container_width=True)

        # Correlation matrix
        matriks_korelasi = data_agregasi[fitur_clustering].astype(float).corr()
        st.write('### Matriks korelasi')
        fig_corr = px.imshow(matriks_korelasi, text_auto='.2f', title='Matriks Korelasi Variabel Clustering', color_continuous_scale='Blues')
        st.plotly_chart(fig_corr, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Prepare X for clustering
    X = data_agregasi[fitur_clustering].apply(pd.to_numeric, errors='coerce')
    if X.isna().any().any():
        st.error('Masih terdapat data kosong pada variabel clustering. Periksa input.')
        st.stop()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    jumlah_varietas = X_scaled.shape[0]
    if jumlah_varietas < 3:
        st.error('Jumlah varietas terlalu sedikit untuk clustering (minimal 3).')
        st.stop()

    K_MAKSIMUM = min(10, jumlah_varietas)
    if menu_utama == "Clustering":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write(f'Jumlah varietas: {jumlah_varietas} — menguji K 1..{K_MAKSIMUM} untuk Elbow')

        daftar_k = range(1, K_MAKSIMUM + 1)
        daftar_wcss = []
        for k in daftar_k:
            model_tmp = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
            model_tmp.fit(X_scaled)
            daftar_wcss.append(model_tmp.inertia_)

        tabel_wcss = pd.DataFrame({'JUMLAH_CLUSTER': list(daftar_k), 'WCSS': daftar_wcss})
        tabel_wcss['PENURUNAN_WCSS'] = tabel_wcss['WCSS'].shift(1) - tabel_wcss['WCSS']
        tabel_wcss['PERSENTASE_PENURUNAN'] = tabel_wcss['PENURUNAN_WCSS'] / tabel_wcss['WCSS'].shift(1) * 100

        st.write('### Tabel WCSS (Elbow)')
        st.dataframe(tabel_wcss.round(4))

        fig_elbow = px.line(tabel_wcss, x='JUMLAH_CLUSTER', y='WCSS', markers=True, title='Elbow Method (WCSS)')
        st.plotly_chart(fig_elbow, use_container_width=True)

        # Choose K
        k_default = 3 if 3 <= K_MAKSIMUM else 2
        k_optimal = st.sidebar.slider('Pilih jumlah cluster (K)', min_value=2, max_value=K_MAKSIMUM, value=k_default)

        # Run final KMeans
        model_kmeans = KMeans(n_clusters=k_optimal, init='k-means++', n_init=10, random_state=42)
        labels = model_kmeans.fit_predict(X_scaled)

        hasil_clustering = data_agregasi.copy()
        hasil_clustering['CLUSTER'] = labels + 1

        # Silhouette
        if k_optimal > 1:
            nilai_silhouette = silhouette_score(X_scaled, labels)
            hasil_clustering['SILHOUETTE_INDIVIDU'] = silhouette_samples(X_scaled, labels)
            st.write(f'### Silhouette Score: {nilai_silhouette:.4f}')
        else:
            st.write('Silhouette tidak tersedia untuk K=1')

        st.write('### Hasil clustering (sample)')
        st.dataframe(hasil_clustering.head(200))

        # Cluster counts
        counts = hasil_clustering['CLUSTER'].value_counts().sort_index()
        fig_counts = px.bar(x=counts.index.astype(str), y=counts.values, labels={'x': 'Cluster', 'y': 'Jumlah Varietas'}, title='Jumlah anggota per cluster')
        st.plotly_chart(fig_counts, use_container_width=True)

        # Centroids
        centroid_standardisasi = model_kmeans.cluster_centers_
        centroid_asli = scaler.inverse_transform(centroid_standardisasi)
        tabel_centroid = pd.DataFrame(centroid_asli, columns=fitur_clustering)
        tabel_centroid.insert(0, 'CLUSTER', range(1, k_optimal + 1))
        tabel_centroid[fitur_clustering] = tabel_centroid[fitur_clustering].round(2)
        st.write('### Tabel centroid (skala asli)')
        st.dataframe(tabel_centroid)

        # Plot cluster scatter
        fig_scatter = px.scatter(hasil_clustering, x='TOTAL_KUANTITAS_TERJUAL', y='JUMLAH_TRANSAKSI', color='CLUSTER', hover_name='VARIETAS', title='Hasil Clustering: Kuantitas vs Transaksi')
        st.plotly_chart(fig_scatter, use_container_width=True)

        fig_scatter2 = px.scatter(hasil_clustering, x='RATA_RATA_HARGA', y='TOTAL_KUANTITAS_TERJUAL', color='CLUSTER', hover_name='VARIETAS', title='Cluster berdasarkan Harga dan Kuantitas')
        st.plotly_chart(fig_scatter2, use_container_width=True)

        # Centroid standardized bar chart
        centroid_z = pd.DataFrame(centroid_standardisasi, columns=fitur_clustering)
        centroid_z.insert(0, 'CLUSTER', range(1, k_optimal + 1))
        fig_centroid = px.bar(centroid_z.melt(id_vars='CLUSTER', var_name='Fitur', value_name='Nilai'), x='CLUSTER', y='Nilai', color='Fitur', barmode='group', title='Profil Centroid (standardized)')
        st.plotly_chart(fig_centroid, use_container_width=True)


    # Karakteristik cluster
    karakteristik_cluster = centroid_z.copy()
    def kategori_centroid(nilai):
        if nilai > 0.5:
            return 'Tinggi'
        elif nilai < -0.5:
            return 'Rendah'
        else:
            return 'Sedang'

    for kol in fitur_clustering:
        karakteristik_cluster[kol] = karakteristik_cluster[kol].apply(kategori_centroid)

    karakteristik_cluster = karakteristik_cluster.rename(columns={
        'RATA_RATA_HARGA': 'KATEGORI_HARGA',
        'TOTAL_KUANTITAS_TERJUAL': 'KATEGORI_KUANTITAS',
        'JUMLAH_TRANSAKSI': 'KATEGORI_TRANSAKSI'
    })

    karakteristik_cluster['KARAKTERISTIK'] = (
        'Harga ' + karakteristik_cluster['KATEGORI_HARGA'] + ', kuantitas ' + karakteristik_cluster['KATEGORI_KUANTITAS'] + ', transaksi ' + karakteristik_cluster['KATEGORI_TRANSAKSI']
    )

    hasil_clustering = hasil_clustering.merge(karakteristik_cluster[['CLUSTER', 'KARAKTERISTIK']], on='CLUSTER', how='left')

    # Cluster picker
    if menu_utama == "Results":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write('### Detail cluster')
        cluster_selected = st.selectbox('Pilih cluster', options=sorted(hasil_clustering['CLUSTER'].unique()))
        anggota = hasil_clustering[hasil_clustering['CLUSTER'] == cluster_selected]
        st.write(f'Karakteristik cluster: {karakteristik_cluster.loc[karakteristik_cluster.CLUSTER==cluster_selected, "KARAKTERISTIK"].iloc[0]}')
        st.write(f'Jumlah varietas: {len(anggota)}')
        st.dataframe(anggota[['VARIETAS', 'RATA_RATA_HARGA', 'TOTAL_KUANTITAS_TERJUAL', 'JUMLAH_TRANSAKSI', 'SILHOUETTE_INDIVIDU']])

        # Plot anggota cluster
        fig_cluster_only = px.scatter(anggota, x='TOTAL_KUANTITAS_TERJUAL', y='JUMLAH_TRANSAKSI', hover_name='VARIETAS', title=f'Anggota Cluster {cluster_selected}')
        st.plotly_chart(fig_cluster_only, use_container_width=True)

        # Download hasil
        csv = hasil_clustering.to_csv(index=False).encode('utf-8')
        st.download_button('Download hasil clustering', data=csv, file_name='hasil_clustering.csv', mime='text/csv')
        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == '__main__':
    main()
