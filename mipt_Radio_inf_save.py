import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table
import warnings
from astropy.io.votable.exceptions import VOWarning

warnings.simplefilter('ignore', category=VOWarning)

# ============================================================
#  ЗАШИТЫЕ ПУТИ (в формате WSL/Linux)
# ============================================================
FILE1 = "/mnt/c/Users/User/Documents/mipt/vizier_votable_full_data_withoutfilter.tsv"
FILE2 = "/mnt/c/Users/User/Downloads/Coordinate_Data (1).csv"
# ============================================================

# ============================================================
#  НАСТРОЙКИ ФИЛЬТРАЦИИ (19 условий с картинки)
# ============================================================
x = 15
MIN_NF = 3             # Nf >= 3
MIN_R = x            # r >= 5
MAX_F = 4              # f < 4
MIN_GOOD_FILTERS = 3   # минимум 3 хороших фильтра
# ============================================================


def read_and_filter_votable(path):
    """Стриминговое чтение VOTable (построчно) + правильная логика фильтрации."""
    print(f"Стриминговое чтение VOTable: {path}")
    
    columns = None
    data_rows = []
    in_cdata = False
    
    idx_nf = None
    idx_r = {}
    idx_x = {}
    idx_f = {}
    filter_names = ['UVW2', 'UVM2', 'UVW1', 'U', 'B', 'V']
    
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.rstrip('\n\r')
            
            if not in_cdata:
                if '<![CDATA[' in line:
                    in_cdata = True
                    line = line.split('<![CDATA[', 1)[1]
                else:
                    continue
            
            if ']]>' in line:
                line = line.split(']]>', 1)[0]
            
            if not line.strip():
                continue
            
            if columns is None:
                columns = [c.strip() for c in line.split(';')]
                if columns and columns[-1] == '':
                    columns = columns[:-1]
                
                for i, col in enumerate(columns):
                    if col == 'Nf':
                        idx_nf = i
                    elif col.startswith('r') and col[1:] in filter_names:
                        idx_r[col[1:]] = i
                    elif col.startswith('x') and col[1:] in filter_names:
                        idx_x[col[1:]] = i
                    elif col.startswith('f') and col[1:] in filter_names:
                        idx_f[col[1:]] = i
                
                print(f"  Найдено колонок: {len(columns)}")
                print(f"  Nf: {idx_nf}, r: {idx_r}, x: {idx_x}, f: {idx_f}")
                continue
            
            if line.startswith(' ') or line.startswith('-') or line.startswith('Name'):
                continue
            
            parts = line.split(';')
            if len(parts) < len(columns):
                parts += [''] * (len(columns) - len(parts))
            elif len(parts) > len(columns):
                parts = parts[:len(columns)]
            
            try:
                nf_val = parts[idx_nf].strip() if idx_nf is not None else ''
                if nf_val == '' or float(nf_val) < MIN_NF:
                    continue
                
                good_filters = 0
                
                for filt in filter_names:
                    r_val = parts[idx_r[filt]].strip() if filt in idx_r else ''
                    if r_val == '': 
                        continue  
                    
                    if float(r_val) < MIN_R:
                        continue
                    
                    x_val = parts[idx_x[filt]].strip() if filt in idx_x else '1'
                    if x_val == '' or int(float(x_val)) != 0:
                        continue
                        
                    f_val = parts[idx_f[filt]].strip() if filt in idx_f else str(MAX_F)
                    if f_val == '' or int(float(f_val)) >= MAX_F:
                        continue
                    
                    good_filters += 1
                
                if good_filters >= MIN_GOOD_FILTERS:
                    data_rows.append(parts)
                    if len(data_rows) % 5000 == 0:
                        print(f"  ... найдено подходящих строк: {len(data_rows)}")
            
            except (ValueError, IndexError):
                continue

    print(f"  Итого подходящих строк: {len(data_rows)}")
    
    if len(data_rows) == 0:
        return pd.DataFrame(columns=columns if columns else [])
    
    return pd.DataFrame(data_rows, columns=columns)


def read_catalog(path):
    if path.lower().endswith('.csv'):
        print(f"Чтение CSV: {path}")
        return pd.read_csv(path)
    elif path.lower().endswith('.tsv') or 'votable' in path.lower():
        return read_and_filter_votable(path)
    else:
        with fits.open(path) as hdul:
            for hdu in hdul:
                if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
                    return Table(hdu.data).to_pandas()
        raise ValueError("Нет таблицы в FITS")


def find_column(df, candidates):
    for name in candidates:
        if name in df.columns:
            return name
    raise ValueError(f"Нет колонки. Доступные: {df.columns.tolist()}")


def cross_match(df1, df2, max_sep_arcsec=x):
    ra1_col = find_column(df1, ['RAJ2000', 'RA_ICRS', '_RAJ2000', 'RA', 'ra'])
    dec1_col = find_column(df1, ['DEJ2000', 'DEC_ICRS', '_DEJ2000', 'DEC', 'dec'])
    ra2_col = find_column(df2, ['RAJ2000', 'RA_ICRS', '_RAJ2000', 'RA', 'ra'])
    dec2_col = find_column(df2, ['DEJ2000', 'DEC_ICRS', '_DEJ2000', 'DEC', 'dec'])

    ra1 = pd.to_numeric(df1[ra1_col], errors='coerce').values
    dec1 = pd.to_numeric(df1[dec1_col], errors='coerce').values
    ra2 = pd.to_numeric(df2[ra2_col], errors='coerce').values
    dec2 = pd.to_numeric(df2[dec2_col], errors='coerce').values

    mask1 = np.isfinite(ra1) & np.isfinite(dec1)
    mask2 = np.isfinite(ra2) & np.isfinite(dec2)
    df1 = df1[mask1].reset_index(drop=True)
    ra1, dec1 = ra1[mask1], dec1[mask1]
    df2 = df2[mask2].reset_index(drop=True)
    ra2, dec2 = ra2[mask2], dec2[mask2]

    ra1 = ra1.astype(float)
    dec1 = dec1.astype(float)
    ra2 = ra2.astype(float)
    dec2 = dec2.astype(float)

    c1 = SkyCoord(ra=ra1 * u.deg, dec=dec1 * u.deg)
    c2 = SkyCoord(ra=ra2 * u.deg, dec=dec2 * u.deg)

    idx2, d2d, _ = c1.match_to_catalog_sky(c2)
    mask = d2d < max_sep_arcsec * u.arcsec
    return df1[mask].copy(), df2.iloc[idx2[mask]].copy(), d2d[mask]


def main():
    df1_filtered = read_catalog(FILE1)
    df2 = read_catalog(FILE2)
    print(f"\nИсходно: df1 (после фильтрации) = {len(df1_filtered)} строк, df2 = {len(df2)} строк")

    if len(df1_filtered) == 0:
        print("После фильтрации ничего не осталось.")
        return

    matched1, matched2, seps = cross_match(df1_filtered, df2, max_sep_arcsec=5.0)
    print(f"\nВсего совпадений (расстояние < {x} arcsec): {len(matched1)}")

    if len(matched1) > 0:
        if 'Name' in matched1.columns:
            matched1['Name'] = matched1['Name'].astype(str).str.strip()
        matched1['Sep_arcsec'] = seps.arcsec
        
        # Формируем строку таблицы один раз, чтобы не делать это дважды
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 250):
            table_str = matched1.to_string(index=False)
            print(table_str)
        
        # ==========================================
        # СОХРАНЕНИЕ В ТЕКСТОВЫЙ ФАЙЛ (.txt)
        # ==========================================
        txt_filename = f'matched_{x}_result_filtered.txt'
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write(f"Всего совпадений (расстояние < {x} arcsec): {len(matched1)}\n")
            f.write("=" * 100 + "\n\n")
            f.write(table_str)
            f.write("\n")
        print(f"Результат сохранён в текстовый файл: {txt_filename}")
        
        # Сохранение в CSV (оставляем для удобства)
        matched1.to_csv(f'matched_{x}_result_filtered.csv', index=False)
        print(f"Результат также сохранён в CSV файл: matched_{x}_result_filtered.csv")
        
    else:
        print("Совпадений не найдено.")


if __name__ == "__main__":
    main()