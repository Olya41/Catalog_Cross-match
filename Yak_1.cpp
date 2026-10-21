// Локальный кросс-матч с проверкой уникальных SrcID
// Сборка: g++ -O2 -std=c++17 -o crossmatch_local.exe code.cpp
// Запуск: crossmatch_local.exe

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstring>
#include <cctype>
#include <cstdlib>
#include <iomanip>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <chrono>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

const double NO_BOUND = std::nan("");

struct UVOTFilter {
    std::string column;
    double min_val;
    double max_val;
};

// ✏️ ТВОИ ФИЛЬТРЫ
const std::vector<UVOTFilter> UVOT_FILTERS = {
    {"sUVW2",  3.0,       NO_BOUND},
    {"sUVM2",  3.0,       NO_BOUND},
    {"sUVW1",  3.0,       NO_BOUND},
    {"fUVW2",  0.0,       0.0},
    {"fUVM2",  0.0,       0.0},
    {"fUVW1",  0.0,       0.0},
    {"xUVW2",  0.0,       0.0},
    {"xUVM2",  0.0,       0.0},
    {"xUVW1",  0.0,       0.0},
    {"Nd",     1.0,       NO_BOUND},
};

const std::unordered_set<std::string> FILTER_ONLY_COLUMNS = {
    "Nf", "rUVW2", "rUVM2", "rUVW1", "rU", "rB", "rV", "Nd",
    "sUVW2", "sUVM2", "sUVW1", "sU", "sB", "sV",
    "xUVW2", "xUVM2", "xUVW1", "xU", "xB", "xV",
    "fUVW2", "fUVM2", "fUVW1", "fU", "fB", "fV",
};

const long long DEFAULT_CHUNK_SIZE = 500000;

double deg2rad(double d) { return d * M_PI / 180.0; }
double angular_distance_arcsec(double ra1, double dec1, double ra2, double dec2) {
    double ra1r = deg2rad(ra1), dec1r = deg2rad(dec1);
    double ra2r = deg2rad(ra2), dec2r = deg2rad(dec2);
    double cd = std::sin(dec1r) * std::sin(dec2r)
              + std::cos(dec1r) * std::cos(dec2r) * std::cos(ra1r - ra2r);
    cd = std::clamp(cd, -1.0, 1.0);
    return std::acos(cd) * 180.0 / M_PI * 3600.0;
}
std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) ++a;
    while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
    return s.substr(a, b - a);
}
bool is_blank(const std::string& s) {
    for (char c : s) if (!std::isspace((unsigned char)c)) return false;
    return true;
}
double safe_stod(const std::string& s) {
    std::string t = trim(s);
    if (t.empty() || t == "--" || t == "---") return 0.0;
    try { return std::stod(t); } catch (...) { return 0.0; }
}
std::vector<std::string> split_line(const std::string& line, char delim) {
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, delim)) out.push_back(cell);
    return out;
}

struct Target { double ra, dec; int id; };
struct UVOTRow { std::vector<std::string> out_cells; double ra, dec; };
struct TableMeta {
    std::vector<std::string> headers;
    int idx_ra = -1, idx_dec = -1, idx_srcid = -1;
    std::vector<int> filter_indices;
    std::vector<int> output_cols;
    std::vector<std::string> output_names;
};

std::vector<Target> read_targets_csv(const std::string& filename) {
    std::vector<Target> targets;
    std::ifstream f(filename);
    if (!f.is_open()) return targets;
    std::string line;
    std::getline(f, line);
    int id = 0;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        auto parts = split_line(line, ',');
        if (parts.size() < 2) continue;
        Target t;
        t.ra  = safe_stod(parts[0]);
        t.dec = safe_stod(parts[1]);
        t.id  = id++;
        targets.push_back(t);
    }
    return targets;
}

bool read_header(std::ifstream& f, TableMeta& m) {
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        if (line[0] == '#') continue;
        if (line.size() > 3 && line.find_first_not_of("-= ") == std::string::npos) continue;
        if (line.find('\t') == std::string::npos) continue;
        if (line.find("RAJ2000") == std::string::npos) continue;

        m.headers = split_line(line, '\t');
        for (auto& h : m.headers) h = trim(h);

        auto find_col = [&](const std::string& name) -> int {
            for (size_t i = 0; i < m.headers.size(); ++i) if (m.headers[i] == name) return (int)i;
            return -1;
        };

        m.idx_ra    = find_col("RAJ2000");
        m.idx_dec   = find_col("DEJ2000");
        m.idx_srcid = find_col("SrcID");
        if (m.idx_ra < 0 || m.idx_dec < 0 || m.idx_srcid < 0) return false;

        m.filter_indices.clear();
        for (const auto& ftr : UVOT_FILTERS) m.filter_indices.push_back(find_col(ftr.column));

        m.output_cols.clear();
        m.output_names.clear();
        for (size_t i = 0; i < m.headers.size(); ++i) {
            if (FILTER_ONLY_COLUMNS.count(m.headers[i])) continue;
            m.output_cols.push_back((int)i);
            m.output_names.push_back(m.headers[i]);
        }
        return true;
    }
    return false;
}

bool passes_filters(const std::vector<std::string>& cells, const std::vector<int>& filter_indices) {
    for (size_t k = 0; k < UVOT_FILTERS.size(); ++k) {
        int idx = filter_indices[k];
        if (idx < 0 || idx >= (int)cells.size()) return false;
        const std::string& cell = cells[idx];
        if (is_blank(cell)) return false;
        double v = safe_stod(cell);
        const auto& f = UVOT_FILTERS[k];
        if (!std::isnan(f.min_val) && v < f.min_val) return false;
        if (!std::isnan(f.max_val) && v > f.max_val) return false;
    }
    return true;
}

// ВНИМАНИЕ: Добавлен битовый массив для SrcID
void process_chunk(std::ifstream& f, const TableMeta& m, long long chunk_size,
                   std::vector<UVOTRow>& chunk_rows, long long& read_lines, bool& reached_eof,
                   std::vector<bool>& seen_srcid, long long& unique_passed) {
    chunk_rows.clear();
    chunk_rows.reserve(chunk_size / 2);
    read_lines = 0;
    reached_eof = false;

    std::string line;
    while (read_lines < chunk_size) {
        if (!std::getline(f, line)) { reached_eof = true; break; }
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;

        auto cells = split_line(line, '\t');
        if (f.eof() && cells.size() < m.headers.size() - 2) { reached_eof = true; break; }
        if (m.idx_ra >= (int)cells.size()) continue;

        std::string ra_s = trim(cells[m.idx_ra]);
        char* end = nullptr;
        std::strtod(ra_s.c_str(), &end);
        if (end == ra_s.c_str() || *end != '\0') continue;

        read_lines++;
        while (cells.size() < m.headers.size()) cells.push_back("");
        if (cells.size() > m.headers.size()) cells.resize(m.headers.size());

        if (!passes_filters(cells, m.filter_indices)) continue;

        int srcid = (int)safe_stod(cells[m.idx_srcid]);
        if (srcid > 0 && srcid < (int)seen_srcid.size()) {
            if (!seen_srcid[srcid]) {
                seen_srcid[srcid] = true;
                unique_passed++;
            }
        }

        UVOTRow row;
        row.ra  = safe_stod(cells[m.idx_ra]);
        row.dec = safe_stod(cells[m.idx_dec]);
        row.out_cells.reserve(m.output_cols.size());
        for (int c : m.output_cols) row.out_cells.push_back(cells[c]);
        chunk_rows.push_back(std::move(row));
    }
}

int main(int argc, char** argv) {
    long long start_chunk = 0;
    long long chunk_size  = DEFAULT_CHUNK_SIZE;
    if (argc >= 2) start_chunk = std::atoll(argv[1]);
    if (argc >= 3) chunk_size  = std::atoll(argv[2]);
    if (chunk_size <= 0) chunk_size = DEFAULT_CHUNK_SIZE;

    const std::string uvot_file = "C:/Users/yakun/uvot_full.tsv";     
    const std::string target_file = "Coordinate_Data.csv";
    const std::string out_file    = "uvot_crossmatch_local.csv";
    const double radius_arcsec    = 10.0;

    auto t0 = std::chrono::steady_clock::now();

    std::cout << "Reading " << target_file << " ...\n";
    auto targets = read_targets_csv(target_file);
    if (targets.empty()) { std::cerr << "No targets.\n"; return 1; }
    std::cout << "  Targets: " << targets.size() << "\n\n";

    std::ifstream f(uvot_file);
    if (!f.is_open()) { std::cerr << "Cannot open: " << uvot_file << "\n"; return 1; }

    TableMeta meta;
    if (!read_header(f, meta)) { std::cerr << "Header failed.\n"; return 1; }

    // Создаем битовый массив на 7 млн (около 875 КБ памяти)
    const int MAX_SRCID = 7000000;
    std::vector<bool> seen_srcid(MAX_SRCID, false);
    long long unique_passed = 0;

    std::ofstream out(out_file);
    out << "target_id,target_ra,target_dec,angDist_arcsec";
    for (auto& n : meta.output_names) out << "," << n;
    out << "\n" << std::fixed << std::setprecision(6);

    int chunk_no = 0;
    int matches_total = 0;
    bool eof = false;

    while (!eof) {
        if (chunk_no < start_chunk) {
            std::vector<UVOTRow> skip;
            long long rl;
            process_chunk(f, meta, chunk_size, skip, rl, eof, seen_srcid, unique_passed);
            chunk_no++;
            continue;
        }

        std::vector<UVOTRow> chunk;
        long long rl;
        process_chunk(f, meta, chunk_size, chunk, rl, eof, seen_srcid, unique_passed);
        if (rl == 0 && chunk.empty()) { chunk_no++; break; }

        double cell = radius_arcsec / 3600.0;
        auto hash_cell = [](int ix, int iy) -> long long {
            return (static_cast<long long>(ix) << 32) ^ (iy & 0xffffffffLL);
        };
        std::unordered_map<long long, std::vector<int>> grid;
        grid.reserve(chunk.size() * 2);
        for (int i = 0; i < (int)chunk.size(); ++i) {
            int ix = (int)std::floor(chunk[i].ra  / cell);
            int iy = (int)std::floor(chunk[i].dec / cell);
            grid[hash_cell(ix, iy)].push_back(i);
        }

        int chunk_matches = 0;
        for (int t_i = 0; t_i < (int)targets.size(); ++t_i) {
            const auto& t = targets[t_i];
            int ix = (int)std::floor(t.ra  / cell);
            int iy = (int)std::floor(t.dec / cell);

            int best_idx = -1;
            double best_d = radius_arcsec;

            for (int dx = -1; dx <= 1; ++dx)
                for (int dy = -1; dy <= 1; ++dy) {
                    auto it = grid.find(hash_cell(ix + dx, iy + dy));
                    if (it == grid.end()) continue;
                    for (int j : it->second) {
                        double d = angular_distance_arcsec(t.ra, t.dec, chunk[j].ra, chunk[j].dec);
                        if (d < best_d) { best_d = d; best_idx = j; }
                    }
                }

            if (best_idx >= 0) {
                const auto& u = chunk[best_idx];
                out << t.id << "," << t.ra << "," << t.dec << "," << best_d;
                for (auto& c : u.out_cells) out << "," << c;
                out << "\n";
                chunk_matches++;
            }
        }
        matches_total += chunk_matches;

        std::cout << "Chunk " << chunk_no << ": read=" << rl << ", passed=" << chunk.size()
                  << ", matches=" << chunk_matches << (eof ? " [EOF]" : "") << "\n";
        chunk_no++;
    }

    out.close();
    auto t1 = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "----------------------------------------\n";
    std::cout << "Done.\n";
    std::cout << "  Targets processed:          " << targets.size() << "\n";
    std::cout << "  Targets with match:         " << matches_total << "\n";
    std::cout << "  UNIQUE SrcIDs passed filters: " << unique_passed << "\n"; // <--- ДОКАЗАТЕЛЬСТВО
    std::cout << "  Total time:                 " << dt << " s\n";
    std::cout << "  Output:                     " << out_file << "\n";
    return 0;
}