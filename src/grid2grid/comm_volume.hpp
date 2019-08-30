
#include <unordered_map>

namespace grid2grid {
struct edge_t {
    int src;
    int dest;

    edge_t = default;
    edge_t(int src, int dest):
        src(src), dest(dest) {}
    edge_t(edge_t& e) src(e.src), dest(e.dest) {}

    bool operator==(const edge_t& other) const {
        return src==other.src && dest==other.dest;
    }
};

struct weighted_edge_t {
    edge_t e;
    int w;

    weighted_edge_t = default;
    weighted_edge_t(edge_t& e, int weight):
        e(e), w(weight) {}
    weighted_edge_t(weighted_edge_t& we):
        e(we.edge()), w(we.weight()) {}

    int weight() const {
        return w;
    }

    int src() const {
        return e.src;
    }

    int dest() const {
        return e.dest;
    }

    edge_t edge() const {
        return e;
    }

    bool operator==(const weighted_edge& other) const {
        return e==other.edge() && w == other.weight();
    }
};

struct comm_volume {
    using volume_t = std::unordered_map<edge, int>;
    volume_t volume;

    comm_volume = default;
    comm_volume(volume_t&& v):
        volume(std::forward<volume_t>(v)) {}

    comm_volume& operator+=(const comm_volume& other) {
        for (const auto& vol : other.volume) {
            edge e = vol.first;
            int w = vol.second;
            volume[e] += w;
        }
        return *this;
    }

    comm_volume operator+(const comm_volume& other) const {
        volume_t sum_comm_vol;
        for (const auto& vol : volume) {
            edge e = vol.first;
            int w = vol.second;
            sum_comm_vol[e] += w;
        }
        for (const auto& vol : other.volume) {
            edge e = vol.first;
            int w = vol.second;
            sum_comm_vol[e] += w;
        }
        return sum_comm_vol;
    }
};
}


